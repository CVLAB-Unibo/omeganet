# Copyright 2020 Fabio Tosi, Filippo Aleotti, Pierluigi Zama Ramirez, Matteo Poggi,
# Samuele Salti, Luigi Di Stefano, Stefano Mattoccia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from networks.ops import *
from helpers.bilinear_sampler import *

NUM_FEATURES = 16
FLOW_SCALING = 0.1
DISP_SCALING = 10.0
MIN_DISP = 0.01
POSE_SCALING = 0.01


def feature_extractor(src_img_1, tgt_img, src_img_2, is_training, name=None):
    """Features extractor
        :param src_img_1: image at time t-1. Tensor with shape [1,H,W,3], dtype=tf.float32
        :param tgt_img: image at time t. Tensor with shape [1,H,W,3], dtype=tf.float32
        :param src_img_2: image at time t+1. Tensor with shape [1,H,W,3], dtype=tf.float32
        :param is_training: training flag. For batchnorm
        :param name: name of the extractor. If name is not None, the name will be feature_extractor_NAME
    """
    batch_norm_params = {"is_training": is_training}
    final_name = "feature_extractor"
    if name is not None:
        final_name = "{}_{}".format(final_name, name)
    with tf.variable_scope(final_name):
        pyramid_src_img_1 = build_pyramid(
            src_img_1, normalizer_params=batch_norm_params
        )
        pyramid_tgt_img = build_pyramid(tgt_img, normalizer_params=batch_norm_params)
        pyramid_src_img_2 = build_pyramid(
            src_img_2, normalizer_params=batch_norm_params
        )
    return pyramid_src_img_1, pyramid_tgt_img, pyramid_src_img_2


def CameraNet(features, is_training):
    """CameraNet
        It estimates both the pose and camera intrinsics.
        :param features: list of features from [src1, tgt, src2]
        :param is_training: training flag. For batchnorm

        :return pose_final: tensor with shape (1, 2, 6)
        :return intrinsics_mat: tensor with shape (1, 1, 3, 3)
    """
    with tf.variable_scope("pose_net"):
        batch_norm_params = {"is_training": is_training}

        pyramid_src_img_1 = features[0]
        pyramid_tgt_img = features[1]
        pyramid_src_img_2 = features[2]
        input_batch = tf.concat(
            [pyramid_src_img_1[4], pyramid_tgt_img[4], pyramid_src_img_2[4]], axis=3
        )

        with tf.variable_scope("conv1_a"):
            conv1_a = conv2d(
                input_batch,
                NUM_FEATURES * 8,
                3,
                1,
                normalizer_params=batch_norm_params,
                activation_fn=tf.nn.relu,
            )
        with tf.variable_scope("conv1_b"):
            conv1_b = conv2d(
                conv1_a,
                NUM_FEATURES * 8,
                3,
                2,
                normalizer_params=batch_norm_params,
                activation_fn=tf.nn.relu,
            )
        with tf.variable_scope("conv2_a"):
            conv2_a = conv2d(
                conv1_b,
                NUM_FEATURES * 16,
                3,
                1,
                normalizer_params=batch_norm_params,
                activation_fn=tf.nn.relu,
            )
        with tf.variable_scope("conv2_b"):
            conv2_b = conv2d(
                conv2_a,
                NUM_FEATURES * 16,
                3,
                2,
                normalizer_params=batch_norm_params,
                activation_fn=tf.nn.relu,
            )

        # POSE ESTIMATOR
        with tf.variable_scope("pred"):
            pose_pred = conv2d(
                conv2_b, 12, 1, 1, normalizer_fn=None, activation_fn=None
            )
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            pose_final = POSE_SCALING * tf.reshape(pose_avg, [-1, 2, 6])

        # INTRINSIC ESTIMATOR
        s = tf.shape(pyramid_tgt_img[0])
        h = tf.to_float(s[1])
        w = tf.to_float(s[2])
        intrinsics_mat = _estimate_intrinsics(conv2_b, w, h)

        return pose_final, intrinsics_mat


def _estimate_intrinsics(bottleneck, image_width, image_height):
    """Estimate intrinsic
    :param bottleneck: feature bottleneck tensor
    :param image_width: width of the resized image
    :param image_height: height of the resized image
    
    :return intrinsic_mat: tensor with shape (1, 1, 3, 3)
    """
    with tf.variable_scope("intrinsics"):
        bottleneck = tf.reduce_mean(bottleneck, axis=[1, 2], keepdims=True)
        focal_lengths = tf.squeeze(
            tf.contrib.layers.conv2d(
                bottleneck,
                2,
                [1, 1],
                stride=1,
                activation_fn=tf.nn.softplus,
                weights_regularizer=None,
                scope="foci",
            ),
            axis=(1, 2),
        ) * tf.to_float(tf.convert_to_tensor([[image_width, image_height]]))

        offsets = (
            tf.squeeze(
                tf.contrib.layers.conv2d(
                    bottleneck,
                    2,
                    [1, 1],
                    stride=1,
                    activation_fn=None,
                    weights_regularizer=None,
                    biases_initializer=None,
                    scope="offsets",
                ),
                axis=(1, 2),
            )
            + 0.5
        ) * tf.to_float(tf.convert_to_tensor([[image_width, image_height]]))

        foci = tf.linalg.diag(focal_lengths)
        intrinsic_mat = tf.concat([foci, tf.expand_dims(offsets, -1)], axis=2)
        batch_size = tf.shape(bottleneck)[0]
        last_row = tf.tile([[[0.0, 0.0, 1.0]]], [batch_size, 1, 1])
        intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)
        intrinsic_mat = tf.expand_dims(intrinsic_mat, axis=1)
        return intrinsic_mat


def DSNet(pyramid_tgt_img, classes, is_training):
    """DSNet
    """
    with tf.variable_scope("monocular_depthnet", reuse=tf.AUTO_REUSE):

        batch_norm_params = {"is_training": is_training}

        # SCALE 5
        with tf.variable_scope("L5"):
            with tf.variable_scope("estimator"):
                conv5 = build_estimator(
                    pyramid_tgt_img[5], normalizer_params=batch_norm_params
                )
            with tf.variable_scope("disparity"):
                disp5 = get_disp(conv5, normalizer_params=batch_norm_params)
                updisp5 = depth_upsampling(disp5, 1)
            with tf.variable_scope("upsampler"):
                upconv5 = bilinear_upsampling_by_convolution(
                    conv5, 2, normalizer_params=batch_norm_params
                )
        # SCALE 4
        with tf.variable_scope("L4"):
            with tf.variable_scope("estimator"):
                conv4 = build_estimator(
                    pyramid_tgt_img[4], upconv5, normalizer_params=batch_norm_params
                )
            with tf.variable_scope("disparity"):
                disp4 = (
                    get_disp(conv4, normalizer_params=batch_norm_params) + updisp5[0]
                )
                updisp4 = depth_upsampling(disp4, 1)
            with tf.variable_scope("upsampler"):
                upconv4 = bilinear_upsampling_by_convolution(
                    conv4, 2, normalizer_params=batch_norm_params
                )
        # SCALE 3
        with tf.variable_scope("L3"):
            with tf.variable_scope("estimator"):
                conv3 = build_estimator(
                    pyramid_tgt_img[3], upconv4, normalizer_params=batch_norm_params
                )
            with tf.variable_scope("disparity"):
                disp3 = (
                    get_disp(conv3, normalizer_params=batch_norm_params) + updisp4[0]
                )
                updisp3 = depth_upsampling(disp3, 1)
            with tf.variable_scope("upsampler"):
                upconv3 = bilinear_upsampling_by_convolution(
                    conv3, 2, normalizer_params=batch_norm_params
                )
        # SCALE 2
        with tf.variable_scope("L2"):
            with tf.variable_scope("estimator"):
                conv2 = build_estimator(
                    pyramid_tgt_img[2], upconv3, normalizer_params=batch_norm_params
                )
            with tf.variable_scope("disparity"):
                disp2 = (
                    get_disp(conv2, normalizer_params=batch_norm_params) + updisp3[0]
                )
                updisp2 = depth_upsampling(disp2, 1)
            with tf.variable_scope("upsampler"):
                upconv2 = bilinear_upsampling_by_convolution(
                    conv2, 2, normalizer_params=batch_norm_params
                )
        # SCALE 1
        with tf.variable_scope("L1"):
            with tf.variable_scope("estimator"):
                conv1 = build_estimator(
                    pyramid_tgt_img[1], upconv2, normalizer_params=batch_norm_params
                )
            with tf.variable_scope("disparity"):
                disp1 = (
                    get_disp(conv1, normalizer_params=batch_norm_params) + updisp2[0]
                )

            with tf.variable_scope("semantic"):
                sem1 = get_semantic(conv1, classes, normalizer_params=batch_norm_params)

        return [disp1, disp2, disp3, disp4, disp5], sem1


def build_pyramid(input_batch, normalizer_params=None, scope="img_pyramid"):
    """Pyramidal feature extractor
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        features = []
        features.append(input_batch)

        with tf.variable_scope("conv1a"):
            conv1a = conv2d(
                input_batch, NUM_FEATURES, 3, 2, normalizer_params=normalizer_params
            )
        with tf.variable_scope("conv1b"):
            conv1b = conv2d(
                conv1a, NUM_FEATURES, 3, 1, normalizer_params=normalizer_params
            )
            features.append(conv1b)
        with tf.variable_scope("conv2a"):
            conv2a = conv2d(
                conv1b, NUM_FEATURES * 2, 3, 2, normalizer_params=normalizer_params
            )
        with tf.variable_scope("conv2b"):
            conv2b = conv2d(
                conv2a, NUM_FEATURES * 2, 3, 1, normalizer_params=normalizer_params
            )
            features.append(conv2b)
        with tf.variable_scope("conv3a"):
            conv3a = conv2d(
                conv2b, NUM_FEATURES * 4, 3, 2, normalizer_params=normalizer_params
            )
        with tf.variable_scope("conv3b"):
            conv3b = conv2d(
                conv3a, NUM_FEATURES * 4, 3, 1, normalizer_params=normalizer_params
            )
            features.append(conv3b)
        with tf.variable_scope("conv4a"):
            conv4a = conv2d(
                conv3b, NUM_FEATURES * 8, 3, 2, normalizer_params=normalizer_params
            )
        with tf.variable_scope("conv4b"):
            conv4b = conv2d(
                conv4a, NUM_FEATURES * 8, 3, 1, normalizer_params=normalizer_params
            )
            features.append(conv4b)
        with tf.variable_scope("conv5a"):
            conv5a = conv2d(
                conv4b, NUM_FEATURES * 16, 3, 2, normalizer_params=normalizer_params
            )
        with tf.variable_scope("conv5b"):
            conv5b = conv2d(
                conv5a, NUM_FEATURES * 16, 3, 1, normalizer_params=normalizer_params
            )
            features.append(conv5b)
        return features


def build_estimator(features, upsampled_disp=None, normalizer_params=None):
    """Single scale estimator
    """
    with tf.variable_scope("build_estimator"):
        if upsampled_disp is not None:
            disp2 = tf.concat([features, upsampled_disp], -1)
        else:
            disp2 = features
        with tf.variable_scope("disp-3"):
            disp3 = conv2d(
                disp2, NUM_FEATURES * 4, 3, 1, normalizer_params=normalizer_params
            )
        with tf.variable_scope("disp-4"):
            disp4 = conv2d(
                disp3, NUM_FEATURES * 3, 3, 1, normalizer_params=normalizer_params
            )
        with tf.variable_scope("disp-5"):
            disp5 = conv2d(
                disp4, NUM_FEATURES * 2, 3, 1, normalizer_params=normalizer_params
            )
        with tf.variable_scope("disp-6"):
            disp6 = conv2d(
                disp5, NUM_FEATURES, 3, 1, normalizer_params=normalizer_params
            )
        return disp6


def get_disp(x, normalizer_params=None, rates=[1, 1]):
    """Disparity prediction layer
    """
    with tf.variable_scope("disparity_estimator"):
        with tf.variable_scope("conv1"):
            conv1 = conv2d(
                x, NUM_FEATURES * 4, 3, 1, normalizer_params=normalizer_params
            )
        with tf.variable_scope("conv2"):
            conv2 = conv2d(
                conv1,
                NUM_FEATURES * 2,
                3,
                1,
                normalizer_params=normalizer_params,
                rate=rates[0],
            )
        with tf.variable_scope("conv3"):
            conv3 = conv2d(
                conv2,
                NUM_FEATURES,
                3,
                1,
                normalizer_params=normalizer_params,
                rate=rates[1],
            )
        with tf.variable_scope("disparity"):
            disparity = (
                DISP_SCALING
                * conv2d(
                    conv3, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None
                )
                + MIN_DISP
            )
        return disparity


def get_semantic(x, classes, normalizer_params=None, rates=[1, 1]):
    """Semantic estimator layer
    """
    with tf.variable_scope("semantic_estimator"):
        with tf.variable_scope("conv1"):
            conv1 = conv2d(
                x, NUM_FEATURES * 4, 3, 1, normalizer_params=normalizer_params
            )
        with tf.variable_scope("conv2"):
            conv2 = conv2d(
                conv1,
                NUM_FEATURES * 2,
                3,
                1,
                normalizer_params=normalizer_params,
                rate=rates[0],
            )
        with tf.variable_scope("conv3"):
            conv3 = conv2d(
                conv2,
                NUM_FEATURES,
                3,
                1,
                normalizer_params=normalizer_params,
                rate=rates[1],
            )
        with tf.variable_scope("disparity"):
            sem = conv2d(conv3, classes, 3, 1, normalizer_params=normalizer_params)
        return sem
