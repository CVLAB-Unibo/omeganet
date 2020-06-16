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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.contrib.slim as slim

import tensorflow as tf


def upsample_nn(x, ratio):
    s = x.get_shape().as_list()
    h = s[1]
    w = s[2]
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])


def conv2d(
    inputs,
    num_outputs,
    kernel_size,
    stride,
    normalizer_fn=slim.batch_norm,
    activation_fn=tf.nn.relu,
    weights_regularizer=slim.l2_regularizer(0.0001),
    normalizer_params=True,
    padding=(1, 1),
    reflect=True,
    rate=1,
):

    if rate > 1:
        w_pad, h_pad = (rate, rate)
    else:
        w_pad, h_pad = tuple(padding)

    if reflect:
        inputs = tf.pad(
            inputs, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], "REFLECT"
        )

    return tf.contrib.layers.conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride,
        padding="VALID",
        normalizer_fn=normalizer_fn,
        activation_fn=activation_fn,
        weights_regularizer=weights_regularizer,
        normalizer_params=normalizer_params,
        rate=rate,
    )


def upconv(
    inputs,
    num_outputs,
    kernel_size,
    stride,
    normalizer_fn=slim.batch_norm,
    activation_fn=tf.nn.relu,
    weights_regularizer=slim.l2_regularizer(0.0001),
    normalizer_params=True,
    padding=(1, 1),
):
    upsample = upsample_nn(inputs, stride)
    return conv2d(
        upsample,
        num_outputs,
        kernel_size,
        1,
        padding=padding,
        normalizer_fn=normalizer_fn,
        activation_fn=activation_fn,
        weights_regularizer=weights_regularizer,
        normalizer_params=normalizer_params,
    )


def gradient_x(img):
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx


def gradient_y(img):
    gy = img[:, :-1, :, :] - img[:, 1:, :, :]
    return gy


def L2_norm(x, axis=3, keepdims=True):
    curr_offset = 1e-10
    l2_norm = tf.norm(tf.abs(x) + curr_offset, axis=axis, keepdims=keepdims)
    return l2_norm


def spatial_normalize(disp):
    with tf.variable_scope("spatial_normalizer"):
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1, 2, 3], keepdims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp / disp_mean


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def reduce_mean_masked(tensor, mask):
    with tf.variable_scope("reduce_mean_masked"):
        valid_points = tf.maximum(tf.reduce_sum(mask), 1)
        loss = tf.reduce_sum(tensor * mask) / valid_points
        return loss


def reduce_mean_probability_masked(tensor, mask, probability):
    with tf.variable_scope("reduce_mean_masked"):
        valid_points = tf.maximum(tf.reduce_sum(mask), 1)
        loss = tf.reduce_sum(tensor * mask * probability) / valid_points
        return loss


# Upsampling layer
def bilinear_upsampling_by_convolution(x, stride, normalizer_params=None):
    with tf.variable_scope("bilinear_upsampling_by_convolution"):
        f = x.get_shape().as_list()[-1]
        return upconv(x, f, 3, stride, normalizer_params=normalizer_params)


def depth_upsampling(x, scales):
    with tf.variable_scope("depth_upsampling"):
        features = []
        for i in range(1, scales + 1):
            with tf.variable_scope("upsampler_pred_" + str(i)):
                up = tf.image.resize_bilinear(
                    x,
                    [
                        x.get_shape().as_list()[1] * (2 ** i),
                        x.get_shape().as_list()[2] * (2 ** i),
                    ],
                )
                features.append(up)
        return features


def stop_features_gradient(features):
    with tf.variable_scope("stop_features_gradient"):
        new_features = []
        for img_x_features in features:
            new_img_x_features = []
            for feat in img_x_features:
                new_img_x_features.append(tf.stop_gradient(feat))
            new_features.append(new_img_x_features)
        return new_features


def couple_imgs_features(features):
    with tf.variable_scope("couple_imgs_features"):
        coupled_features = []
        for tgt_feat, src2_feat in zip(features[1], features[2]):
            couple_feat = tf.concat([tgt_feat, src2_feat], axis=-1)
            coupled_features.append(couple_feat)
        return coupled_features
