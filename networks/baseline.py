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


"""
Baseline network
We learn to predict depth, pose of the camera, intrinsics and the semantic
"""

import tensorflow as tf
import os
from networks.ops import *
from networks.general_network import GeneralNetwork
from networks.network_components import *


class BaselineNet(GeneralNetwork):
    """Baseline network, w/o OFNet and SD-OFNet.
        It contains DSNet and CameraNet
    """

    def __init__(self, batch, is_training, params):
        """BaselineNet constructor:
            :param batch: input of the network. Dictionary
            :param is_training: training flag. For batchnorm
            :params: network settings
        """
        super(BaselineNet, self).__init__(batch, is_training, params)
        self.name = "MultiViewNetwork"
        self.depth_tgt = None
        self.disp_tgt = None
        self.semantic_tgt = None

    def get_features(self, src_img_1, tgt_img, src_img_2, is_training, scope):
        """Extract features from images
            :param src_img_1: tensor with src1 image, (B,H,W,3)
            :param tgt_img: tensor with tgt image, (B,H,W,3)
            :param src_img_2: tensor with src1 image, (B,H,W,3)
            :param is_training: training flag. For batchnorm
            :param scope: name used in the feature extractor
            :return features: list of extracted features
        """
        return feature_extractor(src_img_1, tgt_img, src_img_2, is_training, scope)

    def get_DSNet(self, features, classes, is_training):
        """Build DSNet, in charge of depth and semantic estimation
            :return DSNet: DSNet network
        """
        return DSNet(features, classes, is_training)

    def get_CameraNet(self, src_img_1, tgt_img, src_img_2, is_training, scope="pose"):
        """Build CameraNet, in charge of pose and intrinsic estimation
            :return CameraNet: CameraNet network
        """
        features = self.get_features(src_img_1, tgt_img, src_img_2, is_training, scope)
        return CameraNet(features, self.is_training)

    def disp_normalize(self, disp):
        """Apply spatial normalizer defined in 
            :param disp: disparity (inverse depth)
            :return normalized_disp: tensor with same shape of disp
        """
        with tf.variable_scope("disp_normalize"):
            return spatial_normalize(disp)

    def disp2depth(self, disp):
        """Turn disparity into depth
            :param disp: disparity (inverse depth)
            :return depth: tensor with same shape of disp
        """
        with tf.variable_scope("disp2depth"):
            return 1.0 / disp

    def get_rigid_flow(self, depth, pose, intrinsics, pose_index, reversed_pose):
        """
            Get rigid flow using depth and pose projection
            :param depth: depth estimated by DSNet. Tensor with shape (B,H,W)
            :param pose: pose estimated by CameraNet. Tensor with shape (1,2,6)
            :param pose_index: index of pose to use
            :param reversed_pose: if True, use reversed pose
            :return rigid flow: BxHxWx2 rigid optical flow
            :raise ValueError: if pose_index is not in [0,1]
        """
        with tf.variable_scope("get_rigid_flow"):
            if pose_index not in [0, 1]:
                raise ValueError("pose index must be in [0,1]")
            rigid_flow = compute_rigid_flow(
                depth, pose[:, pose_index, :], intrinsics[:, 0, :, :], reversed_pose
            )
            return rigid_flow

    def prepare_depth(self, disp):
        """
            Turn disp into depth
            :param disp: tensor with disparity estimations
        """
        with tf.variable_scope("prepare_depth"):
            normalized = tf.image.resize_bilinear(
                self.disp_normalize(disp), [self.h, self.w]
            )
            depth = self.disp2depth(normalized)
            depth.set_shape([None, self.params.height, self.params.width, 1])
            depth = tf.squeeze(depth, axis=3)
            return depth

    def prepare_disp(self, disp):
        """ First, normalization is applied to disp, then the result is
            upsampled to (self.params.height, self.params.width).
            :param disp: tensor with shape (B,H,W)
            :return upsampled_normalized_disp: tensor with shape (B, self.params.height, self.params.width)
        """
        with tf.variable_scope("prepare_disp"):
            disp = tf.image.resize_bilinear(self.disp_normalize(disp), [self.h, self.w])
            disp.set_shape([None, self.params.height, self.params.width, 1])
            return disp

    def upsample_semantic(self, semantic):
        """Upsample semantic to [self.params.height,self.params.width]
            :param semantic: tensor with logits or semantic labels
        """
        with tf.variable_scope("upsample_semantic"):
            semantic = tf.image.resize_images(
                semantic, [self.params.height, self.params.width]
            )
            return semantic

    def build_network(self):
        """Build baseline network,
            composed of DSNet and CameraNet
        """
        with tf.variable_scope(self.name):

            self.features = self.get_features(
                self.src_img_1,
                self.tgt_img,
                self.src_img_2,
                self.is_training,
                scope=None,
            )
            self.pred_disp_tgt, self.pred_semantic_logits_tgt = self.get_DSNet(
                self.features[1], self.classes, self.is_training
            )
            print(" [*] Building DSNet: SUCCESS")

            self.pose, self.intrinsics = self.get_CameraNet(
                self.src_img_1, self.tgt_img, self.src_img_2, self.is_training
            )
            print(" [*] Building CameraNet: SUCCESS")

    def build_outputs(self):
        """ Output generated by the network.
            Attributes semantic_tgt, depth_tgt and disp_tgt are updated
        """
        with tf.variable_scope("build_baseline_outputs"):
            self.semantic_tgt = self.upsample_semantic(self.pred_semantic_logits_tgt)
            self.depth_tgt = self.prepare_depth(self.pred_disp_tgt[0])
            self.disp_tgt = self.prepare_disp(self.pred_disp_tgt[0])

    def get_network_params(self):
        """Get network variables to load.
            This function is valid only in the case test, since
            no Adam state is loaded and training from scratch
            is not supported.
            Note that also Batchnorm params are loaded
            """
        with tf.variable_scope("get_network_params"):
            var = [x for x in tf.trainable_variables() if self.name in x.name]
            batch_norm_variables = [
                x
                for x in tf.all_variables()
                if "moving_mean" in x.name or "moving_variance" in x.name
            ]
            var += batch_norm_variables
            return var
