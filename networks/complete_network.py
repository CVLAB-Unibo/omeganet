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
Complete OmegaNet
"""
import tensorflow as tf
import os

from networks.general_network import GeneralNetwork
from networks.baseline import BaselineNet
from helpers import bilinear_sampler
from networks.selflow.selflow_network import flownet
from helpers.utilities import extract_semantic_priors


class OmegaNet(GeneralNetwork):
    """OmegaNet. It contains DSNet, CameraNet and SD-OFNet
    """

    def __init__(self, batch, is_training, params):
        """OmegaNet constructor:
            :param batch: input of the network. Dictionary
            :param is_training: training flag. For batchnorm
            :params: network settings
        """
        super(OmegaNet, self).__init__(batch, is_training, params)
        self.name = "OmegaNet"
        self.disp = None
        self.optical_flow = None
        self.semantic_logits = None
        self.motion_mask = None

    def build_network(self):
        """Build OmegaNet: first, DSNet and CameraNet are instantiated,
            then SD-OFNet
        """
        self.baselineNet = BaselineNet(self.batch, self.is_training, self.params)
        self.baselineNet.build_network()
        self.baselineNet.build_outputs()

        # prepare semantic stuff
        self.semantic_logits = self.baselineNet.pred_semantic_logits_tgt
        self.__semantic = self.prepare_semantic(self.semantic_logits)
        self.__priors = extract_semantic_priors(self.__semantic)
        self.__dynamic_tgt_mask, self.__static_tgt_mask = self.build_semantic_masks()

        # get rigid flow using depth and pose
        self.__sflow_src2_tgt = self.baselineNet.get_rigid_flow(
            self.baselineNet.depth_tgt,
            self.baselineNet.pose,
            self.baselineNet.intrinsics,
            pose_index=1,
            reversed_pose=False,
        )

        # self-distilled optical flow network
        load_flow = not self.params.load_only_baseline
        self.__optical_flow_src2_tgt, _ = flownet(
            self.tgt_img.shape,
            self.src_img_1,
            self.tgt_img,
            self.src_img_2,
            train=False,
            trainable=load_flow,
            reuse=tf.AUTO_REUSE,
            regularizer=None,
            is_scale=True,
            scope="superflow",
        )

    def prepare_final_motion_mask(self):
        """
            :return final_motion_mask: motion binary mask. 1 if pixel is moving
        """
        moving_src2_tgt = self.build_moving_probability_mask(
            self.__optical_flow_src2_tgt, self.__sflow_src2_tgt
        )
        final_motion_mask = self.__dynamic_tgt_mask * tf.where(
            moving_src2_tgt > self.params.tau,
            tf.ones_like(moving_src2_tgt),
            tf.zeros_like(moving_src2_tgt),
        )
        return final_motion_mask

    def prepare_semantic(self, logits, height=None, width=None):
        """Extract semantic map from logits.
            :param logits: semantic logits
            :param height: height of image. Optional (default is params.height)
            :param width: width of image. Optional (default is params.width)
        """
        with tf.variable_scope("prepare_semantic"):
            if height is None:
                height = self.params.height
            if width is None:
                width = self.params.width
            logits = tf.image.resize_images(logits, [height, width])
            semantic = tf.argmax(logits, axis=-1)
            semantic = tf.expand_dims(semantic, -1)
            semantic = tf.cast(semantic, tf.float32)
            return semantic

    def build_outputs(self):
        """Build outputs of the network
        """
        with tf.variable_scope("build_outputs"):

            self.optical_flow = self.__optical_flow_src2_tgt
            self.disp = self.baselineNet.disp_tgt
            self.semantic = self.__semantic
            self.motion_mask = self.prepare_final_motion_mask()

    def tf_cosine_distance(self, a, b):
        """Measure cosine distance between a and b
            :param a: tensor
            :param b: tensor
            :return cosine similarity
        """
        normalize_a = tf.nn.l2_normalize(a, -1)
        normalize_b = tf.nn.l2_normalize(b, -1)
        cos_similarity = tf.reduce_sum(
            tf.multiply(normalize_a, normalize_b), axis=-1, keep_dims=True
        )
        return (1.0 - cos_similarity) / 2.0

    def get_occlusion_mask_from_rigid_flow(self, rigid_flow):
        """Prepare occlusion mask due to rigid motion
            :param rigid_flow: Tensor with rigid flow
            :return mask: mask of occlusions due to rigid camera motion
        """
        with tf.variable_scope("get_occlusion_mask_from_rigid_flow"):
            b, h, w, _ = rigid_flow.shape
            rigid_flow = tf.stop_gradient(rigid_flow)
            mask = bilinear_sampler.flow_warp(
                tf.ones([b, h, w, 1], dtype=tf.float32), rigid_flow
            )
            mask = tf.clip_by_value(mask, 0.0, 1.0)
            return mask

    def build_moving_probability_mask(self, optical_flow, rigid_flow):
        """
            Masks of moving objects
            If the object is moving, this value should be low.
        """
        with tf.variable_scope("build_moving_probability_mask"):
            epsylon = 1e-7
            optical_flow = tf.stop_gradient(optical_flow)
            rigid_flow = tf.stop_gradient(rigid_flow)
            normalized_optical_flow = tf.norm(
                optical_flow, axis=-1, keep_dims=True, name="optical_flow_norm"
            )
            normalized_rigid_flow = tf.norm(
                rigid_flow, axis=-1, keep_dims=True, name="rigid_flow_norm"
            )
            cosine_distance = self.tf_cosine_distance(optical_flow, rigid_flow)
            ratio = (
                epsylon + tf.minimum(normalized_optical_flow, normalized_rigid_flow)
            ) / (epsylon + tf.maximum(normalized_optical_flow, normalized_rigid_flow))
            ratio_distance = 1.0 - ratio
            moving_probability = tf.maximum(cosine_distance, ratio_distance)
            return moving_probability

    def get_network_params(self):
        """Load network params.
            In particular, OmegaNet relies on DSNet, Camnet and self-distilled OFNet
        """
        with tf.variable_scope("get_network_params"):
            baseline_vars = self.baselineNet.get_network_params()
            reflownet_vars = [
                x for x in tf.trainable_variables() if "superflow" in x.name
            ]
            return baseline_vars + reflownet_vars

    def build_semantic_masks(self):
        """
           Prepare masks based on semantic priors
           :return dynamic_tgt_mask: mask of potentially dinamyc objects
           :return static_tgt_mask: mask of potentially static objects
        """
        with tf.variable_scope("build_semantic_masks"):
            dynamic_tgt_mask = self.__priors
            static_tgt_mask = 1.0 - dynamic_tgt_mask
            return dynamic_tgt_mask, static_tgt_mask
