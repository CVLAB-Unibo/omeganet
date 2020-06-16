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

""" Tester for KITTI optical flow
"""

from __future__ import division
import os
import tensorflow as tf
from tqdm import tqdm
import cv2
import numpy as np
from testers.general_tester import GeneralTester
from helpers import utilities


class Tester(GeneralTester):
    """Tester for optical flow on KITTI
    """

    def prepare(self):
        """Create output folders
        """
        dest = os.path.join(self.params.output_path, "flow")
        utilities.create_dir(dest)

    def test(self, network, dataloader, training_flag):
        """Generate optical
            It saves optical flow artifacts in the
            self.params.output_path/flow folder.
            :param network: network to test
            :param dataloader: dataloader for this test
            :param training_flag: training_flag for Batchnorm
        """
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        self.prepare()
        var_list = network.get_network_params()
        saver = tf.train.Saver(var_list=var_list)

        init_op = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )
        sess.run(init_op)

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        saver.restore(sess, self.params.checkpoint_path)
        print(" [*] Load model: SUCCESS")

        predicted_flow = tf.image.resize_images(
            network.optical_flow, [dataloader.image_h, dataloader.image_w]
        )

        print(" [*] Start optical flow artifacts generation")
        with tqdm(total=self.num_test_samples) as pbar:
            for step in range(self.num_test_samples):
                ops = [
                    predicted_flow,
                    dataloader.image_h,
                    dataloader.image_w,
                ]

                outputs = sess.run(ops, feed_dict={training_flag: False})
                name = self.get_name(step)
                flow = outputs[0].squeeze()
                image_h = outputs[1]
                image_w = outputs[2]

                flow = self.scale_flow(flow, image_h, image_w)

                utilities.write_kitti_png_flow(
                    os.path.join(self.params.output_path, "flow", name + ".png"), flow
                )
                pbar.update(1)

        coordinator.request_stop()
        coordinator.join(threads)

    def get_name(self, step):
        """Get right file name
            :param step: current step
            :return name: name of artifact, based on step
        """
        name = (
            self.samples[step]
            .split(" ")[1]
            .replace("/", "_")
            .replace(".png", "")
            .strip()
        )
        return name

    def scale_flow(self, flow, image_h, image_w):
        """Apply the scale factor to the resized optical flow
            :param flow: optional flow. Array with shape (H,W,2)
            :param image_h: height of the original image
            :param image_w: width of the original image
            :return scaled_flow: optical flow rescaled by the scaling factor
        """
        scaling_w = image_w / self.params.width
        scaling_h = image_h / self.params.height
        flow *= np.tile(np.array(scaling_w, scaling_h), (image_h, image_w, 1))
        return flow
