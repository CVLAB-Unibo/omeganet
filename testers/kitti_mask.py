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

from __future__ import division
import tensorflow as tf
import os
import cv2
import numpy as np
from testers.general_tester import GeneralTester
from helpers import utilities
from tqdm import tqdm


class Tester(GeneralTester):
    def prepare(self):
        """Create output folders
        """
        dest = os.path.join(self.params.output_path, "mask")
        utilities.create_dir(dest)

    def test(self, network, dataloader, is_training):
        """ Test motion mask
            It saves motion mask artifacts in the self.params.output_path/mask folder.
            :param network: network to test
            :param dataloader: dataloader for this test
            :param is_training: training_flag for Batchnorm
        """
        # SESSION
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

        segmented_mask = tf.image.resize_images(
            network.motion_mask,
            [dataloader.image_h, dataloader.image_w],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        with tqdm(total=self.num_test_samples) as pbar:
            for step in range(self.num_test_samples):
                ops = [segmented_mask]
                outputs = sess.run(ops, feed_dict={is_training: False})

                name = self.get_name(step)
                seg_mask = outputs[0].squeeze()

                cv2.imwrite(
                    os.path.join(self.params.output_path, "mask", name + ".png"),
                    (seg_mask * 255.0).astype(np.uint8),
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
