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

""" Tester for KITTI depth
"""
import os
import tensorflow as tf
import cv2
import numpy as np
from testers.general_tester import GeneralTester
from helpers import utilities
from tqdm import tqdm


class Tester(GeneralTester):
    """KITTI Depth Tester.
        It produces depth artifacts for the KITTI dataset
    """

    def prepare(self):
        """Create output folders
        """
        dest = os.path.join(self.params.output_path, "depth")
        utilities.create_dir(dest)

    def test(self, network, dataloader, training_flag):
        """Test KITTI depth
            It produces in the params.output_path folder the depth
            artifacts.
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

        prediction_disp = tf.image.resize_images(
            network.disp, [dataloader.image_h, dataloader.image_w]
        )

        print(" [*] Start depth artifacts generation")
        with tqdm(total=self.num_test_samples) as pbar:
            for step in range(self.num_test_samples):
                ops = [prediction_disp]
                outputs = sess.run(ops, feed_dict={training_flag: False})
                name_disp = self.get_name(step)
                inverse_depth = outputs[0].squeeze()
                np.save(
                    os.path.join(self.params.output_path, "depth", name_disp + ".npy"),
                    np.array(inverse_depth),
                )
                pbar.update(1)

        coordinator.request_stop()
        coordinator.join(threads)

    def get_name(self, step):
        """Get right file name
            :param step: current step
            :return name: name of artifact, based on step
        """
        name = str(step)
        return name
