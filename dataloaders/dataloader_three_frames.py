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
Dataloader suited for 3 frames tasks
"""
import tensorflow as tf
import numpy as np
from dataloaders.general_dataloader import GeneralDataloader


class TestDataloader(GeneralDataloader):
    def build(self):
        input_queue = tf.train.string_input_producer(
            [self.filenames_file], shuffle=False
        )
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)
        split_line = tf.string_split([line]).values

        with tf.variable_scope("tester_dataloader_three_frames"):
            with tf.variable_scope("image_reader"):
                src_img_1_path = tf.string_join([self.datapath, split_line[0]])
                tgt_img_path = tf.string_join([self.datapath, split_line[1]])
                src_img_2_path = tf.string_join([self.datapath, split_line[2]])

                src_img_1_o = self.read_image(src_img_1_path)
                tgt_img_o = self.read_image(tgt_img_path)
                src_img_2_o = self.read_image(src_img_2_path)

            with tf.variable_scope("batch_creator"):
                self.src_img_1_batch = tf.stack([src_img_1_o], 0)
                self.tgt_img_batch = tf.stack([tgt_img_o], 0)
                self.src_img_2_batch = tf.stack([src_img_2_o], 0)

            with tf.variable_scope("shape_setter"):
                self.src_img_1_batch.set_shape([1, None, None, 3])
                self.tgt_img_batch.set_shape([1, None, None, 3])
                self.src_img_2_batch.set_shape([1, None, None, 3])

    def get_next_batch(self):
        with tf.variable_scope("get_next_batch"):
            batch = {
                "src_img_1": self.src_img_1_batch,
                "tgt_img": self.tgt_img_batch,
                "src_img_2": self.src_img_2_batch,
            }
            return batch
