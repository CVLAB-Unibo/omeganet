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
Dataloader for Test
"""
import tensorflow as tf
import numpy as np
from collections import namedtuple


dataloader_parameters = namedtuple("dataloader_parameters", "height, width, task")


def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


class GeneralDataloader(object):
    def __init__(
        self, datapath, filenames_file, params,
    ):
        if not datapath.endswith("/"):
            datapath = datapath + "/"
        self.datapath = datapath
        self.params = params
        self.filenames_file = filenames_file
        self.src_img_1_batch = None
        self.src_img_2_batch = None
        self.tgt_img_batch = None
        self.build()

    def build(self):
        pass

    def get_next_batch(self):
        pass

    def read_image(self, image_path):
        """Read an image from the file system
            :params image_path: string, path to image
        """
        with tf.variable_scope("read_image"):
            path_length = string_length_tf(image_path)[0]
            file_extension = tf.substr(image_path, path_length - 3, 3)
            file_cond = tf.equal(file_extension, "jpg")

            image = tf.cond(
                file_cond,
                lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                lambda: tf.image.decode_png(tf.read_file(image_path)),
            )

            self.image_w = tf.shape(image)[1]
            self.image_h = tf.shape(image)[0]

            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_images(
                image,
                [self.params.height, self.params.width],
                tf.image.ResizeMethod.AREA,
            )
            return image
