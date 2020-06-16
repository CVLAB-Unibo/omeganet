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

from abc import ABCMeta, abstractmethod


class GeneralTester(object):
    """Template class for Testers
    """

    __metaclass__ = ABCMeta

    def __init__(self, params):
        self.params = params
        with open(params.filenames_file, "r") as f:
            self.samples = f.readlines()
        self.num_test_samples = len(self.samples)

    @abstractmethod
    def test(self, network, dataloader, training_flag):
        """Principal method of the class.
            Start artifact generation.
            :param network: neural network to run
            :param dataloader: tf.dataloader that loads images from the file system
            :param training_flag: training flag bool. For Batchnorm
        """
        pass
