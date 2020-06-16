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
Factory for dataloaders
"""

import tensorflow as tf
import numpy as np
from dataloaders import dataloader_one_frame, dataloader_three_frames


TESTER_DATALOADERS_FACTORY = {
    "semantic": dataloader_one_frame.TestDataloader,
    "depth": dataloader_one_frame.TestDataloader,
    "flow": dataloader_three_frames.TestDataloader,
    "mask": dataloader_three_frames.TestDataloader,
}


def get_dataloader(task):
    """Return the desired dataloader.
        :param task: task to perform
        :return dataloader: dataloader suited for the task
    """
    assert task in TESTER_DATALOADERS_FACTORY.keys()
    return TESTER_DATALOADERS_FACTORY[task]
