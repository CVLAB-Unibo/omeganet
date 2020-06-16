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
Factory for testers
"""

import tensorflow as tf
import numpy as np
from testers import kitti_depth, kitti_flow, kitti_semantic, kitti_mask, error_tester
from collections import namedtuple

tester_parameters = namedtuple(
    "tester_parameters",
    "output_path " "checkpoint_path, " "width," "height," "filenames_file," "datapath",
)

TESTER_KITTI_FACTORY = {
    "depth": kitti_depth.Tester,
    "flow": kitti_flow.Tester,
    "semantic": kitti_semantic.Tester,
    "mask": kitti_mask.Tester,
}


def get_tester(task):
    """Select best Tester given a tast and a dataset
        If no Tester is available for that task on
        the selected Dataset (ie, depth for CS), then
        an ErrorTester is returned.
        :param task: task to perform
    """
    assert task in TESTER_KITTI_FACTORY
    return TESTER_KITTI_FACTORY[task]
