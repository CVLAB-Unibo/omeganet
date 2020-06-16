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
Test your network on a specific task
"""

import argparse
import tensorflow as tf
import numpy as np
import os
from dataloaders import factory as dataloader_factory
from dataloaders.general_dataloader import dataloader_parameters
from testers import factory as tester_factory
from tensorflow.python.util import deprecation
from networks import general_network
from networks import complete_network
from helpers import utilities

# disable future warnings and info messages for this demo
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="Test your network")

parser.add_argument(
    "--task",
    type=str,
    default="depth",
    help="task to test",
    choices=["depth", "semantic", "flow", "mask"],
)
parser.add_argument("--datapath", type=str, help="path to data", required=True)
parser.add_argument("--ckpt", type=str, help="path to checkpoint", required=True)
parser.add_argument(
    "--filenames_file",
    type=str,
    help="path to filenames file",
    default="filenames/eigen_test.txt",
)
parser.add_argument("--height", type=int, help="height of resized image", default=192)
parser.add_argument("--width", type=int, help="width of resized image", default=640)
parser.add_argument(
    "--dest", type=str, help="where save artifacts", default="./artifacts"
)
parser.add_argument(
    "--load_only_baseline",
    action="store_true",
    help="if set, load only Baseline (CameraNet+DSNet). Otherwise, full OmegaNet will be loaded",
)
parser.add_argument(
    "--cpu", help="the network runs on CPU if enabled", action="store_true"
)
parser.add_argument(
    "--tau",
    type=float,
    help="tau threshold in the paper. For motion segmentation at testing time",
    default=0.5,
)

args = parser.parse_args()


if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def configure_parameters():
    """Prepare configurations for Network, Dataloader and Tester
        :return network_params: configuration for Network
        :return dataloader_params: configuration for Dataloader
        :return testing_params: configuration for Tester
    """
    network_params = general_network.network_parameters(
        height=args.height,
        width=args.width,
        load_only_baseline=args.load_only_baseline,
        tau=args.tau,
    )

    dataloader_params = dataloader_parameters(
        height=args.height, width=args.width, task=args.task
    )

    testing_params = tester_factory.tester_parameters(
        output_path=args.dest,
        checkpoint_path=args.ckpt,
        width=args.width,
        height=args.height,
        filenames_file=args.filenames_file,
        datapath=args.datapath,
    )

    return network_params, dataloader_params, testing_params


def configure_network(network_params, dataloader_params):
    """Build the Dataloader, then build the Network.
        :param network_params: configuration for Network
        :param dataloader_params: configuration for Dataloader
        :return network: built Network
        :return dataloader: built Dataloader
        :return training_flag: bool placeholder. For Batchnorm

    """
    training_flag = tf.placeholder(tf.bool)
    dataloader = dataloader_factory.get_dataloader(args.task)(
        datapath=args.datapath,
        filenames_file=args.filenames_file,
        params=dataloader_params,
    )
    batch = dataloader.get_next_batch()
    network = complete_network.OmegaNet(
        batch, is_training=training_flag, params=network_params
    )

    network.build()
    return network, dataloader, training_flag


def main(_):
    """Create the Dataloader, the Network and the Tester.
        Then, run the Tester.
        :raise ValueError: if model does not exist
    """
    model_exists = utilities.check_model_exists(args.ckpt)
    if not model_exists:
        raise ValueError("Model not found")
    network_params, dataloader_params, testing_params = configure_parameters()
    network, dataloader, training_flag = configure_network(
        network_params, dataloader_params
    )

    tester = tester_factory.get_tester(args.task)(testing_params)
    tester.test(network, dataloader, training_flag)


if __name__ == "__main__":
    tf.app.run()
