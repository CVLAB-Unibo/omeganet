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
Run OmegaNet in a one-shot way:
Given a single tgt image or three images, we run OmegaNet to get the results
for a set of tasks.
At the end, colored images will be saved in the destinatio folder.
"""
from __future__ import division
import tensorflow as tf
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from helpers import utilities
from helpers.flow_tool import flowlib
from networks import complete_network
from networks import general_network
from tensorflow.python.util import deprecation

# disable future warnings and info messages for this demo
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


parser = argparse.ArgumentParser(description="Single shot estimation")
parser.add_argument("--tgt", type=str, help="path to t0 RGB image", required=True)
parser.add_argument(
    "--src1",
    type=str,
    help="path to src_1 RGB image (required in case of optical flow)",
    default=None,
)
parser.add_argument(
    "--src2",
    type=str,
    help="path to src_2 RGB image (required in case of optical flow)",
    default=None,
)
parser.add_argument(
    "--tasks",
    nargs="+",
    type=str,
    help="tasks to perform",
    default=["inverse_depth", "flow", "semantic", "motion_mask"],
)
parser.add_argument(
    "--ckpt", type=str, help="path to complete omeganet checkpoint", required=True
)
parser.add_argument("--height", type=int, help="height of resized image", default=192)
parser.add_argument("--width", type=int, help="width of resized image", default=640)
parser.add_argument(
    "--tau",
    type=float,
    help="tau threshold in the paper. For motion segmentation at testing time",
    default=0.5,
)

parser.add_argument("--dest", type=str, help="where save results", default="./results")
parser.add_argument("--cpu", action="store_true", help="run on cpu")

opts = parser.parse_args()

if opts.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def prepare_input():
    """Prepare input for the network
        :return src1: src1 image, resized at opts.height x opts.width
        :return src1: tgt image, resized at opts.height x opts.width
        :return src1: src2 image, resized at opts.height x opts.width
        :return original_tgt: original tgt image, not resize. For motion mask blending
        :return height: height of original image
        :return width: width of the original image
        In case of single depth or semantic, src1 and src2 are equal to tgt
    """

    expected_more_images = False

    if not os.path.isfile(opts.tgt):
        raise ValueError("Cannot find tgt image:{}".format(opts.tgt))

    if "flow" in opts.tasks or "motion_mask" in opts.tasks:
        if opts.src1 is None or opts.src2 is None:
            raise ValueError(
                "Expected src1 and src2 for optical flow and motion estimation, but are None"
            )
        if not os.path.isfile(opts.src1):
            raise ValueError("Image src1 not found")
        if not os.path.isfile(opts.src2):
            raise ValueError("Image src2 not found")
        expected_more_images = True
    else:
        if not os.path.isfile(opts.tgt):
            raise ValueError("Cannot find tgt:{}".format(opts.tgt))
    if opts.dest is not None:
        utilities.create_dir(opts.dest)

    tgt = cv2.imread(opts.tgt)
    tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
    original_tgt = None
    if "motion_mask" in opts.tasks:
        original_tgt = tgt

    tgt = tgt / 255.0

    if expected_more_images:
        src1 = cv2.imread(opts.src1)
        src1 = cv2.cvtColor(src1, cv2.COLOR_BGR2RGB)
        src1 = src1 / 255.0

        if src1.shape != tgt.shape:
            raise ValueError("tgt and src1 have different shapes")

        src2 = cv2.imread(opts.src2)
        src2 = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)
        src2 = src2 / 255.0

        if src2.shape != tgt.shape:
            raise ValueError("tgt and src2 have different shapes")

    else:
        # NOTE: in case of src1 and src2 are useless,
        # we feed the tensor_src1 and tensor_src2 placeholders
        # with tgt one
        src1 = tgt
        src2 = tgt

    height, width = tgt.shape[0:2]

    src1 = cv2.resize(src1, (opts.width, opts.height))
    tgt = cv2.resize(tgt, (opts.width, opts.height))
    src2 = cv2.resize(src2, (opts.width, opts.height))

    src1 = np.expand_dims(src1, 0).astype(np.float32)
    tgt = np.expand_dims(tgt, 0).astype(np.float32)
    src2 = np.expand_dims(src2, 0).astype(np.float32)
    return src1, tgt, src2, original_tgt, height, width


def main(_):
    """Run the inference
    """
    model_exists = utilities.check_model_exists(opts.ckpt)
    if not model_exists:
        raise ValueError("Model not found")
    src1, tgt, src2, original_tgt, height, width = prepare_input()
    output_tensors = []

    print(" [*] Session creation: SUCCESS")
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    training_flag = tf.placeholder(tf.bool)

    tensor_src1 = tf.placeholder(
        tf.float32, shape=(1, opts.height, opts.width, 3), name="src1"
    )
    tensor_tgt = tf.placeholder(
        tf.float32, shape=(1, opts.height, opts.width, 3), name="tgt"
    )
    tensor_src2 = tf.placeholder(
        tf.float32, shape=(1, opts.height, opts.width, 3), name="src2"
    )
    batch = {"src_img_1": tensor_src1, "tgt_img": tensor_tgt, "src_img_2": tensor_src2}

    network_params = general_network.network_parameters(
        height=opts.height, width=opts.width, load_only_baseline=False, tau=opts.tau,
    )
    network = complete_network.OmegaNet(
        batch, is_training=training_flag, params=network_params
    )
    network.build()
    var_list = network.get_network_params()
    saver = tf.train.Saver(var_list=var_list)

    init_op = tf.group(
        tf.global_variables_initializer(), tf.local_variables_initializer()
    )
    sess.run(init_op)
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    saver.restore(sess, opts.ckpt)
    print(" [*] Load model: SUCCESS")

    index = 0
    output_mapping = {}

    if "inverse_depth" in opts.tasks:
        inverse_depth = tf.image.resize_images(network.disp, [height, width])
        output_tensors.append(inverse_depth)
        output_mapping[index] = "inverse_depth"
        index += 1

    if "semantic" in opts.tasks:
        semantic = network.prepare_semantic(
            network.semantic_logits, height=height, width=width
        )
        output_tensors.append(semantic)
        output_mapping[index] = "semantic"
        index += 1

    if "flow" in opts.tasks:
        optical_flow = tf.image.resize_images(network.optical_flow, [height, width])
        output_tensors.append(optical_flow)
        output_mapping[index] = "flow"
        index += 1

    if "motion_mask" in opts.tasks:
        motion_mask = tf.image.resize_images(
            network.motion_mask,
            [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        output_tensors.append(motion_mask)
        output_mapping[index] = "motion_mask"
        index += 1

    results = sess.run(
        output_tensors,
        feed_dict={
            training_flag: False,
            tensor_src1: src1,
            tensor_tgt: tgt,
            tensor_src2: src2,
        },
    )

    name = os.path.basename(opts.tgt)
    extension = name.split(".")[-1]
    name = name.replace(extension, "png")
    dest = os.path.join(opts.dest, "{}" + name)

    for index, output in enumerate(results):
        output = output.squeeze()
        task = output_mapping[index]

        if task == "inverse_depth":
            plt.imsave(
                dest.format("inverse_depth_"), output, cmap="magma",
            )

        if task == "flow":
            scaling_w = width / opts.width
            scaling_h = height / opts.height
            output *= np.tile(
                np.array((scaling_w, scaling_h), dtype=np.float32), (height, width, 1)
            )
            flow_as_img = flowlib.flow_to_image(output)
            flow_as_img = cv2.cvtColor(flow_as_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dest.format("flow_"), flow_as_img)

        if task == "semantic":
            colored_semantic_map = utilities.color_semantic(output)
            colored_semantic = cv2.cvtColor(
                colored_semantic_map.astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(dest.format("semantic_"), colored_semantic)

        if task == "motion_mask":
            colored_motion_mask = utilities.color_motion_mask(output)
            blended_image = cv2.addWeighted(
                colored_motion_mask, 0.9, original_tgt, 0.8, 0.0,
            )
            blended_image = cv2.cvtColor(
                blended_image.astype(np.uint8), cv2.COLOR_BGR2RGB
            )
            cv2.imwrite(dest.format("moving_objects_"), blended_image)

    print("{} outputs have been produced in {} folder".format(index + 1, opts.dest))
    sess.close()
    coordinator.request_stop()
    coordinator.join(threads)


if __name__ == "__main__":
    tf.app.run()
