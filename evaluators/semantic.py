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


from __future__ import print_function

import tensorflow as tf
import cv2
import numpy as np
import argparse
import os

id2trainId = {
     0 : 255,
     1 : 255,
     2 : 255,
     3 : 255,
     4 : 255,
     5 : 255,
     6 : 255,
     7 :   0,
     8 :   1,
     9 : 255,
    10 : 255,
    11 :   2,
    12 :   3,
    13 :   4,
    14 : 255,
    15 : 255,
    16 : 255,
    17 :   5,
    18 : 255,
    19 :   6,
    20 :   7,
    21 :   8,
    22 :   9,
    23 :  10,
    24 :  11,
    25 :  12,
    26 :  13,
    27 :  14,
    28 :  15,
    29 : 255,
    30 : 255,
    31 :  16,
    32 :  17,
    33 :  18
}


trainId2cat = {
     0 :   0,
     1 :   0,
     2 :   1,
     3 :   1,
     4 :   1, 
     5 :   2,
     6 :   2,
     7 :   2,
     8 :   3,
     9 :   3,
    10 :   4,
    11 :   5,
    12 :   5,
    13 :   6,
    14 :   6,
    15 :   6,
    16 :   6,
    17 :   6,
    18 :   6,
}

trainId2name = {
     0 : "road",
     1 : "sidewalk",
     2 : "building",
     3 : "wall",
     4 : "fence",
     5 : "pole",
     6 : "traffic_light",
     7 : "traffic_sign",
     8 : "vegetation",
     9 : "terrain",
    10 : "sky",
    11 : "person",
    12 : "rider",
    13 : "car",
    14 : "truck",
    15 : "bus",
    16 : "train",
    17 : "motorcycle",
    18 : "bicycle"
}

num_train_classes = 19
num_categories = 7
num_total_classes = 34

parser = argparse.ArgumentParser(description="Evaluation Semantic")
### PATHS
parser.add_argument(
    "--dataset",
    dest="dataset",
    choices=["kitti", "cityscapes"],
    default="kitti",
    help="kitti, cityscapes",
)
parser.add_argument(
    "--datapath", 
    type=str, 
    help="Path to dataset (e.g. data_semantics Kitti 2015)"
)
parser.add_argument(
    "--prediction_folder", 
    type=str, 
    help="Path to predictions"
)
parser.add_argument(
    "--filename_file", 
    default="../filenames/kitti_2015_test_semantic.txt", 
    help="Path to txt input list"
)
### PARAMS
parser.add_argument(
    "--ignore_label", 
    type=int, 
    default=255, 
    help="label to ignore in evaluation",
)
parser.add_argument(
    "--format_pred",
    type=str,
    choices=["id", "trainId"],
    default="trainId",
    help="encoding of predictions, trainId or id",
)
parser.add_argument(
    "--format_gt",
    type=str,
    choices=["id", "trainId"],
    default="id",
    help="encoding of gt, trainId or id",
)
args = parser.parse_args()


def convert_labels(sem, mapping):
    p = tf.cast(sem, tf.uint8)
    m = tf.ones_like(p) * 255
    for i in range(0, len(mapping)):
        mi = tf.multiply(tf.ones_like(p), mapping[i])
        m = tf.where(tf.equal(p, i), mi, m)
    return m


prediction_placeholder = tf.placeholder(tf.int32)
prediction_placeholder.set_shape([None, None, 1])
gt_placeholder = tf.placeholder(tf.int32)

gt = gt_placeholder
prediction = prediction_placeholder

if args.format_pred == "id":
    prediction = convert_labels(prediction, id2trainId)
if args.format_gt == "id":
    gt = convert_labels(gt, id2trainId)


pred_cat = convert_labels(prediction, trainId2cat)
gt_cat = convert_labels(gt, trainId2cat)

### INIT WEIGHTS MIOU
weightsValue = tf.to_float(tf.not_equal(gt, args.ignore_label))
### IGNORE LABELS TO 0, WE HAVE ALREADY MASKED THOSE PIXELS WITH WEIGHTS 0###
gt = tf.where(tf.equal(gt, args.ignore_label), tf.zeros_like(gt), gt)
prediction = tf.where(
    tf.equal(prediction, args.ignore_label), tf.zeros_like(prediction), prediction
)
### ACCURACY ###
acc, update_op_acc = tf.metrics.accuracy(gt, prediction, weights=weightsValue)
### MIOU ###
miou, update_op = tf.metrics.mean_iou(
    labels=tf.reshape(gt, [-1]),
    predictions=tf.reshape(prediction, [-1]),
    num_classes=num_train_classes,
    weights=tf.reshape(weightsValue, [-1]),
)

# CATEGORIES
### INIT WEIGHTS MIOU
weightsValue_cat = tf.to_float(tf.not_equal(gt_cat, args.ignore_label))
### IGNORE LABELS TO 0, WE HAVE ALREADY MASKED THOSE PIXELS WITH WEIGHTS 0###
gt_cat = tf.where(tf.equal(gt_cat, args.ignore_label), tf.zeros_like(gt_cat), gt_cat)
pred_cat = tf.where(
    tf.equal(pred_cat, args.ignore_label), tf.zeros_like(pred_cat), pred_cat
)
### MIOU ###
miou_cat, update_op_cat = tf.metrics.mean_iou(
    labels=tf.reshape(gt_cat, [-1]),
    predictions=tf.reshape(pred_cat, [-1]),
    num_classes=num_categories,
    weights=tf.reshape(weightsValue_cat, [-1]),
    name="mean_iou_cat"
)

init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

miou_value = 0
with tf.Session() as sess:
    sess.run(init_op)
    lines = open(args.filename_file).readlines()
    lenght = len(lines)

    for idx, line in enumerate(lines):
        base_path = line.strip()
        prediction_folder = os.path.join(args.prediction_folder, base_path)
        datapath = os.path.join(args.datapath, "training/semantic", base_path)
        print("GT: ", datapath, " Pred: ", prediction_folder, idx, "/", lenght, end="\r")

        gt_value = cv2.imread(datapath, cv2.IMREAD_GRAYSCALE)
        pred_value = cv2.imread(prediction_folder, cv2.IMREAD_GRAYSCALE)

        image_w = gt_value.shape[1]
        image_h = gt_value.shape[0]

        if args.dataset == "cityscapes":
            crop_height = (image_h * 4) // 5
            gt_value = gt_value[:crop_height, :]
            gt_value = cv2.resize(
                gt_value, (image_w, image_h), interpolation=cv2.INTER_NEAREST
            )

        _, _, _ = sess.run(
            [update_op_acc, update_op, update_op_cat],
            feed_dict={
                prediction_placeholder: np.expand_dims(pred_value, axis=-1),
                gt_placeholder: np.expand_dims(gt_value, axis=-1),
            },
        )
        acc_value, miou_value, miou_cat_value = sess.run(
            [acc, miou, miou_cat],
            feed_dict={
                prediction_placeholder: np.expand_dims(pred_value, axis=-1),
                gt_placeholder: np.expand_dims(gt_value, axis=-1),
            },
        )

    confusion_matrix = (
        tf.get_default_graph()
        .get_tensor_by_name("mean_iou/total_confusion_matrix:0")
        .eval()
    )
    print("")
    for cl in range(confusion_matrix.shape[0]):
        tp_fn = np.sum(confusion_matrix[cl, :])
        tp_fp = np.sum(confusion_matrix[:, cl])
        tp = confusion_matrix[cl, cl]
        if tp == 0 and (tp_fn + tp_fp - tp) == 0:
            IoU_cl = float("nan")
        else:
            IoU_cl = tp / (tp_fn + tp_fp - tp)
        print(trainId2name[cl] + ": {:.8f}".format(IoU_cl))
    print("mIoU: " + str(miou_value))
    print("mIoU Categories: " + str(miou_cat_value))
    print("Pix. Acc.: " + str(acc_value))
