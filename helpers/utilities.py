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


""" Utility functions
"""
from collections import namedtuple
import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib
import matplotlib.cm

Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        "trainId",  # An integer ID that overwrites the ID above, when creating ground truth
        # images for training.
        # For training, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        "category",  # The name of the category that this label belongs to
        "categoryId",  # The ID of this category. Used to create ground truth images
        # on category level.
        "hasInstances",  # Whether this label distinguishes between single instances or not
        "ignoreInEval",  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
    ],
)

labels_all = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    Label("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
    Label("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
    Label("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
    Label("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
    Label("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
    Label("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
    Label("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
    Label("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    Label("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
    Label("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
    Label("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
]

labels_train = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label("road", 0, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 1, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("building", 2, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("wall", 3, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("fence", 4, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("pole", 5, 5, "object", 3, False, False, (153, 153, 153)),
    Label("traffic light", 6, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 7, 7, "object", 3, False, False, (220, 220, 0)),
    Label("vegetation", 8, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain", 9, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("sky", 10, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 11, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 12, 12, "human", 6, True, False, (255, 0, 0)),
    Label("car", 13, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("truck", 14, 14, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus", 15, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("train", 16, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle", 17, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle", 18, 18, "vehicle", 7, True, False, (119, 11, 32)),
]


labels_static_dynamic_trainIds = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label("road", 0, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 1, 0, "flat", 1, False, False, (244, 35, 232)),
    Label("building", 2, 0, "construction", 2, False, False, (70, 70, 70)),
    Label("wall", 3, 0, "construction", 2, False, False, (102, 102, 156)),
    Label("fence", 4, 0, "construction", 2, False, False, (190, 153, 153)),
    Label("pole", 5, 0, "object", 3, False, False, (153, 153, 153)),
    Label("traffic light", 6, 0, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 7, 0, "object", 3, False, False, (220, 220, 0)),
    Label("vegetation", 8, 0, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain", 9, 0, "nature", 4, False, False, (152, 251, 152)),
    Label("sky", 10, 0, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 11, 1, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 12, 1, "human", 6, True, False, (255, 0, 0)),
    Label("car", 13, 1, "vehicle", 7, True, False, (0, 0, 142)),
    Label("truck", 14, 1, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus", 15, 1, "vehicle", 7, True, False, (0, 60, 100)),
    Label("train", 16, 1, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle", 17, 1, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle", 18, 1, "vehicle", 7, True, False, (119, 11, 32)),
]

labels = labels_train
id2Color = {label.id: label.color for label in labels}
id2trainId = {label.id: label.trainId for label in labels_all}
id2name = {label.id: label.name for label in labels}

labels2priors = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
)  # labels_static_dynamic_trainIds.trainId


def extract_semantic_priors(predictions):
    """ Extract priors from a semantic map
        Return a new map, with the same shape of the input, with 1 for possibly moving
        objects and 0 otherwise.
        Params:
            predictions: BxHxWx1
        Returns:
            priors: BxHxWx1
    """
    priors = []
    b, h, w, _ = predictions.shape
    for i in range(b):
        p = tf.py_func(label_to_priors, [predictions[i]], tf.uint8)
        p = tf.cast(p, tf.float32)
        priors.append(p)
    priors = tf.stack(priors, axis=0)
    priors.set_shape(predictions.get_shape())
    return priors


def label_to_priors(predictions):
    predictions = predictions.astype(np.uint8)
    predictions = predictions.squeeze()
    priors = labels2priors[predictions]
    priors = np.expand_dims(priors, -1)
    return priors.astype(np.uint8)


def colormap_semantic(pred_sem, dict_id2color=id2Color):
    p = tf.squeeze(tf.cast(pred_sem, tf.uint8), axis=-1)
    p = tf.stack([p, p, p], axis=-1)
    m = tf.zeros_like(p)
    for i in range(0, len(dict_id2color)):
        mi = tf.multiply(tf.ones_like(p), dict_id2color[i])
        m = tf.where(tf.equal(p, i), mi, m)
    return m


def get_num_classes():
    return len(labels)


def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else "gray")
    colors = tf.constant(cm.colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value


def count_text_lines(file_path):
    f = open(file_path, "r")
    lines = f.readlines()
    f.close()
    return len(lines)


def flow_to_color(flow, mask=None, max_flow=None):
    """
    From Unflow by Meister et al
    https://arxiv.org/pdf/1711.07837.pdf
    https://github.com/simonmeister/UnFlow
    
    Converts flow to 3-channel color image.
    Args:
        flow: tensor of shape [num_batch, height, width, 2].
        mask: flow validity mask of shape [num_batch, height, width, 1].
    """
    n = 8
    num_batch, height, width, _ = tf.unstack(tf.shape(flow))
    mask = tf.ones([num_batch, height, width, 1]) if mask is None else mask
    flow_u, flow_v = tf.unstack(flow, axis=3)
    if max_flow is not None:
        max_flow = tf.maximum(max_flow, 1)
    else:
        max_flow = tf.reduce_max(tf.abs(flow * mask))
    mag = tf.sqrt(tf.reduce_sum(tf.square(flow), 3))
    angle = tf.atan2(flow_v, flow_u)

    im_h = tf.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = tf.clip_by_value(mag * n / max_flow, 0, 1)
    im_v = tf.clip_by_value(n - im_s, 0, 1)
    im_hsv = tf.stack([im_h, im_s, im_v], 3)
    im = tf.image.hsv_to_rgb(im_hsv)
    return im * mask


def tf_color_prior(prior):
    mapping = {0: (0, 0, 255), 1: (0, 255, 0)}
    return colormap_semantic(prior, mapping)


def get_height_width(img):
    s = tf.shape(img)
    h = tf.to_int32(s[1])
    w = tf.to_int32(s[2])
    return h, w


def get_priors_or_default(priors, img, params, mode):
    return (
        priors
        if (params.use_priors and mode == "semantic")
        else tf.zeros_like(img[:, :, :, 0:1])
    )


def create_dir(dirname):
    """Create a directory if not exists
        :param dirname: path of the directory to create
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def mask(img, mask, active):
    with tf.variable_scope("mask"):
        if active:
            return img * mask
        return img


def flow_resize(flow, out_size, is_scale=True, method=0):
    """
        method: 0 mean bilinear, 1 means nearest
    """
    flow_size = tf.to_float(tf.shape(flow)[-3:-1])
    b, _, _, c = flow.get_shape().as_list()
    flow = tf.image.resize_images(flow, out_size, method=method, align_corners=True)
    if is_scale:
        scale = tf.to_float(out_size) / flow_size
        scale = tf.stack([scale[1], scale[0]])
        flow = tf.multiply(flow, scale)
    return flow


def color_semantic(semantic_map, mapping=None):
    """Color a semantic map in numpy
        :param x: input semantic map
        :param mapping: optional color scheme. If not set, a default
            color scheme will be applied
        :return colored: colored semantic map
    """
    if mapping is None:
        mapping = [
            (128, 64, 128),
            (244, 35, 232),
            (70, 70, 70),
            (102, 102, 156),
            (190, 153, 153),
            (153, 153, 153),
            (250, 170, 30),
            (220, 220, 0),
            (107, 142, 35),
            (152, 251, 152),
            (70, 130, 180),
            (220, 20, 60),
            (255, 0, 0),
            (0, 0, 142),
            (0, 0, 70),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 230),
            (119, 11, 32),
        ]
    h, w = semantic_map.shape[:2]
    colored = np.ones([h, w, 3], np.uint8)
    for x in range(len(mapping)):

        color = np.ones_like(colored) * mapping[x]
        current_sem = np.stack((semantic_map, semantic_map, semantic_map), axis=-1)
        index = np.ones_like(current_sem) * x
        colored = np.where(current_sem == index, color, colored)
    return colored


def check_model_exists(ckpt):
    """Check if model exists
        :param ckpt: path to checkpoint
        :return exist: flag. True if model exists
    """
    expected_data = ckpt + ".data-00000-of-00001"
    return os.path.exists(expected_data)


def write_kitti_png_flow(dest, flow_data, mask_data=None):
    """Save optical flow in KITTI format, ie 16 bit png image"
        :param dest: where image will be saved
        :param flow_data: optical flow field. Array with shape (H,W,2)
        :param mask_data: optional mask
    """
    flow_img = np.zeros((flow_data.shape[0], flow_data.shape[1], 3), dtype=np.uint16)
    flow_img[:, :, 2] = flow_data[:, :, 0] * 64.0 + 2 ** 15
    flow_img[:, :, 1] = flow_data[:, :, 1] * 64.0 + 2 ** 15
    if mask_data is None:
        mask_data = np.ones_like(flow_img[:, :, 2])
    flow_img[:, :, 0] = mask_data[:, :]
    cv2.imwrite(dest, flow_img)


def color_motion_mask(mask, color=None):
    """Apply a color scheme to a motion mask
        :param mask: input motion mask
        :param color: RGB tuple, color applied to moving objects. Default (220, 20, 60)
        :return final_mask: colored mask, as np.uint8
    """
    if color is None:
        color = (220, 20, 60)
    h, w = mask.shape
    ext_mask = np.stack([mask, mask, mask], -1).astype(np.uint8)
    color = np.ones_like(ext_mask) * color
    index = np.ones_like(ext_mask) * 1.0
    final_mask = np.where(ext_mask == index, color, ext_mask).astype(np.uint8)
    return final_mask
