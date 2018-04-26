# 1. Split Data into Train, Val and Test
# 2. Convert into .rec file
# 3. return train, val and test iterator.

import mxnet as mx
import numpy as np
import sys, os
curr_dir = os.path.dirname(__file__)
sys.path.append(curr_dir)
import argparse
import logging
from random import shuffle
import config as CFG
import numpy as np
import cv2
import copy

from enum import Enum, IntEnum

import imgaug


# Training setting
BIN, OVERLAP = CFG.BIN, CFG.OVERLAP
NORM_H, NORM_W = CFG.NORM_H, CFG.NORM_W
VEHICLES = CFG.VEHICLES

RAN = CFG.RAN
W_H_C_Ratio = CFG.W_H_C_Ratio


# Resize Strategies
Resize = IntEnum(
    "Resize", ('NONE',                 # Nothing
               'CENTRAL_CROP',         # Crop (and pad if necessary)
               'PAD_AND_RESIZE',       # Pad, and resize to output shape
               'WARP_RESIZE')          # Warp resize
)

# VGG mean parameters
_R_MEAN = 123
_G_MEAN = 117
_B_MEAN = 104

# Some training pre-processing parameters
BBOX_CROP_OVERLAP = 0.5 # Minimum overlap to keep a bbox after cropping
MIN_OBJECT_COVERED = 0.25
CROP_RATIO_RANGE = (0.6, 1.67)
EVAL_SIZE = (300, 300)
MAX_TRUNC = 0.5
MAX_OCC = 3

# Mean of the car, pedestrian and cyclist
dims_avg = {
    'Cyclist': np.array([1.73532436,  0.58028152,  1.77413709]),
    'Van': np.array([2.18928571,  1.90979592,  5.07087755]),
    'Tram': np.array([3.56092896,   2.39601093,  18.34125683]),
    'Car': np.array([1.52159147,  1.64443089,  3.85813679]),
    'Pedestrian': np.array([1.75554637,  0.66860882,  0.87623049]),
    'Truck': np.array([3.07392252,   2.63079903,  11.2190799])
}


def bin_stats(num_bins, range):
    wedge = np.pi*2/num_bins
    cos_half_range = np.cos(0.5*range)
    c = np.asarray(list(xrange(num_bins)), dtype=np.float32) * wedge
    sin_bins = np.sin(c)
    cos_bins = np.cos(c)

    return sin_bins, cos_bins, cos_half_range


SIN_BINS, COS_BINS, COS_HALF_RANGE = bin_stats(BIN, RAN)

def bboxes_filter_labels(out_labels, labels, bboxes, tensor_list):
    """
    Filter the data with subcategories specified in out labels
    :param out_labels: list of index(int)
    :param labels: tensor
    :param bboxes:
    :param tensor_list: list of tensors
    :return:
        1. labels:       gathered tensors
        2. tensor_list:  gathered tensors
    """
    pass

def whiten_image(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    image = image - means

    return image

def compute_anchors(angle):
    anchors = []

    wedge = 2.*np.pi/BIN
    l_index = int(angle/wedge)
    r_index = l_index+1

    if (angle-l_index*wedge) < wedge/2*(1+OVERLAP/2):
        anchors.append([l_index, angle-l_index*wedge])

    if (r_index*wedge - angle) < wedge/2*(1+OVERLAP/2):
        anchors.append([r_index%BIN, angle-r_index*wedge])
    
    return anchors


def parse_annotation(label_dir, image_dir):
    all_objs = []
    dims_avg = {key : np.array([0,0,0]) for key in VEHICLES}
    dims_cnt = {key : 0 for key in VEHICLES}

    for label_file in sorted(os.listdir(label_dir)):
        image_file = label_file.replace('txt', 'png')
        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded = np.abs(float(line[2]))

            if line[0] in VEHICLES and truncated < 0.1 and occluded < 0.1:
                new_alpha = float(line[3]) + np.pi/2
                if new_alpha < 0:
                    new_alpha = new_alpha + 2.*np.pi
                new_alpha = new_alpha - int(new_alpha/(2.*np.pi))*(2.*np.pi)

                obj = {
                    'name' : line[0],
                    # 2d bounding box
                    'xmin' : int(float(line[4])),
                    'ymin' : int(float(line[5])),
                    'xmax' : int(float(line[6])),
                    'ymax' : int(float(line[7])),
                    # 3d bounding box dimension
                    'dims' : np.array([float(number) for number in line[8:11]]),
                    'new_alpha' : new_alpha
                }

                # update dims_avg using current object.
                dims_avg[obj['name']] = dims_cnt[obj['name']]*dims_avg[obj['name']]+obj['dims']
                dims_cnt[obj['name']] += 1
                dims_avg[obj['name']] /= dims_cnt[obj['name']]

                all_objs.append(obj)

    # Flip data
    for obj in all_objs:
        # Fix dimensions
        obj['dims'] = obj['dims'] - dims_avg[obj['name']]

        # Fix orientation and confidence for no flip
        orientation = np.zeros((BIN, 2))
        confidence = np.zeros(BIN)

        anchors = compute_anchors(obj['new_alpha'])

        for anchor in anchors:
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1.

        confidence = confidence / np.sum(confidence)

        obj['orient'] = orientation
        obj['conf'] = confidence

        # Fix orientation and confidence for flip
        orientation = np.zeros((BIN, 2))
        confidence = np.zeros(BIN)

        anchors = compute_anchors(2. * np.pi - obj['new_alpha'])
        for anchor in anchors:
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1

        confidence = confidence / np.sum(confidence)

        obj['orient_flipped'] = orientation
        obj['conf_flipped'] = confidence

    return all_objs


def prepare_input_and_output(image_dir, train_inst):
    # Prepare image batch
    xmin = train_inst['xmin'] + np.random.randint(-CFG.MAX_JIT, CFG.MAX_JIT+1)
    ymin = train_inst['ymin'] + np.random.randint(-CFG.MAX_JIT, CFG.MAX_JIT+1)
    xmax = train_inst['xmax'] + np.random.randint(-CFG.MAX_JIT, CFG.MAX_JIT+1)
    ymax = train_inst['ymax'] + np.random.randint(-CFG.MAX_JIT, CFG.MAX_JIT+1)

    img = cv2.imread(image_dir+train_inst['image'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = copy.deepcopy(img[ymin:ymax+1, xmin:xmax+1]).astype(np.float32)

    # Flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5 : img = cv2.flip(img, 1)

    # Resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    img = img - np.array([[[_R_MEAN, _G_MEAN, _B_MEAN]]])

    # Fix orientation and confidence
    if flip > 0.5:
        return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
    else:
        return img, train_inst['dims'], train_inst['orient'], train_inst['conf']


def data_gen(image_dir, all_objs, batch_size):
    num_obj = len(all_objs)

    keys = range(num_obj)
    np.random.shuffle(keys)

    l_bound = 0
    r_bound = batch_size if batch_size < num_obj else num_obj

    while True:
        if l_bound == r_bound:
            l_bound = 0
            r_bound = batch_size if batch_size < num_obj else num_obj
            np.random.shuffle(keys)
        curr_inst = 0
        x_batch = np.zeros((r_bound-l_bound, 224, 224, 3))
        d_batch = np.zeros((r_bound - l_bound, 3))
        o_batch = np.zeros((r_bound - l_bound, BIN, 2))
        c_batch = np.zeros((r_bound - l_bound, BIN))

        for key in keys[l_bound:r_bound]:
            # Augment input image and fix object's orientation and confidence
            image, dimension, orientation, confidence = prepare_input_and_output(image_dir, all_objs[key])
            x_batch[curr_inst, :] = image
            d_batch[curr_inst, :] = dimension
            o_batch[curr_inst, :] = orientation
            c_batch[curr_inst, :] = confidence

            curr_inst += 1
        yield x_batch, [d_batch, o_batch, c_batch]

        l_bound = r_bound
        r_bound = r_bound + batch_size

        if r_bound > num_obj: r_bound = num_obj


def get_iterators(batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter(
        path_imgrec='./kitti/training/image_2.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        shuffle=True,
        rand_crop=True,
        rand_mirror=True
    )
    val = mx.io.ImageRecordIter(
        path_imgrec = './kitti/validation/image_2.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        rand_crop=False,
        rand_mirror=False
    )

    return train, val



