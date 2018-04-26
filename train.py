import os
import sys
import cv2
import mxnet as mx
from mxnet import gluon
import numpy as np
import preprocessing
from mxnet.gluon import nn
from mxnet import autograd
import config as CFG
import time
from random import shuffle
from preprocessing import *
import argparse
from tqdm import tqdm

from wget import download

import config as CFG

import tensorboard as tb

BIN, OVERLAP = CFG.BIN, CFG.OVERLAP
W = CFG.W
ALPHA = CFG.ALPHA
MAX_JIT = CFG.MAX_JIT
NORM_H, NORM_W = 224, 224
VEHICLES = CFG.VEHICLES
BATCH_SIZE = CFG.BATCH_SIZE
LEARNING_RATE = CFG.LEARNING_RATE
EPOCHS = CFG.EPOCH
SAVE_PATH = CFG.SAVE_PATH

# def get_model(prefix, epoch):
#     download(prefix+'-symbol.json')
#     download(prefix+'-%04d.params' % (epoch,))
#
# get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)


def load_model(model_name=CFG.BACKBONES['resnext50'], epoch=0):
    return mx.model.load_checkpoint(model_name, epoch)


sym, arg_params, aux_params = load_model(CFG.BACKBONES['resnext50'], 0)


def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='stage4_unit3_relu'):
    """
    :param symbol: the pre-trained network symbol
    :param arg_params: the argument parameters of the pretrained model
    :param num_classes: the number of classes for the fine-tune datasets
    :param layer_name: the layer name before the last fully-connected layer
    :return:
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    # print "Here is net", net
    # net = mx.sym.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    # net = mx.sym.SoftmaxOutput(data=net, name='softmax')
    # new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})

    # return all_layers, net, new_args

    # add 3 branches of tasks
    # 1. dimension
    dim = mx.sym.Convolution(data=net, num_filter=512, kernel=(7, 7), stride=(1, 1), no_bias=True, name='dim_fc1')
    dim = mx.sym.Convolution(data=dim, num_filter=3, kernel=(1, 1), stride=(1, 1), no_bias=True, name='dim_fc2')
    dim = mx.sym.Reshape(data=dim, shape=(-1, 3))

    # 2. orientation_loc
    orientation_loc = mx.sym.Convolution(data=net, num_filter=256, kernel=(7, 7), stride=(1, 1), no_bias=True,
                                         name='loc_fc1')
    orientation_loc = mx.sym.Convolution(data=orientation_loc, num_filter=2 * CFG.BIN, kernel=(1, 1), stride=(1, 1),
                                         no_bias=True, name='loc_fc1')
    orientation_loc = mx.sym.L2Normalization(data=orientation_loc, mode='channel', name='l2_norm')
    orientation_loc = mx.sym.Reshape(data=orientation_loc, shape=(-1, 2 * CFG.BIN), )


    # 3. orientation_conf
    orientation_conf = mx.sym.Convolution(data=net, num_filter=256, kernel=(7, 7), stride=(1, 1), no_bias=True,
                                          name='conf_fc1')
    orientation_conf = mx.sym.Convolution(data=orientation_conf, num_filter=1 * CFG.BIN, kernel=(1, 1), stride=(1, 1),
                                          no_bias=True, name='conf_fc2')
    orientation_conf = mx.sym.Reshape(data=orientation_conf, shape=(-1, CFG.BIN))

    new_args = dict({k : arg_params[k] for k in arg_params if 'fc1' not in k})
    # print arg_params.keys() # 163
    # print new_args.keys()   # 161, remove fc1_weight, fc1_bias
    # print len(arg_params), len(new_args)

    return dim, orientation_loc, orientation_conf, new_args


import logging
head = '%(asctime)-15s % (message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

# def fit(symbol, arg_params, aux_params, train, val, batch_size, gpus=CFG.GPUs):
#     devs = [mx.gpu(i) for i in gpus]
#     mod = mx.mod.Module(symbol=symbol, context=devs)
#     mod.fit(
#         train,
#         val,
#         num_epoch=8,
#         arg_params=arg_params,
#         aux_params=aux_params,
#         allow_missing=True,
#         batch_end_callback=mx.callback.Speedometer(batch_size, 10),
#         kvstore='device',
#         optimizer='adam',
#         optimizer_params={'learning_rate': 0.0001},
#         initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2),
#         eval_metric='acc'
#     )
#     metric = mx.metric.Accuracy()
#
#     return mod.score(val, metric)

# num_classes = 256
# batch_per_gpu = 16
# num_gpus = 8
#
# dim, orientation_loc, orientation_conf, arg_params = \
#     get_fine_tune_model(sym, arg_params, num_classes)
# print dim, len(arg_params)
# print orientation_loc
# print orientation_conf
# batch_size = batch_per_gpu * num_gpus
# train, val = get_iterators(batch_size)
# mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)
#
# assert mod_score > 0.77, "Low training accuracy"

# data = mx.sym.Variable('data', shape=(24,224,224))
# net = get_backbone_symbol(10, 152, '3,224,224', 32, 256, 'float32')
# mx.viz.plot_network(net[1])
# graph.render()

# data = mx.sym.Variable("data")
# sym = orientation_conf
# arg_shape, output_shape, aux_shape = sym.infer_shape(data=(8, 3, 224, 224))
# # print arg_shape
# print output_shape
# print aux_shape

def orientation_loc_loss(y_true, y_pred):
    pass

def orientation_conf_loss(y_true, y_pred):
    pass

def dimension_loss(y_true, y_pred):
    pass