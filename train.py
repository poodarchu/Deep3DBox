import os
import sys
import cv2
import mxnet as mx
from mxnet import gluon
import numpy as np
from mxnet.gluon import nn
from mxnet import autograd
import config as CFG
import time
from random import shuffle
from data_processing import *
import argparse
from tqdm import tqdm
from mxnet import gluon
from wget import download

import logging

import config as CFG
from deep3dbox import *

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


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Deep 3D BBox')
    parser.add_argument('--mode', default='test', help='train or test', dest='mode')
    parser.add_argument('--image', help='Image Path', dest='image')
    parser.add_argument('--label', help='Label Path')
    parser.add_argument('--box2d', help='2d detection path')
    parser.add_argument('--output', dest='output', help='Output Path', default=CFG.DATA_DIR+'validation/result_2')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')

    parser.add_argument('train_path_anno_list', type=str,
                        help='train_path_anno_list')
    parser.add_argument('train_path_imgrec', type=str,
                        help='train_path_imgrec')
    parser.add_argument('val_path_anno_list', type=str,
                        help='val_path_anno_list')
    parser.add_argument('val_path_imgrec', type=str,
                        help='val_path_imgrec')
    parser.add_argument('--gpus', type=str, default='',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('batch_size', type=int,
                        help='batch-size')
    parser.add_argument('--lr-factor', type=float, default=0.1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load')
    parser.add_argument('save_model_prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--log-file', type=str,
                        help='the name of log file')
    parser.add_argument('--log-dir', type=str, default="./log",
                        help='directory of the log file')
    parser.add_argument('--begin_num_update', type=int, default=0,
                        help="begin_num_update")
    parser.add_argument('--load-param', type=str,
                        help="load the pretrained model")

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    devs = mx.cpu() if (args.gpu is None or args.gpu == '') else [mx.gpu(int(i)) for i in args.gpu.split(',')]
    print('devs: ', devs)

    # kvstore
    kv = mx.kvstore.create('local')

    # head = '%(asctime)-15s % (message)s'
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    begin_epoch = 0

    arg_params = None
    aux_params = None
    allow_missing_param = False

    # if args.load_param is not None:
    #     logging.info('loading pretrained model from %s ...' %(args.load_param))
    #     sym, arg_params, aux_params = load_model(CFG.BACKBONES[model_prefix], begin_epoch)
    #     allow_missing_param = True
    if args.load_epoch is not None:
        assert model_prefix is not None
        begin_epoch = args.load_epoch
        logging.info('loading model from %s-%d ...' (model_prefix, begin_epoch))
        sym, arg_params, aux_params = load_model(CFG.BACKBONES[model_prefix], begin_epoch)
    else:
        arg_params = None
        aux_params = None

    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = CFG.SAVE_PATH
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    initializer = mx.initializer.Xavier(factor_type='avg', magnitude=0.5)

    # opt_param for sgd
    optimizer_params = {
        'learning_rate' : 0.0001,
        'wd'            : 0.00005,
        'gamma1'        : 0.9,
        'gamma2'        : 0.5,
        'clip_gradient' : 5
    }

    # opt_param for rmsprop


    epoch_size = 60000

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
    if 'lr_factor' in args and args.lr_factor < 1:
        optimizer_params['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step=[70000, 140000, 200000],
            factor=0.1
        )

    optimizer_params['begin_num_update'] = args.begin_num_update
    optimizer_params['global_clip'] = True

    data = mx.sym.Variable('data')        # , shape=(-1, 224, 224, 3), dtype=np.float32)
    d_label = mx.sym.Variable('d_label')  # , shape=(-1, 3), dtype=np.float32)
    o_label = mx.sym.Variable('o_label')  # , shape=(-1, CFG.BIN, 2), dtype=np.float32)
    c_label = mx.sym.Variable('c_label')  # , shape=(-1, CFG.BIN), dtype=np.float32)

    new_sym, new_args, d_loss, o_loss, c_loss, total_loss = \
        get_symbol_detection(data, 'resnext50', d_label, o_label, c_label, is_train=True)

    data_names = ['data']
    label_names = ['d_label', 'o_label', 'c_label']

    train, val = 1

    fit(new_sym, initializer, optimizer_params, new_args, aux_params, train, val, CFG.BATCH_SIZE, devs)

    iters = 7481 // CFG.BATCH_SIZE

    mod = mx.mod.Module(symbol=new_sym, context=devs)

    for epoch in range(CFG.EPOCH):
        epoch_loss = np.zeros((iters, 1), dtype=np.float32)
        tStart_epoch = time.time()
        batch_loss = 0.0
        for num_iters in tqdm(range(iters), ascii=True, desc='Epoch'+str(epoch+1)+' : Loss:'+str(batch_loss)):
            train_img, train_label = train.next()
            _, batch_loss = mod.fit()
            epoch_loss[num_iters] = batch_loss

        # save model
        if (epoch+1) % 5 == 0:
            pass

        # Print some information
        print("Epoch:", epoch + 1, " done. Loss:", np.mean(epoch_loss))
        tStop_epoch = time.time()
        print("Epoch Time Cost:", round(tStop_epoch - tStart_epoch, 2), "s")




