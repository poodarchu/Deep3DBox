import numpy as np

BIN, OVERLAP = 2, 0.1
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram', 'Pedestrian', 'Cyclist', 'Misc']

RAN = np.pi/BIN
W_H_C_Ratio = [12.5, 12.5, 5.]

ALPHA = 1.
W = 1.

MAX_JIT = 3

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCH = 10

# GPUs = [0,1]

SAVE_PATH='./save_model'
DATA_DIR = './kitti/'
PRETRAINED_DIR = './pretrained/'

BACKBONES = {
    'Inception' : PRETRAINED_DIR+'Inception-BN',
    'resnet50' : PRETRAINED_DIR+'resnet-50',
    'resnet152' : PRETRAINED_DIR+'resnet-152',
    'resnext50' : PRETRAINED_DIR+'resnext-50',
    'resnext101' : PRETRAINED_DIR+'resnext-101-64x4d',
    'vgg16' : PRETRAINED_DIR+'vgg16',
    'vgg19' : PRETRAINED_DIR+'vgg19'
}
