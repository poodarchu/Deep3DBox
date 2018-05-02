import mxnet as mx
import numpy as np
import logging
import time
import subprocess
from kitti_util import *

from mxnet.io import DataBatch, DataDesc

logging.basicConfig(level=logging.DEBUG)

class KITTI_Iter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, data_gen, label_names, label_shapes, label_gen, batch_size, shuffle=True):
        super(KITTI_Iter, self).__init__(batch_size)
        self._provide_data  = list(zip(data_names, data_shapes))
        self._provide_label = list(zip(label_names, label_shapes))
        self.num_batches = 7481 // batch_size
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.batch_size = batch_size
        self.curr_batch = 0
        self.shuffle = shuffle

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self.provide_label

    def reset(self):
        self.curr_batch = 0

    def next(self):
        if self.curr_batch <= self.num_batches:
            self.curr_batch += 1
            data = [mx.nd.array(g(d[1])) for g, d in zip(self._provide_data, self.data_gen)]
            label = [mx.nd.array(g(d[1])) for g, d in zip(self._provide_label, self.label_gen)]
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration

from mxnet.test_utils import get_mnist_iterator


class Deep3DBox_Accuracy(mx.metric.EvalMetric):
    """
    Calculate Accuracies for multi label
    """
    def __init__(self, num=None):
        self.num = num
        super(Deep3DBox_Accuracy, self).__init__('deep3dbox_acc')

    def reset(self):
        return

    def update(self):
        return

    def get(self):
        """
        Get current evaluation result
        :return:
        """
        return

    def get_name_values(self):
        """
        :return: zipped name and value pairs
        """
        return