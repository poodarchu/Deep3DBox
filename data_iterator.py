import mxnet as mx
from mxnet.test_utils import get_mnist_iterator
import numpy as np
import logging
import time

logging.basicConfig(level=logging.DEBUG)

class Kitti_Iterator(mx.io.DataIter):
    """
    Multi-label KITTI iterator.
    """
    def __init__(self, data_iter):
        super(Kitti_Iterator, self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label[0]
        # d_label = self.data_iter.provide_label[0]
        # o_label = self.data_iter.provide_label[1]
        # c_label = self.data_iter.provide_label[2]
        return [
            ('d_label', provide_label[0]),
            ('o_label', provide_label[1]),
            ('c_label', provide_label[2])
        ]

    def hard_reset(self):
        pass

    def reset(self):
        self.data_iter.reset()



