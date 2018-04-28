import mxnet as mx
import numpy as np
import logging
import time
import subprocess


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
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        label = batch.label[0]

        return mx.io.DataBatch(
            data=batch.data,
            label=[label, label],
            pad=batch.pad,
            index=batch.index
        )


class Multi_Accuarcy(mx.metric.EvalMetric):
    """
    Calculate Accuracy of multi label.
    """
    def __init__(self, num=None):
        self.num = num
        super(Multi_Accuarcy, self).__init__('multi-accuracy')

    def reset(self):
        self.num_inst = 0 if self.num is None else [0]*self.num
        self.sum_metric = 0.0 if self.num is None else [0.0]*self.num

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)
        if self.num is None:
            assert len(labels) == self.num

        for i in range(len(labels)):
            pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            if self.num is None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst = len(pred_label.flat)
            else:
                self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                self.num_inst[i] += len(pred_label.flat)

    def get(self):
        pass

    def get_name_value(self):
        pass







