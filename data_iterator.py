import mxnet as mx
import numpy as np
import logging
import time
import subprocess
from kitti_util import *

from mxnet.io import DataBatch, DataDesc

logging.basicConfig(level=logging.DEBUG)

class Kitti_Iterator(mx.io.DataIter):
    """
    Multi-label KITTI iterator.
    """
    def __init__(self, image_dir, label_dir, batch_size=0):
        super(Kitti_Iterator, self).__init__()
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.label_dir = label_dir

    def __iter__(self):
        return self

    def reset(self):
        """Reset the iterator to the begin of the data."""
        pass

    def next(self):
        """Get next data batch from iterator.

        Returns
        -------
        DataBatch
            The data of next batch.

        Raises
        ------
        StopIteration
            If the end of the data is reached.
        """
        if self.iter_next():
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        """Move to the next batch.

        Returns
        -------
        boolean
            Whether the move is successful.
        """
        pass

    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        list of NDArray
            The data of the current batch.
        """
        pass

    def getlabel(self):
        """Get label of the current batch.

        Returns
        -------
        list of NDArray
            The label of the current batch.
        """
        pass

    def getindex(self):
        """Get index of the current batch.

        Returns
        -------
        index : numpy.array
            The indices of examples in the current batch.
        """
        return None

    def getpad(self):
        """Get the number of padding examples in the current batch.

        Returns
        -------
        int
            Number of padding examples in the current batch.
        """
        pass


