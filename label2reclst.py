# coding: utf-8

"""
Convert the original KITTI label to Format:

Write the the image list files as follows:
$ integer_image_index \t label_1 \t label_2 \t label_3 \t label_4 \t path_to_image
When use im2rec tools, add a ‘label_width=4’ to the command argument, e.g.
# ./bin/im2rec image.lst image_root_dir output.bin resize=256 label_width=4
In your iterator generation code, set label_width=4 and path_imglist=<<The PATH TO YOUR image.lst>>, e.g.
dataiter = mx.io.ImageRecordIter(
  path_imgrec="data/cifar/train.rec",
  data_shape=(3,28,28),
  path_imglist="data/cifar/image.lst",
  label_width=4
)
Then you’re all set for a multi-label image iterator.
"""

import os

DATA_DIR = './data/kitti'
target = 'training'

labels_dir = os.path.join(DATA_DIR, target, 'label_2')
images_dir = os.path.join(DATA_DIR, target, 'image_2')

fout = open(DATA_DIR+'/train.lst')
for label in os.listdir(labels_dir):
    image = label.replace('txt', 'png')
    fout.write()