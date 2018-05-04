import mxnet as mx
import numpy as np
import logging
import time
import subprocess
import os, sys
import cv2
from mxnet.io import DataBatch, DataDesc

# from rcnn.rcnn.dataset import kitti

logging.basicConfig(level=logging.DEBUG)

class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))



def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


def load_image(img_filename):
    return cv2.imread(img_filename)

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3


class kitti_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.label_dir = os.path.join(self.split_dir, 'label_2')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
        return load_image(img_filename)

    def get_label_objects(self, idx):
        assert (idx < self.num_samples and self.split == 'training')
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return read_label(label_filename)


def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio
    xmin ,ymin ,xmax ,ymax = box2d
    h = ymax -ymin
    w = xmax -xmin
    cx = (xmin +xmax ) /2.0
    cy = (ymin +ymax ) /2.0
    cx2 = cx + w* r * (np.random.random() * 2 - 1)
    cy2 = cy + h * r * (np.random.random() * 2 - 1)
    h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    return np.array([cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0])


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)


def demo():
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/kitti'))
    data_idx = 0

    objects = dataset.get_label_objects(data_idx)
    objects[0].print_object()

    img = dataset.get_image(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
    print('Image Shape: ', img.shape)


def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3,7)]))
    return id_list, type_list, box2d_list, prob_list


def write_2d_rgb_detection(det_filename, split, result_dir):
    ''' Write 2D detection results for KITTI evaluation.
        Convert from Wei's format to KITTI format.

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        result_dir: string, folder path for results dumping
    Output:
        None (will write <xxx>.txt files to disk)
    Usage:
        write_2d_rgb_detection("val_det.txt", "training", "results")
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    # map from idx to list of strings, each string is a line without \n
    results = {}
    for i in range(len(det_id_list)):
        idx = det_id_list[i]
        typename = det_type_list[i]
        box2d = det_box2d_list[i]
        prob = det_prob_list[i]
        output_str = typename + " -1 -1 -10 "
        output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
        output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()


class KITTI_Iter(mx.io.DataIter):
    def __init__(self, req, image_path, label_path, devkit_path, batch_size, shuffle=True):
        super(KITTI_Iter, self).__init__(batch_size)
        self._provide_data  = self._load_kitti_data(devkit_path+image_path)
        self._provide_label = self._load_kitti_annotation(devkit_path+label_path)
        self._devkit_path = devkit_path
        self.classes = ['Car', 'Pedestrian', 'Cyclist']
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        for index in self._load_image_set_index():
            self._load_kitti_annotation(index=index)

        self.req = req
        if req == 'train':
            self.num_batches = 7481 // batch_size
        elif req == 'test':
            self.num_batches = 7481 // batch_size
        self.batch_size = batch_size
        self.curr_batch = 0
        self.shuffle = shuffle

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def random_shift_box2d(self, box2d, shift_ratio=0.1):
        ''' Randomly Jitter box center, randomly scale width and height
        '''
        r = shift_ratio
        xmin, ymin, xmax, ymax = box2d
        h = ymax - ymin
        w = xmax - xmin
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cx2 = cx + w * r * (np.random.random() * 2 - 1)
        cy2 = cy + h * r * (np.random.random() * 2 - 1)
        h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
        w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
        return np.array([cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0])

    def _load_image_set_index(self):
        """
        load the indexes listed in this dataset's image set file
        :return:
        """
        if self.req == 'test':
            image_index = ['{:0>6}'.format(x) for x in range(0, 7481)]
        elif self.req == 'train':
            image_index = ['{:0>6}'.format(x) for x in range(0, 7481)]  # 80% data

        return image_index

    def _load_kitti_data(self, image_path):
        return

    @property
    def provide_data(self):
        return self._provide_data

    def _load_kitti_annotation(self, index):
        """
        Load image and bounding boxes from TXT file into kitti format.
        :param index:
        :return:
        """
        if self._image_set == 'test':
            image_name = os.path.join(self._devkit_path, 'testing/image_2', index+'.png')
            img = cv2.imread(image_name)
            width = img.shape[0]
            height = img.shape[1]

            return {
                'image'    : image_name,
                'height'   : height,
                'width'    : width,
                'flipped'  : False,
                'is_train' : False
            }
        filename = os.path.join(self._devkit_path, 'training/label_2', index+'.txt')
        imagename = os.path.join(self._devkit_path, 'training/label_2', index+'.png')
        img = cv2.imread(imagename)
        width = img.shape[0]
        height = img.shape[1]
        f = open(filename)
        lines = f.readlines()
        num_objs = 0
        for l in lines:
            str_cls = l.split()
            cls = str(str_cls[0])
            if cls in self.classes or cls in ['Truck', 'Tram', 'Van']:
                num_objs += 1
        num_objs = num_objs

        boxes_2d = np.zeros((num_objs, 4), dtype=np.uint16)
        dimensions = np.zeros((num_objs, 3), dtype=np.unit16)
        translations = np.zeros((num_objs, 3), dtype=np.float32)
        ryaws = np.zeros((num_objs,), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        ix = 0

        for line in lines:
            data = line.split(' ')
            data[1:] = [float(x) for x in data[1:]]

            # extract label, truncation, occlusion
            if str(data[0]) in ['Truck', 'Tram', 'Van']:
                data[0] = 'Car'
            if str(data[0]) not in self._classes:
                continue

            type = data[0]  # 'Car', 'Pedestrian', ...
            truncation = data[1]  # truncated pixel ratio [0..1]
            occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
            alpha = data[3]  # object observation angle [-pi..pi]

            # extract 2d bounding box in 0-based coordinates
            xmin = int(data[4])
            ymin = int(data[5])
            xmax = int(data[6])
            ymax = int(data[7])
            cls = self._class_to_ind[data[0]]  # class 0, 1, 2 for car, pedestrian and cyclist

            # extract 3d bounding box information
            h = data[8]  # box height
            w = data[9]  # box width
            l = data[10]  # box length (in meters)
            t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
            ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

            boxes_2d[ix, :] = [xmin,ymin, xmax, ymax]
            dimensions[ix, :] = [h, w, l]
            translations[ix, :] = t
            ryaws[ix, :] = ry
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (xmax-xmin+1)*(ymax-ymin+1)

            ix = ix + 1

        return [
            ('image', imagename),
            ('height', height),
            ('width', width),
            ('boxes_2d', boxes_2d),
            ('dimensions', dimensions),
            ('yaws', ryaws),
            ('gt_classes', gt_classes),
            ('gt_overlaps', overlaps),
            ('max_overlaps', overlaps.max(axis=1)),
            ('max_classes', overlaps.argmax(axis=1)),
            ('flipped', False),
            ('seg_areas', seg_areas),
            ('is_train', True)
        ]

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


