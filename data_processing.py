import mxnet as mx
import numpy as np
import os, sys
import cv2
import config as CFG
import copy
import random

# evaluate method
def read_det_file(self, det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))
    return id_list, type_list, box2d_list, prob_list

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, image_file, label_file):
        label_file = open(label_file, 'r')

        self.curr_image_objs = []

        self.image_size = [CFG.NORM_H, CFG.NORM_W]

        for label_file_line in label_file.readlines():
            data = label_file_line.split(' ')
            data[1:] = [float(x) for x in data[1:]]

            truncated = data[1]
            occluded = data[2]

            if data[0] in CFG.VEHICLES: # and self.truncated < 0.1 and self.occluded < 0.1:

                # extract label, truncation, occlusion
                self.type = data[0]  # 'Car', 'Pedestrian', ...
                self.truncation = truncated  # truncated pixel ratio [0..1]
                self.occlusion = occluded  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown

                new_alpha = data[3] + np.pi / 2
                if new_alpha < 0:
                    new_alpha = new_alpha + 2. * np.pi
                new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)
                self.alpha = new_alpha  # object observation angle [-pi..pi]

                # extract 2d bounding box in 0-based coordinates
                self.xmin = data[4]  # left
                self.ymin = data[5]  # top
                self.xmax = data[6]  # right
                self.ymax = data[7]  # bottom

                self.box2d = self._random_shift_box2d(np.array([self.xmin, self.ymin, self.xmax, self.ymax]))

                # extract 3d bounding box information
                # self.h = data[8]  # box height
                # self.w = data[9]  # box width
                # self.l = data[10]  # box length (in meters)
                self.dimension = np.array([data[8:11]])  # h, w, l
                self.t = np.array(data[11:14])  # location (x,y,z) in camera coord.
                self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

                obj = {
                    'name': self.type,
                    # 2d bounding box
                    'box2d' : self.box2d,
                    # 3d bounding box dimension
                    'dims': self.dimension,
                    'new_alpha': self.alpha
                }

                # update dims_avg using current object.
                # dims_avg[obj['name']] = dims_cnt[obj['name']] * dims_avg[obj['name']] + obj['dims']
                # dims_cnt[obj['name']] += 1
                # dims_avg[obj['name']] /= dims_cnt[obj['name']]

                self.curr_image_objs.append(obj)

            ###### flip data
            for obj in self.curr_image_objs:
                # Fix dimensions
                obj['dims'] = obj['dims'] # - self.dims_avg[obj['name']]

                # Fix orientation and confidence for no flip
                orientation = np.zeros((CFG.BIN, 2))
                confidence = np.zeros(CFG.BIN)

                anchors = self._compute_anchors(obj['new_alpha'])

                for anchor in anchors:
                    orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
                    confidence[anchor[0]] = 1.

                confidence = confidence / np.sum(confidence)

                obj['orient'] = orientation
                obj['conf'] = confidence

                # Fix orientation and confidence for flip
                orientation = np.zeros((CFG.BIN, 2))
                confidence = np.zeros(CFG.BIN)

                anchors = self._compute_anchors(2. * np.pi - obj['new_alpha'])
                for anchor in anchors:
                    orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
                    confidence[anchor[0]] = 1

                confidence = confidence / np.sum(confidence)

                obj['orient_flipped'] = orientation
                obj['conf_flipped'] = confidence

        self.curr_image_data = self._load_image(image_file)

        self.curr_image_num_objs = len(self.curr_image_objs)

    def _compute_anchors(angle):
        anchors = []

        wedge = 2. * np.pi / CFG.BIN
        l_index = int(angle / wedge)
        r_index = l_index + 1

        if (angle - l_index * wedge) < wedge / 2 * (1 + CFG.OVERLAP / 2):
            anchors.append([l_index, angle - l_index * wedge])
        if (r_index * wedge - angle) < wedge / 2 * (1 + CFG.OVERLAP / 2):
            anchors.append([r_index % CFG.BIN, angle - r_index * wedge])

        return anchors

    def _random_shift_box2d(self, box2d, shift_ratio=0.1):
        ''' Randomly shift box center, randomly scale width and height
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

    def _load_image(self, img_filename):
        return cv2.resize(cv2.imread(img_filename), self.image_size)

    @property
    def get_data(self):
        return self.curr_image_data

    @property
    def get_label(self):
        return self.curr_image_objs # all objects in current image file.


class KITTIITER(mx.io.DataIter):
    def __init__(self, root_dir, batch_size, data_name=['data'], label_name=['d_label', 'o_label', 'c_label'], shuffle=True, flipped=True, split='training'):
        super(KITTIITER, self).__init__(batch_size)
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        self.data_name = data_name
        self.label_name = label_name

        if split == 'training':
            self.num_samples = 7481
            self.image_dir = os.path.join(self.split_dir, 'image_2')
            self.calib_dir = os.path.join(self.split_dir, 'calib')
            self.label_dir = os.path.join(self.split_dir, 'label_2')
        elif split == 'testing':
            self.num_samples = 7518
            self.image_dir = os.path.join(self.split_dir, 'image_2')
            self.calib_dir = os.path.join(self.split_dir, 'calib')
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.dims_avg = {
            'Cyclist': np.array([1.73532436, 0.58028152, 1.77413709]),
            'Van': np.array([2.18928571, 1.90979592, 5.07087755]),
            'Tram': np.array([3.56092896, 2.39601093, 18.34125683]),
            'Car': np.array([1.52159147, 1.64443089, 3.85813679]),
            'Pedestrian': np.array([1.75554637, 0.66860882, 0.87623049]),
            'Truck': np.array([3.07392252, 2.63079903, 11.2190799])
        }

        self.cursor = -1

        self.order = []
        self.data_list = []
        self.label_list = []

        cnt = 0

        for label_file in sorted(os.listdir(self.label_dir)):
            image_file = label_file.replace('txt', 'png')

            self.data_list.append(image_file)
            self.label_list.append(label_file)

            self.order.append(cnt)
            cnt += 1

        self.num_samples = cnt

        self.PER_IMAGE_OBJ_NUM = 10

        if shuffle:
            self._shuffle()

        self.dict = {}

    def _shuffle(self):
        random.shuffle(self.order)


    def __len__(self):
        return self.num_samples

    def _compute_anchors(angle):
        anchors = []

        wedge = 2. * np.pi / CFG.BIN
        l_index = int(angle / wedge)
        r_index = l_index + 1

        if (angle - l_index * wedge) < wedge / 2 * (1 + CFG.OVERLAP / 2):
            anchors.append([l_index, angle - l_index * wedge])
        if (r_index * wedge - angle) < wedge / 2 * (1 + CFG.OVERLAP / 2):
            anchors.append([r_index % CFG.BIN, angle - r_index * wedge])

        return anchors

    def _get_data(self, ):
        assert (self.cursor < self.num_samples), "DataIter needs reset."

        # obj = Object3d(self.image_dir + image_file, self.label_dir + label_file)
        #
        # self.all_objs.append(obj.curr_image_objs)
        # self.all_images.append(obj.curr_image_data)

        PER_IMAGE_OBJ_NUM = self.PER_IMAGE_OBJ_NUM

        data = np.zeros((self.batch_size, 3, self.image_dir[0], self.image_size[1]))
        d_label = np.zeros((self.batch_size, PER_IMAGE_OBJ_NUM, 3))
        o_label = np.zeros((self.batch_size, PER_IMAGE_OBJ_NUM, CFG.BIN, 2))
        c_label = np.zeros((self.batch_size, PER_IMAGE_OBJ_NUM, CFG.BIN))

        if self.cursor + self.batch_size <= self.num_samples:
            for i in range(self.batch_size):
                idx = self.order[self.cursor + i]
                obj = Object3d(os.path.join(self.image_dir, self.data_list[idx]),
                               os.path.join(self.image_dir, self.label_list[idx]))
                data[i] = obj.curr_image_data

                max = PER_IMAGE_OBJ_NUM if PER_IMAGE_OBJ_NUM < obj.curr_image_num_objs else obj.curr_image_num_objs

                for j in range(max):
                    d_label[i][j] = obj.curr_image_objs[j]['dims']
                    o_label[i][j] = obj.curr_image_objs[j]['orient']
                    c_label[i][j] = obj.curr_image_objs[j]['conf']
        else:
            for i in range(self.num_samples - self.cursor):
                idx = self.order[self.cursor + i]
                obj = Object3d(os.path.join(self.image_dir, self.data_list[idx]),
                               os.path.join(self.image_dir, self.label_list[idx]))

                data[i] = obj.curr_image_data

                max = PER_IMAGE_OBJ_NUM if PER_IMAGE_OBJ_NUM < obj.curr_image_num_objs else obj.curr_image_num_objs

                for j in range(max):
                    d_label[i][j] = obj.curr_image_objs[j]['dims']
                    o_label[i][j] = obj.curr_image_objs[j]['orient']
                    c_label[i][j] = obj.curr_image_objs[j]['conf']

            # pad = self.batch_size - self.num_data + self.cursor
            #
            # for i in range(pad):
            #     idx = self.order[i]
            # data_, label_ = self._read_img(self.data_list[idx], self.label_list[idx])
            # data[i + self.num_data - self.cursor] = data_
            # label[i + self.num_data - self.cursor] = label_

        return mx.nd.array(data), mx.nd.array(d_label), mx.nd.array(o_label), mx.ndarray(c_label)

    def _getpad(self):
        if self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0

    def _random_shift_box2d(self, box2d, shift_ratio=0.1):
        ''' Randomly shift box center, randomly scale width and height
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

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.num_data

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            data, d_label, o_label, c_label = self._get_data()
            label = [d_label, o_label, c_label]
            return mx.io.DataBatch(
                data=data,
                label=label,
                pad=self._getpad(),
                index=None,
                provide_data=self.provide_data,
                provide_label=self.provide_label
            )
        else:
            raise StopIteration

    @property
    def provide_data(self):
        return [(self.data_name, (self.batch_size, 3, CFG.NORM_H, CFG.NORM_W))]

    @property
    def provide_label(self):
        return [
            (self.label_name[0], (self.batch_size, self.PER_IMAGE_OBJ_NUM, 3)),
            (self.label_name[1], (self.batch_size, self.PER_IMAGE_OBJ_NUM, CFG.BIN, 2)),
            (self.label_name[3], (self.batch_size, self.PER_IMAGE_OBJ_NUM, CFG.BIN))
        ]

    def reset(self):
        self.cursor = -self.batch_size
        self._shuffle()

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

    # def get_image(self, idx):
    #     assert (idx < self.num_samples)
    #     img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
    #     return load_image(img_filename)
    #
    # def get_label_objects(self, idx):
    #     assert (idx < self.num_samples and self.split == 'training')
    #     label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
    #     return read_label(label_filename)

# def prepare_input_and_output(image_dir, train_inst):
#     # Prepare image batch
#     xmin = train_inst['xmin'] + np.random.randint(-CFG.MAX_JIT, CFG.MAX_JIT + 1)
#     ymin = train_inst['ymin'] + np.random.randint(-CFG.MAX_JIT, CFG.MAX_JIT + 1)
#     xmax = train_inst['xmax'] + np.random.randint(-CFG.MAX_JIT, CFG.MAX_JIT + 1)
#     ymax = train_inst['ymax'] + np.random.randint(-CFG.MAX_JIT, CFG.MAX_JIT + 1)
#
#     img = cv2.imread(image_dir + train_inst['image'])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = copy.deepcopy(img[ymin:ymax + 1, xmin:xmax + 1]).astype(np.float32)
#
#     # Flip the image
#     flip = np.random.binomial(1, .5)
#     if flip > 0.5: img = cv2.flip(img, 1)
#
#     # Resize the image to standard size
#     img = cv2.resize(img, (CFG.NORM_H, CFG.NORM_W))
#     img = img - np.array([[[_R_MEAN, _G_MEAN, _B_MEAN]]])
#
#     # Fix orientation and confidence
#     if flip > 0.5:
#         return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
#     else:
#         return img, train_inst['dims'], train_inst['orient'], train_inst['conf']
#
# def data_gen(image_dir, all_objs, batch_size):
#     num_obj = len(all_objs)
#
#     keys = range(num_obj)
#     np.random.shuffle(keys)
#
#     l_bound = 0
#     r_bound = batch_size if batch_size < num_obj else num_obj
#
#     while True:
#         if l_bound == r_bound:
#             l_bound = 0
#             r_bound = batch_size if batch_size < num_obj else num_obj
#             np.random.shuffle(keys)
#         curr_inst = 0
#         x_batch = np.zeros((r_bound - l_bound, 224, 224, 3))
#         d_batch = np.zeros((r_bound - l_bound, 3))
#         o_batch = np.zeros((r_bound - l_bound, CFG.BIN, 2))
#         c_batch = np.zeros((r_bound - l_bound, CFG.BIN))
#
#         for key in keys[l_bound:r_bound]:
#             # Augment input image and fix object's orientation and confidence
#             image, dimension, orientation, confidence = prepare_input_and_output(image_dir, all_objs[key])
#             x_batch[curr_inst, :] = image
#             d_batch[curr_inst, :] = dimension
#             o_batch[curr_inst, :] = orientation
#             c_batch[curr_inst, :] = confidence
#
#             curr_inst += 1
#         yield x_batch, [d_batch, o_batch, c_batch]
#
#         l_bound = r_bound
#         r_bound = r_bound + batch_size
#
#         if r_bound > num_obj: r_bound = num_obj


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


"""
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

3D Object Detection Benchmark
=============================

The goal in the 3D object detection task is to train object detectors for
the classes 'Car', 'Pedestrian', and 'Cyclist'. The object detectors
must provide BOTH the 2D 0-based bounding box in the image as well as the 3D
bounding box (in the format specified above, i.e. 3D dimensions and 3D locations)
and the detection score/confidence. Note that the 2D bounding box should correspond
to the projection of the 3D bounding box - this is required to filter objects
larger than 25 pixel (height). We also note that not all objects in the point clouds
have been labeled. To avoid false positives, detections not visible on the image plane
should be filtered (the evaluation does not take care of this, see 
'cpp/evaluate_object.cpp'). Similar to the 2D object detection benchmark,
we do not count 'Van' as false positives for 'Car' or 'Sitting Person'
as false positive for 'Pedestrian'. Evaluation criterion follows the 2D
object detection benchmark (using 3D bounding box overlap).

"""

