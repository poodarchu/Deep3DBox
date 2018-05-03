import mxnet as mx
import numpy as np
import os, sys
import cv2
import config as CFG
import copy


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, image_file, label_file_line):
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
        # self.h = data[8]  # box height
        # self.w = data[9]  # box width
        # self.l = data[10]  # box length (in meters)
        self.dimension = [data[8], data[9], data[10]] # h, w, l
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        self.image_data = self.load_image(image_file)

    def load_image(self, img_filename):
        return cv2.imread(img_filename)

    @property
    def get_data(self):
        return self.image_data

    @property
    def get_label(self):
        return [self.dimension, self.ry]


class KITTIITER(mx.io.DataIter):
    def __init__(self, root_dir, batch_size, shuffle=True, flipped=True, split='training'):
        super(KITTIITER, self).__init__(batch_size)
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

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

    def __len__(self):
        return self.num_samples

    def random_shift_box2d(self, box2d, shift_ratio=0.1):
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

    def next(self):
        return

    @property
    def provide_data(self):
        return

    @property
    def provide_label(selfs):
        return



    # def get_image(self, idx):
    #     assert (idx < self.num_samples)
    #     img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
    #     return load_image(img_filename)
    #
    # def get_label_objects(self, idx):
    #     assert (idx < self.num_samples and self.split == 'training')
    #     label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
    #     return read_label(label_filename)




    def __init__2(self):
        def compute_anchors(angle):
            anchors = []

            wedge = 2. * np.pi / CFG.BIN
            l_index = int(angle / wedge)
            r_index = l_index + 1

            if (angle - l_index * wedge) < wedge / 2 * (1 + CFG.OVERLAP / 2):
                anchors.append([l_index, angle - l_index * wedge])
            if (r_index * wedge - angle) < wedge / 2 * (1 + CFG.OVERLAP / 2):
                anchors.append([r_index % CFG.BIN, angle - r_index * wedge])

            return anchors

        def parse_annotation(label_dir, image_dir):
            all_objs = []
            dims_avg = {key: np.array([0, 0, 0]) for key in CFG.VEHICLES}
            dims_cnt = {key: 0 for key in CFG.VEHICLES}

            for label_file in sorted(os.listdir(label_dir)):
                image_file = label_file.replace('txt', 'png')
                for line in open(label_dir + label_file).readlines():
                    line = line.strip().split(' ')
                    truncated = np.abs(float(line[1]))
                    occluded = np.abs(float(line[2]))

                    if line[0] in CFG.VEHICLES and truncated < 0.1 and occluded < 0.1:
                        new_alpha = float(line[3]) + np.pi / 2
                        if new_alpha < 0:
                            new_alpha = new_alpha + 2. * np.pi
                        new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)

                        obj = {
                            'name': line[0],
                            # 2d bounding box
                            'xmin': int(float(line[4])),
                            'ymin': int(float(line[5])),
                            'xmax': int(float(line[6])),
                            'ymax': int(float(line[7])),
                            # 3d bounding box dimension
                            'dims': np.array([float(number) for number in line[8:11]]),
                            'new_alpha': new_alpha
                        }

                        # update dims_avg using current object.
                        dims_avg[obj['name']] = dims_cnt[obj['name']] * dims_avg[obj['name']] + obj['dims']
                        dims_cnt[obj['name']] += 1
                        dims_avg[obj['name']] /= dims_cnt[obj['name']]

                        all_objs.append(obj)

            # Flip data
            for obj in all_objs:
                # Fix dimensions
                obj['dims'] = obj['dims'] - dims_avg[obj['name']]

                # Fix orientation and confidence for no flip
                orientation = np.zeros((CFG.BIN, 2))
                confidence = np.zeros(CFG.BIN)

                anchors = compute_anchors(obj['new_alpha'])

                for anchor in anchors:
                    orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
                    confidence[anchor[0]] = 1.

                confidence = confidence / np.sum(confidence)

                obj['orient'] = orientation
                obj['conf'] = confidence

                # Fix orientation and confidence for flip
                orientation = np.zeros((CFG.BIN, 2))
                confidence = np.zeros(CFG.BIN)

                anchors = compute_anchors(2. * np.pi - obj['new_alpha'])

                for anchor in anchors:
                    orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
                    confidence[anchor[0]] = 1

                confidence = confidence / np.sum(confidence)

                obj['orient_flipped'] = orientation
                obj['conf_flipped'] = confidence

            return all_objs

        anchors = compute_anchors(data[14])

        def prepare_input_and_output(image_dir, train_inst):
            # Prepare image batch
            xmin = train_inst['xmin'] + np.random.randint(-CFG.MAX_JIT, CFG.MAX_JIT + 1)
            ymin = train_inst['ymin'] + np.random.randint(-CFG.MAX_JIT, CFG.MAX_JIT + 1)
            xmax = train_inst['xmax'] + np.random.randint(-CFG.MAX_JIT, CFG.MAX_JIT + 1)
            ymax = train_inst['ymax'] + np.random.randint(-CFG.MAX_JIT, CFG.MAX_JIT + 1)

            img = cv2.imread(image_dir + train_inst['image'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = copy.deepcopy(img[ymin:ymax + 1, xmin:xmax + 1]).astype(np.float32)

            # Flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: img = cv2.flip(img, 1)

            # Resize the image to standard size
            img = cv2.resize(img, (CFG.NORM_H, CFG.NORM_W))
            img = img - np.array([[[_R_MEAN, _G_MEAN, _B_MEAN]]])

            # Fix orientation and confidence
            if flip > 0.5:
                return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
            else:
                return img, train_inst['dims'], train_inst['orient'], train_inst['conf']

        def data_gen(image_dir, all_objs, batch_size):
            num_obj = len(all_objs)

            keys = range(num_obj)
            np.random.shuffle(keys)

            l_bound = 0
            r_bound = batch_size if batch_size < num_obj else num_obj

            while True:
                if l_bound == r_bound:
                    l_bound = 0
                    r_bound = batch_size if batch_size < num_obj else num_obj
                    np.random.shuffle(keys)
                curr_inst = 0
                x_batch = np.zeros((r_bound - l_bound, 224, 224, 3))
                d_batch = np.zeros((r_bound - l_bound, 3))
                o_batch = np.zeros((r_bound - l_bound, CFG.BIN, 2))
                c_batch = np.zeros((r_bound - l_bound, CFG.BIN))

                for key in keys[l_bound:r_bound]:
                    # Augment input image and fix object's orientation and confidence
                    image, dimension, orientation, confidence = prepare_input_and_output(image_dir, all_objs[key])
                    x_batch[curr_inst, :] = image
                    d_batch[curr_inst, :] = dimension
                    o_batch[curr_inst, :] = orientation
                    c_batch[curr_inst, :] = confidence

                    curr_inst += 1
                yield x_batch, [d_batch, o_batch, c_batch]

                l_bound = r_bound
                r_bound = r_bound + batch_size

                if r_bound > num_obj: r_bound = num_obj

        pass

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

