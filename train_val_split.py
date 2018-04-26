import shutil
import sys, os

sys.path.append(os.path.abspath(__file__))

with open('ImageSets/val.txt') as val:
    for index in val.readlines():
        index = index.strip()
        src = 'training/calib/%s.txt' % index
        dst = 'validation/calib/%s.txt' % index
        shutil.move(src, dst)
        # os.rename('training/label_2/'+index+'.txt', 'validation/label_2/'+index+'.txt')
        # os.rename('training/calib/'+index+'.txt', 'validation/calib/'+index+'.txt')