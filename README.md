# Deep3DBox
Deep3DBox's MXNet implementation.

1. Train Faster RCNN on KITTI to gain 2D BBox predictions.
2. Add Jitter to GT BBox, then use it and the above generated bbox to train Deep3DBox.
3. inherit IMDB to create KITTI data iter, provide_data and provide_label
4. load pre-trained res-next101, train the model.
5. Use cpp devkit to evaluate the model, use MATLAB to visualize.
