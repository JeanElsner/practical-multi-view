#Multi-Camera Computer Vision and Algorithims
This is an implementation for the practical course *Multi-Camera Computer Vision and Algorithms* at TUM.
## Requirements

* OpenCV

* Ceres

* Dlib
## Configuration
Run binary with path to configuration file as argument. The configuration may look like this:
´´´
[Settings]
fancy_video = 1
verbose     = 1
[Odometry]
min_tracked_features = 400
tracked_features_tol = 150
init_frames          = 5
frames               = 200
bundle_size          = 0
[ceres]
max_iterations = 5
[KITTI]
image_dir          = D:\Odometry\dataset\sequences\07\image_0
camera             = 0
camera_calibration = D:\Odometry\dataset\sequences\07\calib.txt
poses              = D:\Odometry\dataset\poses\07.txt
´´´