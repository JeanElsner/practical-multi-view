# Multi-Camera Computer Vision and Algorithms
This is an implementation for the practical course *Multi-Camera Computer Vision and Algorithms* at TUM.
## Demonstration
Watch the pipeline run on the KITTI benchmark on [Youtube](https://www.youtube.com/watch?v=9_oE1G4kKxU)
## Requirements

* OpenCV 3.3 or later

* Ceres 1.13 or later

* Dlib 19.9 or later
## Build
```
mkdir build
cdir build
cmake ..
cmake --build .
```
## Configuration
Run binary with path to configuration file as argument. The configuration may look like this:
```
[Settings]
fancy_video = 1
verbose     = 1
video_path  = ../../../tracker.avi
[Odometry]
; When extracting 2d features, tries to extract at least this amount
min_tracked_features = 400
; Tolerated number of seen 3d points before triangulating new features
tracked_features_tol = 150
; Number of frames used to initialise odometry pipeline
init_frames          = 5
; Number of frames to track
frames               = 500
; Number of frames used for bundle adjustment
bundle_size          = 20
[ceres]
max_iterations = 2
[KITTI]
image_dir          = D:\Odometry\dataset\sequences\07\image_0
; Number of camera in calibration file
camera             = 0
camera_calibration = D:\Odometry\dataset\sequences\07\calib.txt
poses              = D:\Odometry\dataset\poses\07.txt
```