# OpenCV DNN with ZED Custom Box input

This sample is designed to run a state of the art object detection model using OpenCV with the DNN module. The image are taken from the ZED SDK, and the 2D box detections are then ingested into the ZED SDK to extract 3D informations (localization, 3D bounding boxes) and tracking.

The default model is [Yolov4 from the darknet framework](https://github.com/AlexeyAB/darknet).

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 - Install OpenCV with the DNN module 


## Using the sample

After installing OpenCV with DNN module (and preferably CUDA Backend support), this sample can be built.

### Build the program
 - Build for [Windows](https://www.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://www.stereolabs.com/docs/app-development/cpp/linux/)

### Preparing the model

The default model is running Yolov4 trained on COCO dataset, for a custom model the file containing the name and the `NUM_CLASSES` variable should be udpated accordingly.

 - Download the [Yolov4 weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)

```sh
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

In the sample root folder should be present 3 files for the model :

- `coco.names.txt` containing the class name for display, this file is optionnal
- `yolov4.cfg` containing the network definition
- `yolov4.weights` containing the network weights

The sample should be run from this folder to avoid path issue when loading those files.

For instance on Linux:

```sh
mkdir build
cd build
cmake ..
make
cd ../

# ls
#      build
#      CMakeLists.txt
#      coco.names.txt
#      include
#      README.md
#      src
#      yolov4.cfg
#      yolov4.weights

# Running the sample
./build/opencv_dnn_zed
```

The GUI is composed of 2 window, a 2D OpenCV view of the raw detections and a 3D OpenGL view of the ZED SDK output from the OpenCV DNN detection with 3D informations and tracking extracted.