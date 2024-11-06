# ZED SDK - Instance Detection

This sample shows how to do instance segmentation using the official Pytorch implementation of YOLOv8-seg from a ZED camera and ingest them into the ZED SDK to extract 3D informations and tracking for each objects.

## Getting Started

 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/object-detection/custom-od/)

## Setting up

 - Install ultralytics using pip

```sh
pip install ultralytics
```

## Run the program

```
# With a SVO file
python detector.py --weights yolov8s-seg.pt --img_size 640 --conf_thres 0.4 --svo path/to/file.svo

# With a plugged ZED Camera
python detector.py --weights yolov8s-seg.pt --img_size 640 --conf_thres 0.4
```

### Features

 - The camera point cloud is displayed in a 3D OpenGL view
 - 3D bounding boxes around detected objects are drawn
 - Objects classes and confidences can be changed

## Training your own model

This sample can use any model trained with YOLOv8-seg, including custom trained one. For a getting started on how to trained a model on a custom dataset with YOLOv8-seg, see here https://docs.ultralytics.com/tutorials/train-custom-datasets/

## Support

If you need assistance go to our Community site at https://community.stereolabs.com/
