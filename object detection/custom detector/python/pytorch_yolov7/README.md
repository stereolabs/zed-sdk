# ZED SDK - Object Detection

This sample shows how to detect custom objects using the official Pytorch implementation of YOLOv5 from a ZED camera and ingest them into the ZED SDK to extract 3D informations and tracking for each objects.

## Getting Started

 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/object-detection/custom-od/)

## Setting up

 - Clone Yolov7 into the current folder

```sh
git clone https://github.com/WongKinYiu/yolov7
# Install the dependencies if needed
cd yolov7
pip install -r requirements.txt
```

- Download a model file (or prepare your own) https://github.com/WongKinYiu/yolov7/releases/tag/v0.1

```
# Downloading by commmand line
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

## Run the program

*NOTE: The ZED v1 is not compatible with this module*

```
python detector.py --weights yolov7.pt # [--img_size 512 --conf_thres 0.1 --svo path/to/file.svo]
```

### Features

 - The camera point cloud is displayed in a 3D OpenGL view
 - 3D bounding boxes around detected objects are drawn
 - Objects classes and confidences can be changed

## Training your own model

This sample can use any model trained with YOLOv7, including custom trained one. For a getting started on how to trained a model on a custom dataset with YOLOv7, see here example notebook https://colab.research.google.com/drive/1X9A8odmK4k6l26NDviiT6dd6TgR-piOa 

## Support

If you need assistance go to our Community site at https://community.stereolabs.com/