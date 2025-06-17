# ZED SDK - Yolov8 / v6 / v5 custom ONNX ran in the ZED SDK

This sample is designed to run a state of the art object detection model using the ZED SDK and optimizing your model with the highly optimized TensorRT framework. Internally, the ZED SDK takes its images, run inference on it to obtain 2D box detections and extract 3D informations (localization, 3D bounding boxes) and tracking.

This sample shows how to pass your custom YOLO-like onnx model to the ZED SDK.

A custom detector can be trained with the same architecture. These tutorials walk you through the workflow of training a custom detector :

- Yolov8 https://docs.ultralytics.com/modes/train
- Yolov6 https://github.com/meituan/YOLOv6/blob/main/docs/Train_custom_data.md
- Yolov5 https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

## Getting Started

 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/object-detection/custom-od/)

## Workflow

This sample is expecting an ONNX file exported using the original YOLO code. Please refer to the section corresponding to the needed version.

## Run the program

```
python custom_internal_detector.py --custom_onnx yolov8m.onnx # [--svo path/to/file.svo]
```

### Features

 - The camera point cloud is displayed in a 3D OpenGL view
 - 3D bounding boxes around detected objects are drawn
 - Objects classes and confidences can be changed

## Support

If you need assistance go to our Community site at https://community.stereolabs.com/