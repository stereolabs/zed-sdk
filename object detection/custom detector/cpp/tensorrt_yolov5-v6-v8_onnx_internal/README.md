# Yolov8 / v6 / v5 custom ONNX ran in the ZED SDK

This sample is designed to run a state of the art object detection model using the ZED SDK and optimizing your model with the highly optimized TensorRT framework. Internally, the ZED SDK takes its images, run inference on it to obtain 2D box detections and extract 3D informations (localization, 3D bounding boxes) and tracking.

This sample shows how to pass your custom YOLO-like onnx model to the ZED SDK.

A custom detector can be trained with the same architecture. These tutorials walk you through the workflow of training a custom detector :

- Yolov8 https://docs.ultralytics.com/modes/train
- Yolov6 https://github.com/meituan/YOLOv6/blob/main/docs/Train_custom_data.md
- Yolov5 https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

## Getting Started

 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 - [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)

## Workflow

This sample is expecting a TensorRT engine, optimized from an ONNX model. The ONNX model can be exported from Pytorch using the original YOLO code. Please refer to the section corresponding to the needed version.

### Build the sample

 - Build for [Windows](https://www.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://www.stereolabs.com/docs/app-development/cpp/linux/)

### Running the sample with the engine generated

```sh
./yolo_onnx_zed [.onnx] [zed camera id / optional svo filepath]

# For example yolov8n
./yolo_onnx_zed yolov8n.onnx 0      # 0  for zed camera id 0

# With an SVO file
./yolo_onnx_zed yolov8n.onnx ./foo.svo
```
