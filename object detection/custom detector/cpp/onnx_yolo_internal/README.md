# YOLO26 / v12 / v11 / v10 / v9 / v8 / v5 custom ONNX ran in the ZED SDK

This sample demonstrates how to use custom YOLO models (YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, YOLO26) in ONNX format with the ZED SDK's internal preprocessing, inference engine (using the highly optimized TensorRT framework) and postprocessing for optimal performance.

The ZED SDK optimizes your model using TensorRT and provides 3D object detection capabilities including localization, 3D bounding boxes, and tracking.

A custom detector can be trained with the same architecture. Please refer to [Ultralytics Train](https://docs.ultralytics.com/modes/train/)

## Features

- 3D OpenGL point cloud visualization
- 3D bounding boxes around detected objects
- Configurable object classes and confidence thresholds
- Real-time tracking
- **ZED SDK internal handling of preprocessing, inference and postprocessing**

## Getting Started

 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 - [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)

## Workflow

This sample is expecting an ONNX exported using the original YOLO code. Please refer to [Ultralytics Export](https://www.stereolabs.com/docs/yolo/export).

### Build the sample

 - Build for [Windows](https://www.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://www.stereolabs.com/docs/app-development/cpp/linux/)

### Running the sample with the engine generated

```sh
./yolo_onnx_zed [.onnx] [zed camera id / optional svo filepath]

# For example yolo26m
./yolo_onnx_zed yolo26m.onnx 0      # 0  for zed camera id 0

# With an SVO file
./yolo_onnx_zed yolo26m.onnx ./foo.svo
```

## Additional Resources

- [ZED SDK Documentation - Custom Object Detection](https://www.stereolabs.com/docs/object-detection/custom-od/)
If you need assistance go to our Community site at https://community.stereolabs.com/
- [Community Support](https://community.stereolabs.com/)
