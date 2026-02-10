# TensorRT YOLOv8 ONNX Async Detector

This sample demonstrates asynchronous object detection using custom YOLO models (YOLOv5, YOLOv6, YOLOv8) in ONNX format with TensorRT optimization. The asynchronous processing improves performance by running inference in parallel with camera capture.

## Features

- **Asynchronous processing for improved performance**
- **TensorRT optimization for maximum performance**

## Prerequisites

- [ZED SDK](https://www.stereolabs.com/developers/release/)
- [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- CUDA Toolkit

## Workflow

This sample requires a TensorRT engine optimized from an ONNX model. The workflow is the same as the standard TensorRT implementation, but with asynchronous processing enabled.

### Model Preparation (YOLOv8 Recommended)

**Installation:**
```bash
python -m pip install ultralytics
```

**ONNX Export:**
```bash
# Standard model
yolo export model=yolov8n.pt format=onnx simplify=True dynamic=False imgsz=608

# Custom model
yolo export model=yolov8l_custom_model.pt format=onnx simplify=True dynamic=False imgsz=512
```

**TensorRT Engine Generation:**
```bash
# Generate engine from ONNX
./yolo_onnx_zedOne -s yolov8s.onnx yolov8s.engine

# For dynamic dimensions (if exported with dynamic=True)
./yolo_onnx_zedOne -s yolov8s.onnx yolov8s.engine images:1x3x608x608
```

## Build and Usage

### Building the Sample
- [Build for Windows](https://www.stereolabs.com/docs/app-development/cpp/windows/)
- [Build for Linux/Jetson](https://www.stereolabs.com/docs/app-development/cpp/linux/)

### Running the Sample
```bash
./yolo_onnx_zedOne [.engine] [zed camera id / optional svo filepath]

# Examples:
./yolo_onnx_zedOne yolov8n.engine 0           # ZED camera ID 0
./yolo_onnx_zedOne yolov8n.engine ./foo.svo   # SVO file
```

## Training Custom Models

For training custom detectors with the supported architectures:
- **YOLOv8**: [Ultralytics Training Guide](https://docs.ultralytics.com/modes/train)
- **YOLOv6**: [YOLOv6 Custom Training](https://github.com/meituan/YOLOv6/blob/main/docs/Train_custom_data.md)
- **YOLOv5**: [YOLOv5 Custom Training](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

## Additional Resources

- [ZED SDK Documentation - Custom Object Detection](https://www.stereolabs.com/docs/object-detection/custom-od/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [Community Support](https://community.stereolabs.com/)