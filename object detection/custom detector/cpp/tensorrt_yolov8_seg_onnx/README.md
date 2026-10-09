# ZED SDK - TensorRT YOLOv8 Instance Segmentation ONNX Detector

This sample demonstrates instance segmentation using custom YOLOv8-seg models in ONNX format with TensorRT optimization. The detected 2D boxes and predicted instance masks are ingested into the ZED SDK to extract 3D information including localization, 3D bounding boxes, and tracking.

## Features

- 3D OpenGL point cloud visualization
- 3D bounding boxes around detected objects
- Configurable object classes and confidence thresholds
- Real-time tracking
- **Instance segmentation with precise object boundaries**
- **TensorRT optimization for maximum performance**

## Prerequisites

- [ZED SDK](https://www.stereolabs.com/developers/release/)
- [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- CUDA Toolkit

## Workflow

This sample requires a TensorRT engine optimized from an ONNX model exported from YOLOv8-seg.

### Model Preparation

**Installation:**
```bash
python -m pip install ultralytics
```

**ONNX Export:**
```bash
# Standard segmentation model
yolo export model=yolov8s-seg.pt format=onnx simplify=True dynamic=False imgsz=640

# Custom segmentation model
yolo export model=yolov8s-seg_custom_model.pt format=onnx simplify=True dynamic=False imgsz=640
```

**TensorRT Engine Generation:**

TensorRT applies heavy optimization by processing the network structure and benchmarking all available implementations. This process takes a few minutes but should be done only once per model change.

*Note: This sample requires a fixed size and squared images (e.g., 640x640).*

```bash
# Implicit input size
./yolov8_seg_onnx_zed -s yolov8s-seg.onnx yolov8s-seg.engine

# Explicit input size
./yolov8_seg_onnx_zed -s yolov8s-seg.onnx yolov8s-seg.engine images:1x3x640x640
```

## Build and Usage

### Building the Sample
- [Build for Windows](https://docs.stereolabs.com/docs/app-development/cpp/windows/)
- [Build for Linux/Jetson](https://docs.stereolabs.com/docs/app-development/cpp/linux/)

### Running the Sample
```bash
./yolov8_seg_onnx_zed [.engine] [zed camera id / optional svo filepath]

# Examples:
./yolov8_seg_onnx_zed yolov8s-seg.engine           # ZED camera (auto-detect)
./yolov8_seg_onnx_zed yolov8s-seg.engine ./foo.svo # SVO file
```

## Training Custom Models

For training custom YOLOv8 segmentation models, see:
- [Ultralytics Training Guide](https://docs.ultralytics.com/modes/train)
- [YOLOv8 Instance Segmentation Training](https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolov8-instance-segmentation-on-custom-dataset.ipynb)

## Additional Resources

- [ZED SDK Documentation - Custom Object Detection](https://docs.stereolabs.com/docs/object-detection/custom-od/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [Community Support](https://community.stereolabs.com/)
