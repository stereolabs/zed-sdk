# ONNX YOLO Internal Detector

This sample demonstrates how to use custom YOLO models (YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, YOLO26) in ONNX format with the ZED SDK's internal preprocessing, inference engine (using the highly optimized TensorRT framework) and postprocessing for optimal performance.

The ZED SDK optimizes your model using TensorRT and provides 3D object detection capabilities including localization, 3D bounding boxes, and tracking.

## Features

- 3D OpenGL point cloud visualization
- 3D bounding boxes around detected objects
- Configurable object classes and confidence thresholds
- Real-time tracking
- **ZED SDK internal handling of preprocessing, inference and postprocessing**

## Prerequisites

- [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/development/python/install)
- ONNX model file (YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12 or YOLO26)

## Training Custom Models

To train custom detectors with the supported architectures: [Ultralytics Training Guide](https://docs.ultralytics.com/modes/train)

## Exporting Custom Models

To export custom detectors with the supported architectures: [Ultralytics Export Guide](https://docs.ultralytics.com/modes/export)

## Setup (Optional)

For improved data retrieval and handling on GPU:

```bash
# Install CuPy for GPU acceleration
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x

# Install CUDA Python bindings
pip install cuda-python
```

## Usage

This sample expects an ONNX file

```bash
python custom_internal_detector.py --custom_onnx yolo11.onnx [--svo path/to/file.svo]
```

## Additional Resources

- [ZED SDK Documentation - Custom Object Detection](https://www.stereolabs.com/docs/object-detection/custom-od/)
- [Community Support](https://community.stereolabs.com/)