# PyTorch YOLO Segmentation Detector

This sample demonstrates instance segmentation using the official PyTorch implementation of YOLO segmentation models (YOLOv8-seg, YOLOv9-seg, YOLO11-seg, YOLO26-seg) with a ZED camera. The detected instances are ingested into the ZED SDK to extract 3D information and tracking.

## Features

- 3D OpenGL point cloud visualization
- 3D bounding boxes around detected objects
- Configurable object classes and confidence thresholds
- Real-time tracking
- **Instance segmentation with precise object boundaries**

## Prerequisites

- [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/development/python/install)
- *Note: ZED v1 is not compatible with this module*

## Setup

```bash
# Install ZED SDK and Python API
# Download from: https://www.stereolabs.com/developers/release/

# Install YOLO dependencies
pip install ultralytics

# Optional: GPU acceleration of data processing and visualization
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
pip install cuda-python
```

## Usage

```bash
python detector.py --weights yolov8s-seg.pt [--img_size 640 --conf_thres 0.4 --svo path/to/file.svo]
```

## Training Custom Models

This sample supports any model trained with YOLO segmentation (YOLOv8-seg, YOLOv9-seg, YOLO11-seg, YOLO26-seg), including custom trained models. For training on custom datasets, see the [Ultralytics Training Guide](https://docs.ultralytics.com/tutorials/train-custom-datasets/).

## Additional Resources

- [ZED SDK Documentation - Custom Object Detection](https://www.stereolabs.com/docs/object-detection/custom-od/)
- [Community Support](https://community.stereolabs.com/)
