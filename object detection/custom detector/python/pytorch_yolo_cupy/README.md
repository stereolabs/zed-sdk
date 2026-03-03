# PyTorch YOLO Detector with CupY

This sample demonstrates object detection using the official PyTorch implementation of YOLO (YOLOv8, v9, v10, v11, v12, v26) with a ZED camera. The preprocessing is done on the GPU using CuPy and PyTorch (and with zero-copy) to improve performance compared to classical NumPy handling.

## Features

- 3D OpenGL point cloud visualization
- 3D bounding boxes around detected objects
- Configurable object classes and confidence thresholds
- Real-time tracking
- **Pre-processing using CuPy and PyTorch for improved performance**

## Prerequisites

- [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/development/python/install)
- *Note: ZED v1 is not compatible with this module*

## Setup

```bash
# Install ZED SDK and Python API
# Download from: https://www.stereolabs.com/developers/release/

# Install YOLO dependencies
pip install ultralytics

# GPU acceleration of data processing and visualization
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
pip install cuda-python
```

## Usage

```bash
python detector.py --weights yolov8m.pt [--img_size 512 --conf_thres 0.1 --svo path/to/file.svo]
```

## Training Custom Models

This sample supports any model trained with YOLO (YOLOv8, v9, v10, v11, v12, v26), including custom trained models. For training on custom datasets, see the [Ultralytics Training Guide](https://docs.ultralytics.com/tutorials/train-custom-datasets/).

## Additional Resources

- [ZED SDK Documentation - Custom Object Detection](https://www.stereolabs.com/docs/object-detection/custom-od/)
- [CuPy Development Documentation](https://cupy.dev/)
- [Community Support](https://community.stereolabs.com/)