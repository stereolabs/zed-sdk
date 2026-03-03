# Custom Object Detection in Python with Ultralytics models

This directory contains comprehensive examples demonstrating how to use custom models with the ZED SDK for real-time custom object detection and 3D capabilities including localization, 3D bounding boxes, and tracking.

## Directory Structure

```
python/
├── onnx_yolo_internal
├── pytorch_yolo
├── pytorch_yolo_async
├── pytorch_yolo_cupy
├── pytorch_yolo_seg
└── README.md
```

## Implementation Variants

### 1. **(RECOMMENDED) onnx_yolo_internal**
- **Models Supported**: YOLOv5, v8, v9, v10, v11, v12, v26 (ONNX format)
- **Features**: Uses ZED SDK's internal inference engine, preprocessing and postprocessing with a custom onnx yolo-like model

### 2. **pytorch_yolo**
- **Models Supported**: YOLOv8, v9, v10, v11, v12, v26
- **Features**: Standard YOLO implementation

### 3. **pytorch_yolo_async**
- **Models Supported**: YOLOv8, v9, v10, v11, v12, v26
- **Features**: Asynchronous processing version of the standard YOLO implementation

### 4. **pytorch_yolo_cupy**
- **Models Supported**: YOLOv8, v9, v10, v11, v12, v26
- **Features**: GPU-accelerated (with CuPy) preprocessing version of the standard YOLO implementation

### 5. **pytorch_yolo_seg**
- **Models Supported**: yolov8-seg, yolov9-seg, yolo11-seg, yolo26-seg
- **Features**: Standard YOLO instance segmentation implementation

## Prerequisites

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

## Additional Resources

- [ZED SDK Download](https://www.stereolabs.com/developers/release/)
- [ZED SDK Documentation - Using the Object Detection API with a Custom Detector](https://www.stereolabs.com/docs/object-detection/custom-od/)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Ultralytics model Zoo](https://github.com/ultralytics/ultralytics#models)

---

**Need help?** Check the individual README files in each subdirectory for specific implementation details.
