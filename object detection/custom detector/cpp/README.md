# ZED SDK - Custom Object Detection in C++ with Ultralytics models

This directory contains comprehensive examples demonstrating how to use custom models with the ZED SDK for real-time custom object detection and 3D capabilities including localization, 3D bounding boxes, and tracking.

## Directory Structure

```
cpp/
├── onnx_yolo_internal
├── opencv_dnn_yolov4
├── tensorrt_yolov5-v6-v8_onnx
├── tensorrt_yolov5-v6-v8_onnx_async
├── tensorrt_yolov8_seg_onnx
└── README.md
```

## Implementation Variants

### 1. **(RECOMMENDED) onnx_yolo_internal**
- **Models Supported**: YOLOv5, v8, v9, v10, v11, v12, v26 (ONNX format)
- **Features**: Uses ZED SDK's internal inference engine, preprocessing and postprocessing with a custom onnx yolo-like model

### 2. **opencv_dnn_yolov4**
- **Models Supported**: YOLOv4
- **Features**: OpenCV with DNN module implementation 

### 3. **tensorrt_yolov5-v6-v8_onnx**
- **Models Supported**: YOLOv8 (ONNX format)
- **Features**: Standard YOLOv8 implementation using TensorRT

### 4. **tensorrt_yolov5-v6-v8_onnx_async**
- **Models Supported**: YOLOv8 (ONNX format)
- **Features**: Asynchronous processing version of the TensorRT YOLOv8 implementation

### 5. **tensorrt_yolov8_seg_onnx**
- **Models Supported**: YOLOv8-seg (ONNX format)
- **Features**: YOLOv8 instance segmentation implementation using TensorRT

## Additional Resources

- [ZED SDK Download](https://www.stereolabs.com/developers/release/)
- [ZED SDK Documentation - Using the Object Detection API with a Custom Detector](https://www.stereolabs.com/docs/object-detection/custom-od/)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Ultralytics model Zoo](https://github.com/ultralytics/ultralytics#models)

---

**Need help?** Check the individual README files in each subdirectory for specific implementation details.
