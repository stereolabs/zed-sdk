# TensorRT YOLOv5 / v6 / v8 ONNX Detector

This sample demonstrates how to use custom YOLO models (YOLOv5, YOLOv6, YOLOv8) in ONNX format with TensorRT optimization for object detection with the ZED SDK. The detected 2D boxes are ingested into the ZED SDK to extract 3D information including localization, 3D bounding boxes, and tracking.

## Features

- 3D OpenGL point cloud visualization
- 3D bounding boxes around detected objects
- Configurable object classes and confidence thresholds
- Real-time tracking
- **TensorRT optimization for maximum performance**

## Prerequisites

- [ZED SDK](https://www.stereolabs.com/developers/release/)
- [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- CUDA Toolkit

## Workflow

This sample requires a TensorRT engine optimized from an ONNX model. The ONNX model can be exported from PyTorch using the original YOLO code.

### Model Export and Engine Generation

#### YOLOv8 (Recommended)

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
./yolo_onnx_zed -s yolov8s.onnx yolov8s.engine

# For dynamic dimensions (if exported with dynamic=True)
./yolo_onnx_zed -s yolov8s.onnx yolov8s.engine images:1x3x608x608
```

#### YOLOv6

**Installation:**
```bash
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt
pip install onnx>=1.10.0
```

**ONNX Export:**
```bash
# Download and export standard model
wget https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6s.pt
python ./deploy/ONNX/export_onnx.py --weights yolov6s.pt --img 640 --batch 1 --simplify

# Custom model
python ./deploy/ONNX/export_onnx.py --weights yolov6l_custom_model.pt --img 640 --batch 1 --simplify
```

**TensorRT Engine Generation:**
```bash
./yolo_onnx_zed -s yolov6s.onnx yolov6s.engine
```

#### YOLOv5

**Installation:**
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

**ONNX Export:**
```bash
# Standard model
python export.py --weights yolov5s.pt --include onnx --imgsz 640

# Custom model
python export.py --weights yolov8l_custom_model.pt --include onnx
```

**TensorRT Engine Generation:**
```bash
# Generate engine
./yolo_onnx_zed -s yolov5s.onnx yolov5s.engine

# For dynamic dimensions
./yolo_onnx_zed -s yolov5s.onnx yolov5s.engine images:1x3x608x608
```

## Build and Usage

### Building the Sample
- [Build for Windows](https://docs.stereolabs.com/docs/app-development/cpp/windows/)
- [Build for Linux/Jetson](https://docs.stereolabs.com/docs/app-development/cpp/linux/)

### Running the Sample
```bash
./yolo_onnx_zed [.engine] [zed camera id / optional svo filepath]

# Examples:
./yolo_onnx_zed yolov8n.engine 0           # ZED camera ID 0
./yolo_onnx_zed yolov8n.engine ./foo.svo   # SVO file
```

## Training Custom Models

For training custom detectors with the supported architectures:
- **YOLOv8**: [Ultralytics Training Guide](https://docs.ultralytics.com/modes/train)
- **YOLOv6**: [YOLOv6 Custom Training](https://github.com/meituan/YOLOv6/blob/main/docs/Train_custom_data.md)
- **YOLOv5**: [YOLOv5 Custom Training](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

## Additional Resources

- [ZED SDK Documentation - Custom Object Detection](https://docs.stereolabs.com/docs/object-detection/custom-od/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [Community Support](https://community.stereolabs.com/)