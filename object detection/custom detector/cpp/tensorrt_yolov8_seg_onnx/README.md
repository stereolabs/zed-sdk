# Yolov8 with ZED Custom Box & Masks input

This sample is designed to run a state-of-the-art instance segmentation model using the highly optimized TensorRT framework. The images are taken from the ZED SDK, and the 2D box detections and predicted masks are then ingested into the ZED SDK to extract 3D information (localization, 3D bounding boxes) and tracking.

This sample uses a TensorRT-optimized ONNX model. It is compatible with YOLOv8-seg. It can be used with the default model trained on the COCO dataset (80 classes) provided by the framework maintainers.

A custom detector can be trained with the same architecture. This tutorials walk you through the workflow of training a custom detector: https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolov8-instance-segmentation-on-custom-dataset.ipynb

## Getting Started

 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 - [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)

## Workflow

This sample is expecting a TensorRT engine, optimized from an ONNX model. The ONNX model can be exported from Pytorch using the original YOLO code. Please refer to the section corresponding to the needed version.

## YOLOv8 (recommended)

### Installing yolov8

YOLOv8 can be installed directly from pip using the following command:

```
python -m pip install ultralytics
```

### ONNX file export

In this documentation, we'll use the CLI for export https://docs.ultralytics.com/modes/export/

```
yolo export model=yolov8s-seg.pt format=onnx simplify=True dynamic=False imgsz=640
```

For a custom model model the weight file can be changed:

```
yolo export model=yolov8s-seg_custom_model.pt format=onnx simplify=True dynamic=False imgsz=640
```

Please refer to the corresponding documentation for more details https://github.com/ultralytics/ultralytics


### ONNX to TensorRT Engine

TensorRT applies heavy optimization by processing the network structure itself and benchmarking all the available implementations of each inference function to take the fastest. The result is the inference engine. This process can take a few minutes so we usually want to generate it the first time and then save it for later reload. This step should be done at each model or weight change only once.

Please note that this sample requires a fixed size and that image should also be squared (e.g. 640x640).

You can either set the input size implicitely, with

```sh
./yolov8_seg_onnx_zed -s yolov8s-seg.onnx yolov8s-seg.engine
```

Or explicitely with

```sh
./yolov8_seg_onnx_zed -s yolov8s-seg.onnx yolov8s-seg.engine images:1x3x640x640
```

### Build the sample

 - Build for [Windows](https://www.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://www.stereolabs.com/docs/app-development/cpp/linux/)

### Running the sample with the generated engine

```sh
./yolov8_seg_onnx_zed [.engine] [zed camera id / optional svo filepath]  // deserialize and run inference

# For example yolov8s-seg with a plugged zed camera
./yolov8_seg_onnx_zed yolov8s-seg.engine

# For example yolov8s-seg with an SVO file
./yolov8_seg_onnx_zed yolov8s-seg.engine ./foo.svo
```
