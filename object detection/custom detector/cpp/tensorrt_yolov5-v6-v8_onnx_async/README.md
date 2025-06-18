# Yolov8 / v6 / v5 with ZED Custom Box input

This sample is designed to run a state of the art object detection model using the highly optimized TensorRT framework. The image are taken from the ZED SDK, and the 2D box detections are then ingested into the ZED SDK to extract 3D informations (localization, 3D bounding boxes) and tracking.

This sample is using a TensorRT optimized ONNX model. It is compatible with YOLOv8, YOLOv5 and YOLOv6. It can be used with the default model trained on COCO dataset (80 classes) provided by the framework maintainers.

A custom detector can be trained with the same architecture. These tutorials walk you through the workflow of training a custom detector :

- Yolov6 https://github.com/meituan/YOLOv6/blob/main/docs/Train_custom_data.md
- Yolov5 https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

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
yolo export model=yolov8n.pt format=onnx simplify=True dynamic=False imgsz=608
```

For a custom model model the weight file can be changed:

```
yolo export model=yolov8l_custom_model.pt format=onnx simplify=True dynamic=False imgsz=512
```

Please refer to the corresponding documentation for more details https://github.com/ultralytics/ultralytics


### ONNX to TensorRT Engine

TensorRT apply heavy optimisation by processing the network structure itself and benchmarking all the available implementation of each inference function to take the fastest. The result in the inference engine. This process can take a few minutes so we usually want to generate it the first time than saving it for later reload. This step should be done at each model or weight change, but only once.


```sh
./yolo_onnx_zed -s yolov8s.onnx yolov8s.engine
```

When using dynamic dimension (exported with `dynamic=True`), it should be specified as an argument. Please note that this sample requires a fixed size and doesn't handle range currently, the image should also be squared.


```sh
./yolo_onnx_zed -s yolov8s.onnx yolov8s.engine images:1x3x608x608
```

### Build the sample

 - Build for [Windows](https://www.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://www.stereolabs.com/docs/app-development/cpp/linux/)

### Running the sample with the engine generated

```sh
./yolo_onnx_zed [.engine] [zed camera id / optional svo filepath]  // deserialize and run inference

# For example yolov8n
./yolo_onnx_zed yolov8n.engine 0      # 0  for zed camera id 0

# With an SVO file
./yolo_onnx_zed yolov8n.engine ./foo.svo
```

## YOLOv6

The sample was mainly tested with YOLOv6 v3.0 but should work with other version with minor to no modifications.


### Installing yolov6

YOLOv6 can be installed directly from pip using the following command:

```sh
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt
pip install onnx>=1.10.0
```

### ONNX file export


```sh
wget https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6s.pt
python ./deploy/ONNX/export_onnx.py \
    --weights yolov6s.pt \
    --img 640 \
    --batch 1 \
    --simplify
```

For a custom model model the weight file can be changed:

```sh
python ./deploy/ONNX/export_onnx.py \
    --weights yolov6l_custom_model.pt \
    --img 640 \
    --batch 1 \
    --simplify
```

Please refer to the corresponding documentation for more details https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX

### ONNX to TensorRT Engine

TensorRT apply heavy optimisation by processing the network structure itself and benchmarking all the available implementation of each inference function to take the fastest. The result in the inference engine. This process can take a few minutes so we usually want to generate it the first time than saving it for later reload. This step should be done at each model or weight change, but only once.


```sh
./yolo_onnx_zed -s yolov6s.onnx yolov6s.engine
```

### Build the sample

 - Build for [Windows](https://www.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://www.stereolabs.com/docs/app-development/cpp/linux/)

### Running the sample with the engine generated

```sh
./yolo_onnx_zed [.engine] [zed camera id / optional svo filepath]  // deserialize and run inference

# For example yolov6s
./yolo_onnx_zed yolov6s.engine 0      # 0  for zed camera id 0

# With an SVO file
./yolo_onnx_zed yolov6s.engine ./foo.svo
```


## YOLOv5


### Installing yolov5

YOLOv5 can be installed directly from pip using the following command:

```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

### ONNX file export


```
python export.py --weights yolov5s.pt --include onnx --imgsz 640
```

For a custom model model the weight file can be changed:

```
python export.py --weights yolov8l_custom_model.pt --include onnx
```

Please refer to the corresponding documentation for more details https://docs.ultralytics.com/yolov5/tutorials/model_export/


### ONNX to TensorRT Engine

TensorRT apply heavy optimisation by processing the network structure itself and benchmarking all the available implementation of each inference function to take the fastest. The result in the inference engine. This process can take a few minutes so we usually want to generate it the first time than saving it for later reload. This step should be done at each model or weight change, but only once.

```
./yolo_onnx_zed -s yolov5s.onnx yolov5s.engine
```

When using dynamic dimension (exported with `--dynamic`), it should be specified as an argument. Please note that this sample requires a fixed size and doesn't handle range currently, the image should also be squared.


```
./yolo_onnx_zed -s yolov5s.onnx yolov5s.engine images:1x3x608x608
```

### Build the sample

 - Build for [Windows](https://www.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://www.stereolabs.com/docs/app-development/cpp/linux/)

### Running the sample with the engine generated

```sh
./yolo_onnx_zed [.engine] [zed camera id / optional svo filepath]  // deserialize and run inference

# For example yolov5s
./yolo_onnx_zed yolov5s.engine 0      # 0  for zed camera id 0

# With an SVO file
./yolo_onnx_zed yolov5.engine ./foo.svo
```