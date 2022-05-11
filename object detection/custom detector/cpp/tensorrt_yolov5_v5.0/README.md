# Yolov5 with ZED Custom Box input

This sample is designed to run a state of the art object detection model using the highly optimized TensorRT framework. The image are taken from the ZED SDK, and the 2D box detections are then ingested into the ZED SDK to extract 3D informations (localization, 3D bounding boxes) and tracking.

This sample is a fork from [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5). The Pytorch implementation is [ultralytics/yolov5](https://github.com/ultralytics/yolov5). The default model was trained on COCO dataset (80 classes), a custom detector can be trained with the same architecture, it requires a configuration tweak (see below). This tutorial walk you through the workflow of training a custom detector https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data.


## Getting Started

 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 - [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)

## Different versions of yolov5

- For yolov5 v5.0, download .pt from [yolov5 release v5.0](https://github.com/ultralytics/yolov5/releases/tag/v5.0), `git clone -b v5.0 https://github.com/ultralytics/yolov5.git` 

## Config

- Choose the model s/m/l/x/s6/m6/l6/x6 from command line arguments.
- Input shape defined in yololayer.h
- Number of classes defined in yololayer.h, **DO NOT FORGET TO ADAPT THIS, If using your own model**
- FP16/FP32 can be selected by the macro in yolov5.cpp, FP16 is faster if the GPU support it (all jetsons or GeForce RTX cards), 
- GPU id can be selected by the macro in yolov5.cpp
- NMS thresh in yolov5.cpp
- BBox confidence thresh in yolov5.cpp

## Using the sample


### 1. (Optional for first run) Generate .wts from pytorch with .pt

**This file has already been generated and can be downloaded [here](https://download.stereolabs.com/sample_custom_objects/yolov5s.wts.zip)** (and needs to be unzipped) to run the sample. 

This procedure can be applied to other models (such as `l` or `m` variants) or custom dataset trained model.

The goal is to export the PyTorch model `.pt` into a easily readable weight file `.wts`.

```sh
git clone -b v5.0 https://github.com/ultralytics/yolov5.git
# Download the pretrained weight file
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
cp gen_wts.py {ultralytics}/yolov5
cd {ultralytics}/yolov5
python gen_wts.py yolov5s.pt
# a file 'yolov5s.wts' will be generated.
```


### 2. Build the sample

If a custom model is used, let's say trained on another dataset than COCO, with a different number of classes, `CLASS_NUM` in yololayer.h must be updated accordingly.

 - Build for [Windows](https://www.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://www.stereolabs.com/docs/app-development/cpp/linux/)


### 3. Generate the TensorRT engine

TensorRT apply heavy optimisation by processing the network structure itself and benchmarking all the available implementation of each inference function to take the fastest. The result in the inference engine. This process can take a few minutes so we usually want to generate it the first time than saving it for later reload. This step should be done at each model or weight change, but only once.

```sh
./yolov5 -s [.wts] [.engine] [s/m/l/x/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file
# For example yolov5s
./yolov5 -s yolov5s.wts yolov5s.engine s
# For example Custom model with depth_multiple=0.17, width_multiple=0.25 in yolov5.yaml
./yolov5 -s yolov5_custom.wts yolov5.engine c 0.17 0.25
```

### 4. Running the sample with the engine generated

```sh
./yolov5 -d [.engine] [optional svo filepath]  // deserialize and run inference
# For example yolov5s
./yolov5 -d yolov5s.engine
# With an SVO file
./yolov5 -d yolov5.engine ./foo.svo
```
