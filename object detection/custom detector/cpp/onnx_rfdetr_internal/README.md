# RF-DETR custom ONNX ran in the ZED SDK

This sample demonstrates how to use a custom [RF-DETR](https://github.com/roboflow/rf-detr) model (a real-time, NMS-free detection transformer from Roboflow) in ONNX format with the ZED SDK's internal preprocessing, inference engine (using the highly optimized TensorRT framework) and postprocessing.

Unlike YOLO models, RF-DETR (and other DETR-like detectors) output **two tensors**, bounding boxes (`cxcywh`, normalized) and per-class logits, and require **ImageNet normalization** and **no NMS** (top-K selection over sigmoid scores). The ZED SDK detects and handles all of this internally, so the workflow is identical to the YOLO custom sample.

The ZED SDK optimizes your model using TensorRT and provides 3D object detection capabilities including localization, 3D bounding boxes, and tracking.

## Choosing the detection mode

Three modes drive the internal custom inference:

| `OBJECT_DETECTION_MODEL`             | Behaviour                                                             |
|--------------------------------------|----------------------------------------------------------------------|
| `CUSTOM_RFDETRLIKE_BOX_OBJECTS`      | Forces the RF-DETR / DETR-like decoder (used by this sample).         |
| `CUSTOM_YOLOLIKE_BOX_OBJECTS`        | Forces the YOLO decoder.                                             |
| `CUSTOM_BOX_OBJECTS_AUTODETECT`      | Auto-detects YOLO vs RF-DETR/DETR-like from the model outputs (recommended default for custom models). |

## Features

- 3D OpenGL point cloud visualization
- 3D bounding boxes around detected objects
- Configurable object classes and confidence thresholds
- Real-time tracking
- **ZED SDK internal handling of preprocessing, inference and postprocessing**

## Getting Started

 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
 - Check the [Documentation](https://docs.stereolabs.com/docs/)
 - [RF-DETR repository](https://github.com/roboflow/rf-detr)

## Workflow

### Export an RF-DETR model to ONNX

```sh
pip install rfdetr
```

```python
from rfdetr import RFDETRNano   # or RFDETRSmall / RFDETRMedium / RFDETRBase
model = RFDETRNano()            # loads the pretrained COCO checkpoint
model.export(output_dir="rfdetr_out")   # writes rfdetr_out/inference_model.onnx (or rfdetr-nano.onnx)
```

Notes:
- RF-DETR is exported with a **fixed square input** (RFDETRNano = 384, RFDETRSmall = 512, RFDETRMedium = 576, RFDETRBase = 560, ...). Set `detection_parameters.custom_onnx_dynamic_input_shape` to match your export.
- RF-DETR uses the **COCO 91-class** indexing: index 0 is background/no-object, 1 = person, 2 = bicycle, 3 = car, ... Configure per-class thresholds and `native_mapped_class` accordingly (this differs from YOLO's 80-class scheme).

### Build the sample

 - Build for [Windows](https://docs.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://docs.stereolabs.com/docs/app-development/cpp/linux/)

### Running the sample

```sh
./rfdetr_onnx_zed [.onnx] [zed camera id / optional svo filepath]

# For example rfdetr-nano
./rfdetr_onnx_zed rfdetr-nano.onnx 0      # 0 for zed camera id 0

# With an SVO file
./rfdetr_onnx_zed rfdetr-nano.onnx ./foo.svo
```

The first run optimizes the model with TensorRT for your GPU (this happens once and is cached).

## Additional Resources

- [ZED SDK Documentation - Custom Object Detection](https://docs.stereolabs.com/docs/object-detection/custom-od/)
- [Community Support](https://community.stereolabs.com/)
