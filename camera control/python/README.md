# Stereolabs ZED - Camera Control

This sample shows how to capture images with the ZED SDK and adjust camera settings.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

### Prerequisites

- Windows 10, Ubuntu LTS
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [ZED Python API](https://www.stereolabs.com/docs/app-development/python/)
- OpenCV python (`python -m pip install opencv-python`)

## Features

This sample demonstrates how to captures images and change camera settings with the ZED SDK.
The following parameters can be changed:

  - Exposure
  - Gain
  - Saturation
  - Hue
  - Contrast
  - Brightness

## Run the example
    
```
python "camera_control.py"
```