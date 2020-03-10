# Stereolabs ZED - Camera Control

This sample shows how to capture images with the ZED SDK and adjust camera settings.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

## Features

This sample demonstrates how to captures images and change camera settings with the ZED SDK.
The following parameters can be changed:

  - Exposure
  - Gain
  - Saturation
  - Hue
  - Contrast
  - Brightness

## Run the examples

Some Python samples require OpenCV and OpenGL, you can install them via pip with **opencv-python** and **PyOpenGL** packages.

### Live camera

Live camera sample showing the camera information and video in real time and allows to control the different settings.
    
```
python "camera_control.py"
```