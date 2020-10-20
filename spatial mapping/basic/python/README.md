# Stereolabs ZED - Camera Control

This sample shows how to capture a real-time 3D map of the scene with the ZED API.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

## Run the examples

Some Python samples require OpenCV and OpenGL, you can install them via pip with **opencv-python** and **PyOpenGL** packages.

### Spatial Mapping

Spatial Mapping sample shows mesh information after filtering and applying texture on frames. The mesh and its filter parameters can be saved.

```
python "mesh_example.py" svo_file.svo
```
