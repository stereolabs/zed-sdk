# Live Global Localization Sample

## Overview

This sample demonstrates how to use the ZED SDK Global Localization module to achieve **global scale localization** on a real-world map using the ZED camera. The ZED SDK Live Global Localization sample fuses visual odometry from the ZED SDK with external GNSS data in real-time, making it a valuable resource for applications such as autonomous robotics and drone navigation.

## Features

- Displays the camera's path in an OpenGL window in 3D
- Displays path data, including translation and rotation
- Displays the fused path on a map in a web browser
- Exports KML files for the fused trajectory and raw GNSS data

## Dependencies

Before using this sample, ensure that you have the following dependencies installed on your system:

- ZED SDK: download and install from the official [Stereolabs website](https://www.stereolabs.com/developers/release/).
- `gpsd`: required to use an external GNSS sensor.
  > **Note**: Since [`gpsd`](https://gpsd.gitlab.io/gpsd/index.html) does not support Windows, this sample is not supported on Windows.

### C++

- `libgps-dev`: used to read data from `gpsd`.

### Python

- `gpsdclient`: used to read data from `gpsd`.

## Installation and Usage

To use the ZED SDK Global Localization sample, follow these steps:

1. Download and install the ZED SDK on your system from the official [Stereolabs website](https://www.stereolabs.com/developers/release/).
2. Install dependencies using your operating system's package manager.
3. Connect your ZED camera and GNSS sensor to your computer.
4. Open a terminal and navigate to the live sample directory.
5. Compile the sample for C++ in a *build* directory.
6. Run the `ZED_Live_Global_Localization` executable for C++ and `live.py` for Python.
7. The sample will display the camera's path and path data in a 3D window.
8. Go to the [map server sample](../map%20server) and run a simple server.
