# Global Localization Data Recording Sample

## Overview

The Global Localization Data Recording sample demonstrates how to record data for global localization on real-world maps using the ZED camera. The sample generates data in the form of an SVO file, which contains camera data, and a JSON file, which contains pre-recorded GNSS data for use in the playback sample. This sample is a useful resource for developers working on autonomous driving, robotics, and drone navigation applications.

## Features

- Displays the camera's path in an OpenGL window in 3D.
- Displays path data, including translation and rotation.
- Generates KML files for displaying raw GNSS data and fused position on google maps after capture.
- Generates an SVO file corresponding to camera data.
- Generates a JSON file corresponding to recorded GNSS data.

## Dependencies

Before using this sample, ensure that you have the following dependencies installed on your system:

- ZED SDK: download and install from the official [Stereolabs website](https://www.stereolabs.com/developers/release/).
- `gpsd`: required to use an external GNSS sensor.
  > **Note**: Since [`gpsd`](https://gpsd.gitlab.io/gpsd/index.html) does not support Windows, this sample is not supported on Windows.

### C++

- `libgps-dev`: used to read data from `gpsd`.

### Python

- `gpsdclient`: used to read data from `gpsd`.

## Usage

To use the Global Localization Data Recording sample, follow these steps:

1. Download and install the ZED SDK on your system from the official [Stereolabs website](https://www.stereolabs.com/developers/release/).
2. Install dependencies using your operating system's package manager.
3. Connect your ZED camera and GNSS sensor to your computer.
4. Open a terminal and navigate to the Global Localization Data Recording sample directory.
5. Compile the sample for C++ in a *build* directory.
6. Run the `ZED_Global_Localization_Recording` executable for C++ and `live.py` for Python.
7. The sample will display the camera's path and path data in a 3D window. KML files will be generated for displaying the raw GNSS data and fused position on a real-world map like google maps after capture. Additionally, an SVO file corresponding to camera data and a JSON file corresponding to recorded GNSS data will be generated.
8. Go to the [map server sample](./map%20server) and run a simple server.
