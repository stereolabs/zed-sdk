# Live Geotracking Sample

## Overview

This sample demonstrates how to use the ZED SDK Geotracking module to achieve **global scale localization** on a real-world map using the ZED camera. The ZED SDK Live Geotracking sample fuses visual odometry from the ZED SDK with external GNSS data in real-time, making it a valuable resource for applications such as autonomous robotics and drone navigation.

## Features

- Displays the camera's path in an OpenGL window in 3D
- Displays path data, including translation and rotation
- Displays the fused path on a map on ZED Hub
- Exports KML files for the fused trajectory and raw GNSS data

## Dependencies

Before using this sample, ensure that you have the following dependencies installed on your system:
- ZED Hub Edge Agent: to be able to display the computed trajectory on a real-world map, connect your device to [ZED Hub](https://hub.stereolabs.com/). Detailed tutorials can be found [here](https://www.stereolabs.com/docs/cloud/overview/setup-device/).
- libgps-dev: required to use an external GNSS sensor.

## Installation and Usage

To use the ZED SDK Geotracking sample, follow these steps:
1. Download and install the ZED SDK on your system from the official [Stereolabs website](https://www.stereolabs.com/developers/release/).
2. Install Edge Agent from [ZED Hub](https://hub.stereolabs.com/) and the libgps-dev dependency using your operating system's package manager.
3. Connect your ZED camera and GNSS sensor to your computer.
4. Open a terminal and navigate to the zed-geotracking sample directory.
5. Compile the sample.
6. Run the zed-geotracking executable.
7. The sample will display the camera's path and path data in a 3D window. The fused path will be displayed on ZED Hub's maps page, and KML files will be generated for the fused trajectory and raw GNSS data.