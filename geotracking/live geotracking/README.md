# ZED SDK - Live Geotracking for Global Scale Localization on Real-World Map

## Overview

This sample demonstrates how to use geotracking to achieve global scale localization on a real-world map using the ZED camera. The ZED SDK Geotracking sample fuses visual odometry from the ZED SDK with external GNSS data in real-time, making it a valuable resource for applications such as autonomous robotics and drone navigation.

### Features

- Displays the camera's path in an OpenGL window in 3D
- Displays path data, including translation and rotation
- Displays the fused path on a map on ZedHub
- Exports KML files for the fused trajectory and raw GNSS data

### Dependencies

Before using this sample, ensure that you have the following dependencies installed on your system:
- ZEDHub edge-cli: required for displaying the computed trajectory on a real-world map.
- libgps-dev: required to use an external GNSS sensor.

### Installation and Usage

To use the ZED SDK Geotracking sample, follow these steps:
1. Download and install the ZED SDK on your system from the official Stereolabs website.
2. Install the ZEDHub edge-cli from the ZEDHub website and the libgps-dev dependency using your operating system's package manager.
3. Connect your ZED camera and GNSS sensor to your computer.
4. Open a terminal and navigate to the zed-geotracking sample directory.
5. Compile the sample.
6. Run the zed-geotracking executable.
7. The sample will display the camera's path and path data in a 3D window. The fused path will be displayed on a map on ZedHub, and KML files will be generated for the fused trajectory and raw GNSS data.

### Support and Resources

If you have any questions or encounter any issues while using the ZED SDK, please visit the official Stereolabs support forums. Here, you can find helpful resources such as tutorials, documentation, and a community of developers to assist you in troubleshooting any problems you may encounter.
