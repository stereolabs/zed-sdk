# Geotracking Data Playback

## Overview

The ZED SDK Geotracking Playback sample demonstrates how to fuse pre-recorded GNSS data (saved in a JSON file) and pre-recorded camera data (saved into an SVO file) for achieving global scale localization on a real-world map. This sample is useful for applications such as offline analysis of sensor data or simulation / testing.

## Features

- Displays the camera's path in an OpenGL window.
- Displays path data, including translation and rotation.
- Displays the fused path on a map on ZedHub.
- Exports KML files for the fused trajectory and raw GNSS data.

## Dependencies

Before using this sample, ensure that you have the following dependencies installed on your system:
- ZED SDK: download and install from the official Stereolabs website (https://www.stereolabs.com/developers/release/).
- ZED Hub Edge Agent: to be able to display the computed trajectory on a real-world map, connect your device to [ZED Hub](https://hub.stereolabs.com/). Detailed tutorials can be found [here](https://www.stereolabs.com/docs/cloud/overview/setup-device/).

## Installation and Usage

To use the ZED SDK Geotracking Playback sample, follow these steps:
1. Download and install the ZED SDK on your system from the official Stereolabs website (https://www.stereolabs.com/developers/release/).
2. Install the ZEDub edge-cli from the ZEDHub website.
3. Open a terminal and navigate to the zed-geotracking-playback sample directory.
4. Compile it.
5. Run the zed-geotracking-playback executable, passing the path to the SVO file as the first input argument of the command line and the path to gnss file as second argument.
6. The sample will playback the SVO file and display the camera's path and path data in a 3D window. The fused path will be displayed on a map on ZedHub, and KML files will be generated for the fused trajectory and raw GNSS data.