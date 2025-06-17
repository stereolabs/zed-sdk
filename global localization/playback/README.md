# Global Localization Data Playback

## Overview

The ZED SDK Global Localization Playback sample demonstrates how to fuse pre-recorded GNSS data (saved in a JSON file) and pre-recorded camera data (saved into an SVO file) for achieving global scale localization on a real-world map. This sample is useful for applications such as offline analysis of sensor data or simulation / testing.

## Features

- Displays the camera's path in an OpenGL window.
- Displays path data, including translation and rotation.
- Displays the fused path on a map in a web browser.
- Exports KML files for the fused trajectory and raw GNSS data.

## Dependencies

Before using this sample, ensure that you have the following dependencies installed on your system:

- ZED SDK: download and install from the official [Stereolabs website](https://www.stereolabs.com/developers/release/).

## Installation and Usage

To use the ZED SDK Global Localization Playback sample, follow these steps:

1. Download and install the ZED SDK on your system from the official [Stereolabs website](https://www.stereolabs.com/developers/release/).
2. Open a terminal and navigate to the playback sample directory.
3. Compile the sample for C++ in a _build_ directory.
4. Run the `ZED_Global_Localization_Playback` executable for C++ and `live.py` for Python, passing the path to the SVO file as the first input argument of the command line and the path to GNSS file as second argument.
5. The sample will display the camera's path and path data in a 3D window.
6. Go to the [map server sample](../map%20server) and run a simple server.
7. The sample will playback the SVO file and display the camera's path and path data in a 3D window. The fused path will be displayed on a map on web browser, and KML files will be generated for the fused trajectory and raw GNSS data.
