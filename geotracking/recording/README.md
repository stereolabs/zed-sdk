# Geotracking Data Recording Sample

## Overview 
The Geotracking Data Recording sample demonstrates how to record data for geotracking localization on real-world maps using the ZED camera. The sample generates data in the form of an SVO file, which contains camera data, and a JSON file, which contains pre-recorded GNSS data for use in the playback sample. This sample is a useful resource for developers working on autonomous driving, robotics, and drone navigation applications.

## Features

- Displays the camera's path in an OpenGL window in 3D.
- Displays path data, including translation and rotation.
- Generates KML files for displaying raw GNSS data and fused position on google maps after capture.
- Generates an SVO file corresponding to camera data.
- Generates a JSON file corresponding to recorded GNSS data.

## Dependencies

Before using this sample, ensure that you have the following dependencies installed on your system:
- libgps: required to use an external GNSS sensor.

## Usage

To use the Geotracking Data Recording sample, follow these steps:

1. Download and install the ZED SDK on your system from the official Stereolabs website (https://www.stereolabs.com/developers/release/).
2. Install the libgps dependency using your operating system's package manager.
3. Connect your ZED camera and GNSS sensor to your computer.
4. Open a terminal and navigate to the Geotracking Data Recording sample directory.
5. Compile the sample.
6. Run the Geotracking Data Recording executable.
7. The sample will display the camera's path and path data in a 3D window. KML files will be generated for displaying the raw GNSS data and fused position on a real-world map like google maps after capture. Additionally, an SVO file corresponding to camera data and a JSON file corresponding to recorded GNSS data will be generated.