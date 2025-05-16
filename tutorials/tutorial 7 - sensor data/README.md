# Tutorial 7: Getting sensor data from ZED Mini and ZED2

This tutorial shows how to use retrieve sensor data from ZED Mini and ZED2. It will loop until 800 data samples are grabbed, printing the updated values on console.<br/>
We assume that you have followed previous tutorials.

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

# Code overview

## Capture data

The sensor data can be retrieved in two ways: synchronized or not synchronized to the image frames.

## Process data

To be sure that data are updated, since they are not synchronized to camera frames, we must check that the timestamp as changed from the previous retrieved values. We use the timestamp of the IMU sensor as main value since it's the sensor that runs at higher frequency.

If we are using a ZED2 we have more sensor data to be acquired and elaborated:

- IMU Temperature
- Magnetic fields
- Barometer data