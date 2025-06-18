# ZED SDK - Object Detection

This sample shows how to detect and track objects in space.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)

## Build the program
 - Build for [Windows](https://www.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://www.stereolabs.com/docs/app-development/cpp/linux/)

## Run the program
*NOTE: The ZED v1 is not compatible with this module*
- Navigate to the build directory and launch the executable
- Or open a terminal in the build directory and run the sample :

      ./ZED_Object_detection_birds_eye_viewer

### Features
 - The camera point cloud is displayed in a 3D OpenGL view
 - 3D bounding boxes around detected objects are drawn
 - Objects classes and confidences can be changed

### Notes
This sample supports the batching system introduced in 3.5, but note that it is not supported on all platforms. 
See ZED SDK 3.5 Release notes for the supported platforms. 

This sample displays 2D detections and track objects in real time. However, with the use of batching, it is able to re attribute ID of objects that disappeared. The detections witch batching are stored in a queue and are shown in the log 
with a delay (see the latency parameter) : 
```bash
detection_parameters.batch_parameters.latency = 3.f;
```
If you want to introduce re-identification to your needs, you can also display the output of batching detection, either by displaying real time image and objects detections with a delay, or by keeping in memory past images to have a display with image retention, meaning image and detections will be synched but with a delay to the real time camera output. If you want to do so, creating a BatchHandler class to handle the differents queues is a good start. 


## Support
If you need assistance go to our Community site at https://community.stereolabs.com/
