# ZED SDK - Object Detection

This sample shows how to detect and track objects in space.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 
## Run the program
*NOTE: The ZED v1 is not compatible with this module*
To run the program, use the following command in your terminal : 
```bash
python object_detection_birds_view.py
```
If you wish to run the program from an input_svo_file, or an IP adress, or specify a resolution run : 

```bash
python object_detection_birds_view.py --input_svo_file <input_svo_file> --ip_address <ip_address> --resolution <resolution> --disable_gui --enable_batching_reid    
```
Arguments: 
  - --input_svo_file A path to an existing .svo file, that will be playbacked. If this parameter and ip_adress are not specified, the soft will use the camera wired as default.  
  - --ip_address IP Address, in format a.b.c.d:port or a.b.c.d. If specified, the soft will try to connect to the IP.
  - --resolution Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA.
  - --disable_gui Flag to disable the GUI to increase detection performances. On low-end hardware such as Jetson Nano, the GUI significantly slows down the detection and increase the memory consumption.
  - --enable_batching_reid Flag to enable the batching re identification. Difference on ID tracking will be displayed in the console.

### Features
 - The camera point cloud is displayed in a 3D OpenGL view
 - 3D bounding boxes around detected objects are drawn
 - Objects classes can be changed

### Notes
This sample supports the batching system introduced in 3.5, but note that it is not supported on all platforms. 
See ZED SDK 3.5 Release notes for the supported platforms. 

This sample displays 2D detections and track objects in real time. However, with the use of batching, it is able to re attribute ID of objects that disappeared. The detections witch batching are stored in a queue and are shown in the log 
with a delay (see the latency parameter) : 
```bash
batch_parameters.latency = 3.0
```
If you want to introduce re-identification to your needs, you can also display the output of batching detection, either by displaying real time image and objects detections with a delay, or by keeping in memory past images to have a display with image retention, meaning image and detections will be synched but with a delay to the real time camera output. If you want to do so, creating a BatchHandler class to handle the differents queues is a good start. 


## Support
If you need assistance go to our Community site at https://community.stereolabs.com/
