# ZED SDK - Object Detection

This sample shows how to detect and track objects in space.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 
## Run the program
*NOTE: The ZED v1 is not compatible with this module*
To run the program, use the following command in your terminal : 
```bash
python object_detection_image_viewer.py
```
If you wish to run the program from an input_svo_file, or an IP adress, or specify a resolution run : 

```bash
python object_detection_image_viewer.py --input_svo_file <input_svo_file> --ip_address <ip_address> --resolution <resolution>
```
Arguments: 
  - --input_svo_file A path to an existing .svo file, that will be playbacked. If this parameter and ip_adress are not specified, the soft will use the camera wired as default.  
  - --ip_address IP Address, in format a.b.c.d:port or a.b.c.d. If specified, the soft will try to connect to the IP. 
  - --resolution Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA
### Features
 - The camera point cloud is displayed in a 3D OpenGL view
 - 3D bounding boxes around detected objects are drawn
 - Objects classes and confidences can be changed

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/