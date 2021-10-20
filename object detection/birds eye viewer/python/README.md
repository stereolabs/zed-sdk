# ZED SDK - Object Detection

This sample shows how to detect and track objects in space.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 
## Run the program
*NOTE: The ZED v1 is not compatible with this module*

      python "object_detection_birds_view.py"

### Features
 - The camera point cloud is displayed in a 3D OpenGL view
 - 3D bounding boxes around detected objects are drawn
 - Objects classes and confidences can be changed

### Notes
This sample supports the batching system introduced in 3.5, but note that it is not supported on all platforms. 
See ZED SDK 3.5 Release notes for the supported platforms. 

- To use batching system, set the `USE_BATCHING` flag to `True` in [object_detection_birds_view.py](object_detection_birds_view.py). 
The batching works in 2 different modes: 

--> without image retention : images and point cloud will be live ( no latency) but data will be output with the batching latency parameters. 
This means that images and data will not be synced and you will see that objects are moving with a latency compared to the image.

--> with image retention : images and point cloud will be stored and available when data is output with the batching latency parameters.
This means that images and data will be synced, but this requires more memory to be able to store and pop the images/data correctly. Therefore, use this mode with caution. 
Since the sync is based on objects output, note that images/point cloud can be drawn only if an object is detected in the scene. 

--> The image retention flag is defined in the file [batch_system_handler.py](batch_system_handler.py) by the variable `WITH_IMAGE_RETENTION`

- If you want to use the batching system in your application, a good starting point is to use the BatchSystemHandler class and modify it to your needs.

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/