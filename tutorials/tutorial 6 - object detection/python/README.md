# Tutorial 6: 3D Object Detection with ZED 2

This tutorial shows how to use the object detection module with the ZED 2.<br/>
We assume that you have followed previous tutorials.

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [ZED SDK Python API](https://www.stereolabs.com/docs/app-development/python/install/)
- ZED 2 Camera

# Code overview
## Create a camera

As in previous tutorials, we create, configure and open the ZED 2. Please note that the ZED 1 is **not** compatible with the object detection module.

This module uses the GPU to perform deep neural networks computations. On platforms with limited amount of memory such as jetson Nano, it's advise to disable the GUI to improve the performances and avoid memory overflow.

``` python
# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_units = sl.UNIT.METER
init_params.sdk_verbose = True

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)
```

## Enable Object detection

We will define the object detection parameters. Notice that the object tracking needs the positional tracking to be able to track the objects in the world reference frame.

```python
# Define the Objects detection module parameters
obj_param = sl.ObjectDetectionParameters()
obj_param.enable_tracking=True
obj_param.image_sync=True
obj_param.enable_mask_output=True

# Object tracking requires the positional tracking module
camera_infos = zed.get_camera_information()
if obj_param.enable_tracking :
    zed.enable_positional_tracking()
```

Then we can start the module, it will load the model. This operation can take a few seconds. The first time the module is used, the model will be optimized for the hardware and will take more time. This operation is done only once.

```python
err = zed.enable_object_detection(obj_param)
if err != sl.ERROR_CODE.SUCCESS :
    print (repr(err))
    zed.close()
    exit(1)
```

The object detection is now activated.

## Capture data

The object confidence threshold can be adjusted at runtime to select only the revelant objects depending on the scene complexity. Since the parameters have been set to `image_sync`, for each `grab` call, the image will be fed into the AI module and will output the detections for each frames.

```python
# Detection Output
objects = sl.Objects()
# Detection runtime parameters
obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
while zed.grab() == sl.ERROR_CODE.SUCCESS:
    zed_error = zed.retrieve_objects(objects, obj_runtime_param);
    if objects.is_new :
        print(str(len(objects.object_list))+" Object(s) detected ("+str(zed.get_current_fps())+" FPS)")
```

## Disable modules and exit

Once the program is over the modules can be disabled and the camera closed. This step is optional since the `zed.close()` will take care of disabling all the modules. This function is also called automatically by the destructor if necessary.<br/>

```python
# Disable object detection and close the camera
zed.disable_object_detection()
zed.close()
return 0
```

And this is it!<br/>

You can now detect object in 3D with the ZED 2.
