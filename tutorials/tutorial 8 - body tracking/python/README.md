# Tutorial 8: Body Tracking with ZED 2

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
init_params.sdk_verbose = 1

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)
```

## Enable Body Tracking

We will define the object detection parameters. Notice that the object tracking needs the positional tracking to be able to track the objects in the world reference frame.

```python
# Define the Objects detection module parameters
body_params = sl.BodyTrackingParameters()
# Different model can be chosen, optimizing the runtime or the accuracy
body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
body_params.enable_tracking = True
body_params.enable_segmentation = False
# Optimize the person joints position, requires more computations
body_params.enable_body_fitting = True
# Object tracking requires the positional tracking module

if body_params.enable_tracking:
    positional_tracking_param = sl.PositionalTrackingParameters()
    # positional_tracking_param.set_as_static = True
    positional_tracking_param.set_floor_as_origin = True
    zed.enable_positional_tracking(positional_tracking_param)
```

Then we can start the module, it will load the model. This operation can take a few seconds. The first time the module is used, the model will be optimized for the hardware and will take more time. This operation is done only once.

```python
err = zed.enable_body_tracking(body_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Enable Body Tracking : "+repr(err)+". Exit program.")
    zed.close()
    exit()
```

The object detection is now activated.

## Capture data

The object confidence threshold can be adjusted at runtime to select only the revelant skeletons depending on the scene complexity. For each `grab` call, the image will be fed into the AI module and will output the detections for each frames.

```python
bodies = sl.Bodies()
body_runtime_param = sl.BodyTrackingRuntimeParameters()
# For outdoor scene or long range, the confidence should be lowered to avoid missing detections (~20-30)
# For indoor scene or closer range, a higher confidence limits the risk of false positives and increase the precision (~50+)
body_runtime_param.detection_confidence_threshold = 40

i = 0 
    while i < 100:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            err = zed.retrieve_bodies(bodies, body_runtime_param)
            if bodies.is_new:
                body_array = bodies.body_list
                print(str(len(body_array)) + " Person(s) detected\n")
```

## Disable modules and exit

Once the program is over the modules can be disabled and the camera closed. This step is optional since the `zed.close()` will take care of disabling all the modules. This function is also called automatically by the destructor if necessary.<br/>

```python
# Close the camera
zed.disable_body_tracking()
zed.close()
```

And this is it!<br/>

You can now detect object in 3D with the ZED 2.
