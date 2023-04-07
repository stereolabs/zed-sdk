# Tutorial 3: Depth sensing with the ZED

This tutorial shows how to get the depth from the ZED SDK. The program will loop until 50 frames are grabbed.
We assume that you have followed previous tutorials (opening the ZED and image capture).

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [ZED SDK Python API](https://www.stereolabs.com/docs/app-development/python/install/)

# Code overview
## Create a camera

As in other tutorials, we create, configure and open the ZED.
We set the ZED in HD720 mode at 60fps and enable depth in PERFORMANCE mode. The ZED SDK provides different depth modes: PERFORMANCE, MEDIUM, QUALITY. For more information, see online documentation.

```python
# Create a ZED camera
zed = sl.Camera()

# Create configuration parameters
init_params = sl.InitParameters()
init_params.sdk_verbose = True # Enable the verbose mode
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Set the depth mode to performance (fastest)


# Open the camera
err = zed.open(init_params)
if (err!=sl.ERROR_CODE.SUCCESS):
  exit(-1)
```

<i>Note: Default parameter for depth mode is DEPTH_MODE.PERFORMANCE. In practice, it is not necessary to set the depth mode in InitParameters. </i>

## Capture data

Now that the ZED is opened, we can capture images and depth. Here we loop until we have successfully captured 50 images.
Retrieving the depth map is as simple as retrieving an image:
* We create a Mat to store the depth map.
* We call retrieve_measure() to get the depth map.

```
# Capture 50 images and depth, then stop
i = 0
image = sl.Mat()
depth = sl.Mat()
while (i < 50) :
    # Grab an image
    if (zed.grab() == sl.ERROR_CODE.SUCCESS) :
        # A new image is available if grab() returns SUCCESS
        zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH) # Retrieve depth Mat. Depth is aligned on the left image
        i = i + 1
    }
}
```


Now that we have retrieved the depth map, we may want to get the depth at a specific pixel. 
In the example, we extract the distance of the point at the center of the image (width/2, height/2)

```python
# Get and print distance value in mm at the center of the image
# We measure the distance camera - object using Euclidean distance
x = image.get_width() / 2
y = image.get_height() / 2
point_cloud_value = point_cloud.get_value(x, y)

distance = math.sqrt(point_cloud_value[0]*point_cloud_value[0] + point_cloud_value[1]*point_cloud_value[1] + point_cloud_value[2]*point_cloud_value[2])
printf("Distance to Camera at (", x, y, "): ", distance, "mm")
```

Once 50 frames have been grabbed, we close the camera.

```
# Close the camera
zed.close()
```

You are now using the ZED as a depth sensor.You can move on to the next tutorial to learn how to use the ZED as a positional tracker.


