# Tutorial 2: Image capture

This tutorial shows how to capture left images of the ZED camera. The program will loop until we have successfully grabbed 50 images.
We assume that you have read the tutorial 1 and successfully opened your ZED.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [ZED SDK Python API](https://www.stereolabs.com/docs/app-development/python/install/)

# Code overview
## Create a camera

As with previous tutorial, we create, configure and open the ZED. Here we show how to set a resolution and a framerate. We want to work in H1080 at 30 fps (default) in this example.


```python
# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080 # Use HD1080 video mode
init_params.camera_fps = 30 # Set fps at 30

# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS) :
    exit(-1);
```


## Capture data

Now that the ZED is opened, we can now capture the images coming from it.
We create a loop that capture 50 images and exit.

To capture an image and process it, you need to call Camera.grab() function. This function take runtime parameters as well, but we leave them to default in this tutorial.
Each time you want a new image, you need to call this function. if grab() returns SUCCESS, a new image has been capture and is now available. Otherwise, you can check the status of grab() which will tell you if there is no new frame available (depending on the framerate of the camera) or if something wrong happened.

```python
# Grab an image
if (zed.grab() == sl.ERROR_CODE.SUCCESS) :
	# A new image is available if grab() returns SUCCESS
```

Once grab has been done, you can get all the data provided with the ZED SDK. In this tutorial, we want to retrieve the left image and its timestamp. To do so, we use `the Camera.retrieve_image()` and `Camera.get_timestamp()` functions.

```python
zed.retrieve_image(image,sl.VIEW.LEFT) # Get the left image
timestamp = zed.get_current_timestamp(sl.TIME_REFERENCE.IMAGE) # Get the timestamp of the image
print("Image resolution: ", image.get_width(), " x ", image.get_height()," || Image timestamp: ", timestamp.get_milliseconds())
```

retrieve_image() takes a sl.Mat as parameter, as well as a VIEW mode. We first need to create the Mat before starting the loop. Note that creating a Mat does not allocate its memory, therefore the first retrieve_image() will automatically allocate its memory for us.

Since we want to stop the loop once we capture 50 images, we just increment the counter when a grab is successful.

```python
# Capture 50 frames and stop
i = 0
image = sl.Mat()
while (i < 50) :
    # Grab an image
    if (zed.grab() == sl.ERROR_CODE.SUCCESS) :
        # A new image is available if grab() returns SUCCESS
        zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE) # Get the timestamp at the time the image was captured
	print("Image resolution: ", image.get_width(), " x ", image.get_height()," || Image timestamp: ", timestamp.get_milliseconds())
        i = i+1
```
<i>Note:</i> the image timestamp is given in nanoseconds. You can compare the timestamp between two grab() : it should be close to the framerate time, if you don't have frames dropped.

Now that we have captured 50 images, we can close the camera and exit the program.

```
# Close the camera
zed.close()
return 0
```

And this is it!<br/>
Now you can move on to the next tutorial to learn how to get the depth from the ZED camera.


*You can find the complete source code of this sample in main.cpp located in the same folder*

