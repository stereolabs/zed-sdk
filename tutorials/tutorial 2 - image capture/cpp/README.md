# Tutorial 2: Image capture

This tutorial shows how to capture left images of the ZED camera. The program will loop until we have successfully grabbed 50 images.
We assume that you have read the tutorial 1 and successfully opened your ZED.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

## Build the program

Download the sample and follow the instructions below: [More](https://www.stereolabs.com/docs/getting-started/application-development/)

#### Build for Windows

- Create a "build" folder in the source folder
- Open cmake-gui and select the source and build folders
- Generate the Visual Studio `Win64` solution
- Open the resulting solution and change configuration to `Release`
- Build solution

#### Build for Linux

Open a terminal in the sample directory and execute the following command:

    mkdir build
    cd build
    cmake ..
    make
	
# Code overview
## Create a camera

As with previous tutorial, we create, configure and open the ZED. Here we show how to set a resolution and a framerate. We want to work in H1080 at 30 fps (default) in this example.


```
// Create a ZED camera object
Camera zed;

// Set configuration parameters
InitParameters init_params;
init_params.camera_resolution = RESOLUTION_HD1080; // Use HD1080 video mode
init_params.camera_fps = 30; // Set fps at 30

// Open the camera
ERROR_CODE err = zed.open(init_params);
if (err != ERROR_CODE::SUCCESS)
    exit(-1);
```


## Capture data

Now that the ZED is opened, we can now capture the images coming from it.
We create a loop that capture 50 images and exit.

To capture an image and process it, you need to call Camera::grab() function. This function take runtime parameters as well, but we leave them to default in this tutorial.
Each time you want a new image, you need to call this function. if grab() returns ERROR_CODE::SUCCESS, a new image has been capture and is now available. Otherwise, you can check the status of grab() which will tell you if there is no new frame available (depending on the framerate of the camera) or if something wrong happened.

```
// Grab an image
if (zed.grab() == ERROR_CODE::SUCCESS) {
	// A new image is available if grab() returns ERROR_CODE::SUCCESS
}
```

Once grab has been done, you can get all the data provided with the ZED SDK. In this tutorial, we want to retrieve the left image and its timestamp. To do so, we use `the Camera::retrieveImage()` and `Camera::getCameraTimestamp()` functions.

```
zed.retrieveImage(image,VIEW_LEFT); // Get the left image
unsigned long long timestamp = zed.getCameraTimestamp(); // Get the timestamp of the image
printf("Image resolution: %d x %d  || Image timestamp: %llu\n", image.getWidth(), image.getHeight(), timestamp);
```

retrieveImage() takes a sl::Mat as parameter, as well as a VIEW mode. We first need to create the Mat before starting the loop. Note that creating a Mat does not allocate its memory, therefore the first retrieveImage() will automatically allocate its memory for us.

Since we want to stop the loop once we capture 50 images, we just increment the counter when a grab is successful.

```
// Capture 50 frames and stop
int i = 0;
sl::Mat image;
while (i < 50) {
    // Grab an image
    if (zed.grab() == ERROR_CODE::SUCCESS) {
        // A new image is available if grab() returns ERROR_CODE::SUCCESS
        zed.retrieveImage(image, VIEW::LEFT); // Get the left image
        auto timestamp = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE); // Get the timestamp at the time the image was captured
        printf("Image resolution: %d x %d  || Image timestamp: %llu\n", image.getWidth(), image.getHeight(), timestamp);
        i++;
    }
}
```
<i>Note:</i> the image timestamp is given in nanoseconds. You can compare the timestamp between two grab() : it should be close to the framerate time, if you don't have frames dropped.

Now that we have captured 50 images, we can close the camera and exit the program.

```
// Close the camera
zed.close();
return 0;
```

And this is it!<br/>
Now you can move on to the next tutorial to learn how to get the depth from the ZED camera.


*You can find the complete source code of this sample in main.cpp located in the same folder*
