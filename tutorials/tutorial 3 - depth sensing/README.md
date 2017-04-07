# Tutorial 3: Depth sensing with the ZED


This tutorial shows how to get the depth from the ZED SDK. The program will loop until 50 frames are grabbed.
We assume that you have read previous tutorials (opening the ZED and image capture).


## Create a camera

As with previous tutorial, we create, configure and open the ZED.
We also want to work in HD720 at 60fps and enable the depth in PERFORMANCE mode. The ZED SDK provides different depth mode (PERFORMANCE, MEDIUM, QUALITY): for more information, refers to the documentation API or the online documentation.

```
// Create a ZED camera
Camera zed;

// Create configuration parameters
InitParameters init_params;
init_params.sdk_verbose = true; // Enable the verbose mode
init_params.depth_mode = DEPTH_MODE_PERFORMANCE; // Set the depth mode to performance (fastest)


// Open the camera
ERROR_CODE err = zed.open(init_params);
if (err!=SUCCESS)
  exit(-1);
```

<i>Note: Default parameter for depth mode is DEPTH_MODE_PERFORMANCE. Therefore, in practice, in the above example, the depth mode line is not really needed... But it is always good to set a parameter just to be sure. </i>

## Capture data

Now that the ZED is opened, we can now capture the images and the depth.
In a similar way than previous tutorial, we will loop until we have successfully captured 50 images.
Retrieving the depth map is as simple as getting the left image:
* We create a Mat to handle the depth map.
* We call Camera::retrieveMeasure() to get the depth map.

```
// Capture 50 images and depth, then stop
int i = 0;
sl::Mat image, depth;
while (i < 50) {
    // Grab an image
    if (zed.grab(runtime_parameters) == SUCCESS) {

        // A new image is available if grab() returns SUCCESS
        zed.retrieveImage(image, VIEW_LEFT); // Get the left image
        zed.retrieveMeasure(depth, MEASURE_DEPTH); // Retrieve depth Mat. Depth is aligned on the left image

        // Get and print depth value in mm at the center of the image
        int x = image.getWidth() / 2;
        int y = image.getHeight() / 2;
        float depth_value = 0.f;
        depth.getValue(x, y, &depth_value);
        printf("Depth at (%d, %d): %f mm\n", x, y, depth_value);
        i++;
    }
}
```


Now that we have retrieved the depth map, we may want to get the depth at a specific pixel. To do so, we use `Mat::getValue()` function.


In the example, we extract the depth at the center of the image (width/2, height/2)

```
// Get and print depth value in mm at the center of the image
int x = image.getWidth() / 2;
int y = image.getHeight() / 2;
float depth_value = 0.f;
depth.getValue(x, y, &depth_value);
printf("Depth at (%d, %d): %f mm\n", x, y, depth_value);
```

Once 50 frames have been grabbed, we close the camera.

```
// Close the camera
zed.close();
```


And this is it!<br/>
You are now using the ZED as a depth sensor.You can move on to the next tutorial to learn how to use the ZED as a positional tracker.


*You can find the complete source code of this sample in main.cpp located in the same folder*
