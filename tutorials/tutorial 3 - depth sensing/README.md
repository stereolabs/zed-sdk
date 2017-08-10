# Tutorial 3: Depth sensing with the ZED

This tutorial shows how to get the depth from the ZED SDK. The program will loop until 50 frames are grabbed.
We assume that you have followed previous tutorials (opening the ZED and image capture).

### Prerequisites

- Windows 7 64bits or later, Ubuntu 16.04
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

## Build the program

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

As in other tutorials, we create, configure and open the ZED.
We set the ZED in HD720 mode at 60fps and enable depth in PERFORMANCE mode. The ZED SDK provides different depth modes: PERFORMANCE, MEDIUM, QUALITY. For more information, see online documentation.

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

<i>Note: Default parameter for depth mode is DEPTH_MODE_PERFORMANCE. In practice, it is not necessary to set the depth mode in InitParameters. </i>

## Capture data

Now that the ZED is opened, we can capture images and depth. Here we loop until we have successfully captured 50 images.
Retrieving the depth map is as simple as retrieving an image:
* We create a Mat to store the depth map.
* We call retrieveMeasure() to get the depth map.

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
        i++;
    }
}
```


Now that we have retrieved the depth map, we may want to get the depth at a specific pixel. 
In the example, we extract the distance of the point at the center of the image (width/2, height/2)

```
// Get and print distance value in mm at the center of the image
// We measure the distance camera - object using Euclidean distance
int x = image.getWidth() / 2;
int y = image.getHeight() / 2;
sl::float4 point_cloud_value;
point_cloud.getValue(x, y, &point_cloud_value);

float distance = sqrt(point_cloud_value.x*point_cloud_value.x + point_cloud_value.y*point_cloud_value.y + point_cloud_value.z*point_cloud_value.z);
printf("Distance to Camera at (%d, %d): %f mm\n", x, y, distance);
```

Once 50 frames have been grabbed, we close the camera.

```
// Close the camera
zed.close();
```

You are now using the ZED as a depth sensor.You can move on to the next tutorial to learn how to use the ZED as a positional tracker.

