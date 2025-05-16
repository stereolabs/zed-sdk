# Tutorial 10: Split Process with the ZED

This tutorial shows how to get an image then the depth from the ZED SDK. The program will loop until 50 frames are grabbed.
We assume that you have followed previous tutorials (opening the ZED, image capture and depth sensing).

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

As in other tutorials, we create, configure and open the ZED.
We set the ZED in HD720 mode at 60fps and enable depth in NEURAL mode. The ZED SDK provides different depth modes: NEURAL_PLUS, NEURAL, NEURAL_LIGHT. For more information, see online documentation.

```
// Create a ZED camera
Camera zed;

// Create configuration parameters
InitParameters init_params;
init_params.sdk_verbose = 1; // Enable the verbose mode
init_params.depth_mode = DEPTH_MODE::NEURAL; // Set the depth mode to NEURAL
init_params.coordinate_units = UNIT::MILLIMETER; // Use millimeter units

// Open the camera
ERROR_CODE err = zed.open(init_params);
if (err != ERROR_CODE::SUCCESS) {
    std::cout << "Error " << err << ", exit program.\n"; // Display the error
    return -1;
}
```

<i>Note: Default parameter for depth mode is DEPTH_MODE::NEURAL. In practice, it is not necessary to set the depth mode in InitParameters. </i>

## Capture data

In previous tutorials, you have seen that images and depth can be acquired using the grab method (see the Depth Sensing Tutorial).

It is also possible to acquire the image first and then the depth. To do this, you can use the read method and then the grab method.

This split between image capture (read) and depth computation (grab) also allows you to create a custom workflow where you can have a different frame rate between images and depth information. Just remember that most of our modules are based on the depth data, so you still need to call grab if you want to be able to retrieve body data for example.


```
while (frame_count < 150) {
        // A new image is available if read() returns ERROR_CODE::SUCCESS
        if (zed.read() == ERROR_CODE::SUCCESS) {
            // Retrieve left image
            zed.retrieveImage(image, VIEW::LEFT);
            frame_count++;
        }

        // Measurement are available if grab() returns ERROR_CODE::SUCCESS
        if(((frame_count % depth_every_n_frames) == 0) && (zed.grab() == ERROR_CODE::SUCCESS)) {
            // Retrieve depth map. Depth is aligned on the left image
            zed.retrieveMeasure(depth, MEASURE::DEPTH);
            // Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA);

            // Get and print distance value in mm at the center of the image
            // We measure the distance camera - object using Euclidean distance
            const int x = point_cloud.getWidth() / 2;
            const int y = point_cloud.getHeight() / 2;
            sl::float4 point_cloud_value;
            point_cloud.getValue(x, y, &point_cloud_value);

            if(std::isfinite(point_cloud_value.z)) // convert to float3 to use norm(), the 4th component is used to store the color
                cout<<"Distance to Camera at {"<<x<<";"<<y<<"}: "<<sl::float3(point_cloud_value).norm()<<"mm"<<endl;
            else
                cout<<"The Distance can not be computed at {"<<x<<";"<<y<<"}"<<endl;
            depth_count++;            
        }
    }
```

Once 150 frames have been grabbed, we compute the image and depth framerate and close the camera.

```
auto diff = (sl::getCurrentTimeStamp() - start_ts).getSeconds();
// Print the FPS
if(diff > 0)
    std::cout << "Image: " << (frame_count / diff) << "FPS / Depth: " << (depth_count / diff) <<"FPS" << std::endl;

// Close the camera
zed.close();
```