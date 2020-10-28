# Tutorial 6: Object Detection with the ZED 2

This tutorial shows how to use the object detection module with the ZED 2.<br/>
We assume that you have followed previous tutorials.

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

As in previous tutorials, we create, configure and open the ZED 2. Please note that the ZED 1 is **not** compatible with the object detection module.

This module uses the GPU to perform deep neural networks computations. On platforms with limited amount of memory such as jetson Nano, it's advise to disable the GUI to improve the performances and avoid memory overflow.

```cpp
// Create ZED objects
Camera zed;
InitParameters initParameters;
initParameters.camera_resolution = RESOLUTION::HD720;
initParameters.depth_mode = DEPTH_MODE::PERFORMANCE;
initParameters.sdk_verbose = true;

// Open the camera
ERROR_CODE zed_error = zed.open(initParameters);
if (zed_error != ERROR_CODE::SUCCESS) {
	std::cout << "Error " << zed_error << ", exit program.\n";
	return 1; // Quit if an error occurred
}
```

## Enable Object detection

We will define the object detection parameters. Notice that the object tracking needs the positional tracking to be able to track the objects in the world reference frame.

```cpp
// Define the Objects detection module parameters
ObjectDetectionParameters detection_parameters;
detection_parameters.enable_tracking = false;
detection_parameters.enable_mask_output = false;
detection_parameters.image_sync = false;

// Object tracking requires the positional tracking module
if (detection_parameters.enable_tracking)
	zed.enablePositionalTracking();
```

Then we can start the module, it will load the model. This operation can take a few seconds. The first time the module is used, the model will be optimized for the hardware and will take more time. This operation is done only once.

```cpp
std::cout << "Object Detection: Loading Module..." << std::endl;
zed_error = zed.enableObjectDetection(detection_parameters);
if (zed_error != ERROR_CODE::SUCCESS) {
	std::cout << "Error " << zed_error << ", exit program.\n";
	zed.close();
	return 1;
}
```

The object detection is now activated.

## Capture data

The object confidence threshold can be adjusted at runtime to select only the revelant objects depending on the scene complexity. Since the parameters have been set to `image_sync`, for each `grab` call, the image will be fed into the AI module and will output the detections for each frames.

```cpp
// Detection runtime parameters
ObjectDetectionRuntimeParameters detection_parameters_rt;
detection_parameters_rt.detection_confidence_threshold = 40;

// Detection output
Objects objects;

while (zed.grab() == ERROR_CODE::SUCCESS) {
	zed_error = zed.retrieveObjects(objects, detection_parameters_rt);

	if (objects.is_new) {
		std::cout << objects.object_list.size() << " Object(s) detected ("
				<< zed.getCurrentFPS() << " FPS)" << std::endl;
	}
}
```

## Disable modules and exit

Once the program is over the modules can be disabled and the camera closed. This step is optional since the `zed.close()` will take care of disabling all the modules. This function is also called automatically by the destructor if necessary.<br/>

```cpp
// Disable object detection and close the camera
zed.disableObjectDetection();
zed.close();
return 0;
```

And this is it!<br/>

You can now detect object in 3D with the ZED 2.
