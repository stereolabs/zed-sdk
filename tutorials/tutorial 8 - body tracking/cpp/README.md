# Tutorial 6: Body Tracking with the ZED 2

This tutorial shows how to use the body tracking module with the ZED 2.<br/>
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

As in previous tutorials, we create, configure and open the ZED 2. Please note that the ZED 1 is **not** compatible with the body tracking module.

This module uses the GPU to perform deep neural networks computations. On platforms with limited amount of memory such as jetson Nano, it's advise to disable the GUI to improve the performances and avoid memory overflow.

```cpp
// Create ZED objects
Camera zed;
InitParameters initParameters;
initParameters.camera_resolution = RESOLUTION::AUTO;
initParameters.depth_mode = DEPTH_MODE::NEURAL;
init_parameters.coordinate_units = UNIT::METER;
initParameters.sdk_verbose = true;

// Open the camera
ERROR_CODE zed_error = zed.open(initParameters);
if (zed_error != ERROR_CODE::SUCCESS) {
	std::cout << "Error " << zed_error << ", exit program.\n";
	return 1; // Quit if an error occurred
}
```

## Enable Object detection

We will define the bodies detection parameters. Notice that the body tracking needs the positional tracking to be able to track the bodies in the world reference frame.

```cpp
// Define the Objects detection module parameters
BodyTrackingParameters detection_parameters;
// Different model can be chosen, optimizing the runtime or the accuracy
detection_parameters.detection_model = BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
// Body format
detection_parameters.body_format = BODY_FORMAT::BODY_38;
// Track the detected bodies across time and space
detection_parameters.enable_tracking = true;
// Optimize the person joints position, requires more computations
detection_parameters.enable_body_fitting = true;

// If you want to have body tracking you need to enable positional tracking first
if (detection_parameters.enable_tracking)
	zed.enablePositionalTracking();
```

Then we can start the module, it will load the model. This operation can take a few seconds. The first time the module is used, the model will be optimized for the hardware and will take more time. This operation is done only once.

```cpp
cout << "Body Tracking: Loading Module..." << endl;
returned_state = zed.enableBodyTracking(detection_parameters);
if (returned_state != ERROR_CODE::SUCCESS) {
	cout << "Error " << returned_state << ", exit program.\n";
	zed.close();
	return EXIT_FAILURE;
}
```

The body tracking is now activated.

## Capture data

The bodies confidence threshold can be adjusted at runtime to select only the revelant persons depending on the scene complexity. For each `grab` call, the image will be fed into the AI module and will output the detections for each frames.

```cpp
// Detection runtime parameters
BodyTrackingRuntimeParameters detection_parameters_rt;
// For outdoor scene or long range, the confidence should be lowered to avoid missing detections (~20-30)
// For indoor scene or closer range, a higher confidence limits the risk of false positives and increase the precision (~50+)
detection_parameters_rt.detection_confidence_threshold = 40;

// Detection output
Bodies bodies;

int nb_detection = 0;
while (nb_detection < 100) {

	if (zed.grab() == ERROR_CODE::SUCCESS) {
		zed.retrieveBodies(bodies, detection_parameters_rt);

		if (bodies.is_new) {
			cout << bodies.body_list.size() << " Person(s) detected\n\n";
			// Do something with the bodies
		}
	}
}
```

## Disable modules and exit

Once the program is over the modules can be disabled and the camera closed. This step is optional since the `zed.close()` will take care of disabling all the modules. This function is also called automatically by the destructor if necessary.<br/>

```cpp
// Disable body tracking and close the camera
zed.disableBodyTracking();
zed.close();
return 0;
```

And this is it!<br/>

You can now detect bodies in 3D with the ZED 2.
