# Tutorial 8: Body Tracking with the ZED 2

This tutorial shows how to use the object detection module with the ZED 2.<br/>
We assume that you have followed previous tutorials.

### Prerequisites

- Windows 10
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

## Build the program

Download the sample and follow the instructions below: [More](https://www.stereolabs.com/docs/getting-started/application-development/)

#### Build for Windows

- Create a "build" folder in the source folder
- Open cmake-gui and select the source and build folders
- Generate the Visual Studio `Win64` solution
- Open the resulting solution and change configuration to `Release`
- Build solution

# Code overview
## Create a camera

As in previous tutorials, we create, configure and open the ZED 2. Please note that the ZED 1 is **not** compatible with the object detection module.

This module uses the GPU to perform deep neural networks computations. On platforms with limited amount of memory such as jetson Nano, it's advise to disable the GUI to improve the performances and avoid memory overflow.

```C#
// Create ZED objects
Camera zedCamera = new Camera(0);
InitParameters init_params = new InitParameters();
init_params.resolution = RESOLUTION.HD720;
init_params.coordinateUnits = UNIT.METER;
init_params.sdkVerbose = true;

// Open the camera
ERROR_CODE err = zedCamera.Open(ref init_params);
if (err != ERROR_CODE.SUCCESS)
    Environment.Exit(-1);
```

## Enable Object detection

We will define the object detection parameters. Notice that the object tracking needs the positional tracking to be able to track the objects in the world reference frame.

```C#
// Define the Objects detection module parameters
ObjectDetectionParameters object_detection_parameters = new ObjectDetectionParameters();
object_detection_parameters.detectionModel = sl.DETECTION_MODEL.HUMAN_BODY_FAST;
object_detection_parameters.enableObjectTracking = true;
object_detection_parameters.imageSync = true;

// Object tracking requires the positional tracking module
// Enable positional tracking
PositionalTrackingParameters trackingParams = new PositionalTrackingParameters();
// If you want to have object tracking you need to enable positional tracking first
err = zedCamera.EnablePositionalTracking(ref trackingParams);
```

Then we can start the module, it will load the model. This operation can take a few seconds. The first time the module is used, the model will be optimized for the hardware and will take more time. This operation is done only once.

```C#
Console.WriteLine("Object Detection: Loading Module...");
err = zedCamera.EnableObjectDetection(ref object_detection_parameters);
if (err != ERROR_CODE.SUCCESS)
    Environment.Exit(-1);
```

The object detection is now activated.

## Capture data

The object confidence threshold can be adjusted at runtime to select only the revelant persons depending on the scene complexity. Since the parameters have been set to `image_sync`, for each `grab` call, the image will be fed into the AI module and will output the detections for each frames.

```C#
// Detection runtime parameters
ObjectDetectionRuntimeParameters obj_runtime_parameters = new ObjectDetectionRuntimeParameters();
obj_runtime_parameters.detectionConfidenceThreshold = 40;

// Detection output
Objects objects = new Objects();

while (zedCamera.Grab(ref runtimeParameters) == ERROR_CODE.SUCCESS){
	zed_error = zedCamera.RetrieveObjects(ref objects, ref obj_runtime_parameters);

	if (Convert.ToBoolean(objects.isNew)){
		Console.WriteLine(objects.numObject + " Person(s) detected");
	}
}
```

## Disable modules and exit

Once the program is over the modules can be disabled and the camera closed. This step is optional since the `zed.close()` will take care of disabling all the modules. This function is also called automatically by the destructor if necessary.<br/>

```C#
// Disable object detection and close the camera
zedCamera.DisableObjectDetection();
zedCamera.DisablePositionalTracking("");
zedCamera.Close();
```

And this is it!<br/>

You can now detect object in 3D with the ZED 2.
