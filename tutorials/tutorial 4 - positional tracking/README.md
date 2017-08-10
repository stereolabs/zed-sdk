# Tutorial 4: Positional tracking with the ZED

This tutorial shows how to use the ZED as a positional tracker. The program will loop until 1000 position are grabbed.
We assume that you have followed previous tutorials.

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

As in previous tutorials, we create, configure and open the ZED. 

```
// Create a ZED camera object
Camera zed;

// Set configuration parameters
InitParameters init_params;
init_params.camera_resolution = RESOLUTION_HD720; // Use HD720 video mode (default fps: 60)
init_params.coordinate_system = COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP; // Use a right-handed Y-up coordinate system
init_params.coordinate_units = UNIT_METER; // Set units in meters

// Open the camera
ERROR_CODE err = zed.open(init_params);
if (err != SUCCESS)
    exit(-1);
```

## Enable positional tracking

Once the camera is opened, we must enable the positional tracking module in order to get the position and orientation of the ZED.

```
// Enable positional tracking with default parameters
sl::TrackingParameters tracking_parameters;
err = zed.enableTracking(tracking_parameters);
if (err != SUCCESS)
    exit(-1);
```

In the above example, we leave the default tracking parameters. For the list of available parameters, check the online documentation.

## Capture pose data

Now that the ZED is opened and the positional tracking enabled, we create a loop to grab and retrieve the camera position.

The camera position is given by the class sl::Pose. This class contains the translation and orientation of the camera, as well as image timestamp and tracking confidence (quality).<br/>
A pose is always linked to a reference frame. The SDK provides two reference frame : REFERENCE_FRAME_WORLD and REFERENCE_FRAME_CAMERA.<br/> It is not the purpose of this tutorial to go into the details of these reference frame. Read the documentation for more information.<br/>
In the example, we get the device position in the World Frame.

```
// Track the camera position during 1000 frames
int i = 0;
sl::Pose zed_pose;
while (i < 1000) {
    if (zed.grab() == SUCCESS) {

        zed.getPosition(zed_pose, REFERENCE_FRAME_WORLD); // Get the pose of the left eye of the camera with reference to the world frame

        // Display the translation and timestamp
        printf("Translation: Tx: %.3f, Ty: %.3f, Tz: %.3f, Timestamp: %llu\n", zed_pose.getTranslation().tx, zed_pose.getTranslation().ty, zed_pose.getTranslation().tz, zed_pose.timestamp);

        // Display the orientation quaternion
        printf("Orientation: Ox: %.3f, Oy: %.3f, Oz: %.3f, Ow: %.3f\n\n", zed_pose.getOrientation().ox, zed_pose.getOrientation().oy, zed_pose.getOrientation().oz, zed_pose.getOrientation().ow);

  i++;
    }
}
```

This will loop until the ZED has been tracked for 1000 frames. We display the camera translation (in meters) in the console window and close the camera before exiting the application.

```
// Disable positional tracking and close the camera
zed.disableTracking();
zed.close();
return 0;
```

You can now use the ZED as an inside-out positional tracker. You can also read on the next tutorial to learn how to use Spatial Mapping.
