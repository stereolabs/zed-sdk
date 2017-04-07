# Tutorial 4: Positional tracking with the ZED


This tutorial shows how to use the ZED as a positional tracker. The program will loop until 1000 position are grabbed.
We assume that you have read previous tutorials (Tutorial 1 - Opening the ZED at least).

## Create a camera

As with previous tutorial, we create, configure and open the ZED. In this tutorial, we choose a right handed coordinate system with Y axis going up. We also want to get the position (translation) in meters.<br/>
Those parameters not only refers to the positional tracking but also the depth units and depth axis. That is why they are located in the initialization parameters.

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

Once the camera is opened, we must activate the positional tracking module to be able to get the position and orientation of the ZED.

```
// Enable positional tracking with default parameters
sl::TrackingParameters tracking_parameters;
err = zed.enableTracking(tracking_parameters);
if (err != SUCCESS)
    exit(-1);
```

In the above example, we leave the default parameters for tracking. Note that it is possible to configure more deeply the positional tracking module:<br/>
For example, we could use a previously created area file for spatial memory in case the ZED is in the same area. Those parameters are more detailed in the documentation and will be subject to another tutorial.

## Capture data

Now that the ZED is opened and the positional tracking enable, we can create a loop to grab and retrieve the camera position.

The camera position is given by the class sl::Pose. This class contains the translation and orientation of the camera, as well as the image timestamp this pose refers to and the tracking confidence (quality).<br/>
A pose is always linked to a reference frame. The SDK provides two reference frame : REFERENCE_FRAME_WORLD and REFERENCE_FRAME_CAMERA.<br/> It is not the purpose of this tutorial to go into the details of both reference frame. For that it is recommended to read the documentation for this section.<br/>
In the example, we are using the default reference frame : REFERENCE_FRAME_WORLD.

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

This will loop until the ZED has been tracked for 1000 frames. We are displaying the translation (in meters since this is what we have chosen during initialization) in the console window.

Once done, never forget to disable the tracking module and close the camera before exiting the program.

```
// Disable positional tracking and close the camera
zed.disableTracking();
zed.close();
return 0;
```

And this is it!<br/>
You can now use the ZED as an inside-out positional tracker. A good use case is to use the ZED with a VR headset: Take a look at the unity plug-in if you want to go more deeply in a ZED/VR headset combination.

You can also move on to the next tutorial to learn how to use the spatial mapping module.



*You can find the complete source code of this sample in main.cpp located in the same folder*
