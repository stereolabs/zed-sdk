# Tutorial 1: Hello ZED

This tutorial simply shows how to configure and open the ZED, then print its serial number and then close the camera. This is the most basic step and a good start for using the ZED SDK.


## Overview

In order to use the ZED in your application, you will need to create a Camera object, then configure and open it.
The SDK can handle two different inputs: the ZED itself (Live mode) or a SVO file created by the ZED (playback mode). The configuration is done before opening the camera and cannot be changed while the camera is in use.

## Configure your ZED

 First, we need to create an ZED camera object. Since we provide a default constructor, it will be as simple as :

```
// Create a ZED camera object
Camera zed;
```


The ZED camera object is now created but not configured.This configuration is done  using the initialization parameters `InitParameters`.<br/>
This class has different options you can configure. Note that those parameters cannot be changed during use. If you want to change them, you need to close the camera and open it again with a new set of parameters. <br/>
Those parameters are classified with their scope :

You can setup:
* the camera configuration parameters, using the `camera_*` entries (resolution, image flip...).
* the SDK configuration parameters, using the `sdk_*` entries (verbosity, GPU device used...).
* the depth configuration parameters, using the `depth_*` entries (depth mode, minimum distance...).
* the coordinates configuration parameters, using the `coordinate_*` entries (coordinate system, coordinates unit...).
* the SVO parameters if you want to work with SVO file in playback mode (filename, real-time mode...)



## Open the ZED and retrieve information

In this tutorial, we will mostly use the default parameters (no need to configure them) and just set the SDK verbosity at false (no console messages).

```
// Set configuration parameters
InitParameters init_params;
init_params.sdk_verbose = false; // Disable verbose mode
```

Since initialization parameters are created and set, we can now open the camera. To do it, just call Camera::open() function with the initialization parameters we have just created.
Always check the status of this function to know if everything went well. If open() returns an error code different from SUCCESS, then refers to the API documentation for more information. In the example below, we exit the program, since there is no reason to keep on if camera didn't successfully open.

```
// Open the camera
ERROR_CODE err = zed.open(init_params);  
if (err != SUCCESS)
    exit(-1);
```

Since open() is done, we can now get the camera information such as the serial number:

```
// Get camera information (serial number)
int zed_serial = zed.getCameraInformation().serial_number;
printf("Hello! This is my serial number: %d\n", zed_serial);
```

In the console window, you should see a line with the serial number of your camera (also available on the ZED USB cable).

<i> Note: </i>The `CameraInformation` class also contains the firmware version of the ZED, as well as calibration parameters.

Once done, just close the camera and exit the program.

```
// Close the camera
zed.close();
return 0;
```

And this is it!<br/>

*You can find the complete source code of this sample here*
