# Stereolabs ZED - Video

This sample shows how to tweak the ZED Camera parameters with the ZED SDK.

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

## Run the program

- Navigate to the build directory and launch the executable file
- Or open a terminal in the build directory and run the sample :

        ./ZED_Video

## Features

This sample demonstrates how to grab images and change the camera settings with the ZED SDK.
The following parameters can be changed :

  - Exposure
  - Gain
  - Saturation
  - Hue
  - Contrast
  - Brightness

This sample also lets you display the unrectified images as well as gray images.


### Keyboard shortcuts

This table lists keyboard shortcuts that you can use in the sample application.

Parameter             | Description                   |   Hotkey
---------------------|------------------------------------|-------------------------------------------------
Left image view      | Display left rectified RGB image.                      |         '0'                             
Right image view      | Display right rectified RGB image.                        |          '1'                              
Left gray image view      | Display left rectified gray image.                |         '2'                             
Right gray image view      | Display right rectified gray image.              |          '3'                              
Left unrectified image view      | Display left unrectified RGB image.                      |         '4'                             
Right unrectified image view      | Display right unrectified RGB image.                        |          '5'                              
Left unrectified gray image view      | Display left unrectified gray image.                      |         '6'                             
Right unrectified gray image view      | Display right unrectified gray image.                        |          '7'                              
Reset calibration      | Re-compute stereo alignment on the fly                       |          'a'                              
Increase exposure | Increase exposure by 1 (max. value 100) | 'g'
Decrease exposure | Decrease exposure by 1 (min. value 0) | 'h'
Auto exposure | Restore auto exposure | 'j'
Increase gain | Increase gain by 1 (max. value 100) | 't'
Decrease gain | Decrease gain by 1 (min. value 0) | 'y'
Auto gain | Restore auto gain | 'u'
Increase saturation | Increase saturation by 1 (max. value 8) | 's'
Decrease saturation | Decrease saturation by 1 (min. value 0) | 'd'
Auto saturation | Restore auto saturation | 'f'
Increase hue | Increase hue by 1 (max. value 11) | 'v'
Decrease hue | Decrease hue by 1 (min. value 0) | 'b'
Auto hue | Restore auto hue | 'n'
Increase contrast | Increase contrast by 1 (max. value 8) | 'i'
Decrease contrast | Decrease contrast by 1 (min. value 0) | 'o'
Auto contrast | Restore auto contrast | 'p'
Increase brightness | Increase brightness by 1 (max. value 8) | 'w'
Decrease brightness | Decrease brightness by 1 (min. value 0) | 'x'
Auto brightness | Restore auto brightness | 'c'
Increase color temperature (white balance) | Increase temperature by 100 (max. value 6500) | 'k'
Decrease color temperature (white balance) | Decrease temperature by 100 (min. value 2800) | 'l'
Auto white balance | Restore auto brightness | 'm'
Auto mode | Restore all parameters to auto mode | 'z'
Exit         | Quit the application.             | 'q'

