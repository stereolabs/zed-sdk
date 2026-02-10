# ZED SDK - Positional Tracking
This sample shows how to track the camera pose in a world frame of reference.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)

### Build the program
 - Build for [Windows](https://www.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://www.stereolabs.com/docs/app-development/cpp/linux/)
 
## Usage
Navigate to the build directory and launch the executable with desired options:
```
Usage: ./ZED_Positional_Tracking [options]

Options:
  --help                      Shows usage information.
  --resolution <mode>         Optional. Resolution options: (HD2K | HD1200 | HD1080 | HD720 | SVGA | VGA)
  --svo <filename.svo>        Optional. Use SVO file input. Mutually exclusive with --stream.
  --stream <ip[:port]>        Optional. Use network streaming input. Mutually exclusive with --svo.
  -i <input_area_file>        Optional. Input area file used in explore mode (default) or --map mode
  --map -o <output_area_file> Optional. Map mode creates or updates an .area file.
                                        Requires -o <output_area_filename> for generated map
  --roi <roi_filepath>        Optional. Region of interest image mask to ignore a static area
  --custom-initial-pose       Optional. Use custom initial pose (see code comments for more detail)
  --2d-ground-mode            Optional. Enable 2D ground mode
  --export-tum                Optional. Export camera trajectory to out.tum file in TUM format

Examples:
  ./build/ZED_Positional_Tracking --map -o new_map.area
  ./build/ZED_Positional_Tracking --svo recording.svo2 -i map.area
```

The sample will take the provided input (connected camera, SVO file, or stream) and display the video feed, tracking status, and 3D rendering of the camera's current position. When using the `--map` option, a `.area` file that contains the mapped area will be exported after the tracking session.  
  
You can interact with the sample while it's running:
```
'space' to toggle camera view visibility
'd' to switch background color from dark to light
'p' to enable / disable current live point cloud display
'l' to enable / disable landmark display
'f' to follow the camera
'z' to reset the view
'ctrl' + drag to rotate
'esc' to exit
```

### Support
If you need assistance go to our Community site at https://community.stereolabs.com/
