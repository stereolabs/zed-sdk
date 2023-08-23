# ZED SDK - Positional Tracking

This sample shows how to get the camera pose in a world reference

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 
## Run the program

To run the program, use the following command in your terminal : 
```bash
python positional_tracking.py
```
If you wish to run the program from an input_svo_file, or an IP adress, or specify a resolution or a mode concerning imu, run : 
```bash
python positional_tracking.py --input_svo_file <input_svo_file> --ip_address <ip_address> --resolution <resolution> --imu_only
```
Arguments: 
  - --input_svo_file A path to an existing .svo file, that will be playbacked with plane detection feature. If this parameter and ip_adress are not specified, the soft will use the camera wired as default.  
  - --ip_address IP Address, in format a.b.c.d:port or a.b.c.d. If specified, the soft will try to connect to the IP and apply the plane detection features. 
  - --resolution Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA
  - --imu_only Either the tracking should be done with imu data only (that will remove translation estimation  
### Features
 - An OpenGL window displays the camera path in a 3D window
 - path data, translation and rotation, are displayed

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/