# ZED SDK - Body Tracking

This sample shows how to detect and track human bodies in space.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)

## Dependencies
```
python3 -m pip install pyopengl
```
 
## Run the program
*NOTE: The ZED v1 is not compatible with this module*

To run the program, use the following command in your terminal : 
```bash
python body_tracking.py
```
If you wish to run the program from an input_svo_file, or an IP adress, or specify a resolution run : 

```bash
python body_tracking.py --input_svo_file <input_svo_file> --ip_address <ip_address> --resolution <resolution> 
```
Arguments: 
  - --input_svo_file A path to an existing .svo file, that will be playbacked. If this parameter and ip_adress are not specified, the soft will use the camera wired as default.  
  - --ip_address IP Address, in format a.b.c.d:port or a.b.c.d. If specified, the soft will try to connect to the IP.
  - --resolution Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA
## Features
 - Display bodies bounding boxes by pressing the `b` key.

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/