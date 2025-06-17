# ZED SDK - SVO Playback

This sample demonstrates how to read a SVO video file.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 
## Run the program

To run the program, use the following command in your terminal:
```bash
python svo_playback.py --input_svo_file <input_svo_file> 
```

Arguments: 
   - --input_svo_file Path to an existing .svo file 

### Features
 - Displays readed frame as an OpenCV image
 - Press 's' to save the current image as a PNG
 - Press 'f' to move forward in the recorded file
 - Press 'b' to move backward in the recorded file

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/