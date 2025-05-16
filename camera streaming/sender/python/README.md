# Stereolabs ZED - Streaming Sender

This sample shows how to enable streaming module of the ZED SDK as image sender.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 
## Run the program

To run the program, use the following command in your terminal:
```bash
python streaming_sender.py --resolution <resolution>
```
Arguments: 
  - --resolution Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA


### Features
 - Defines camera resolution and its frame-rate
 - Broadcast Camera images on network

To setup a basic streaming setup, open two terminals, and navigate to the streaming_receiver and to the streaming_sender samples. Type in the first one : 
```bash
python streaming_sender.py
```
Find the port displayed, and then type in the other : 
```bash
python streaming_receiver.py --ip_address 127.0.0.1:port
```

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/