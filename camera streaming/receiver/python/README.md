# ZED SDK - Streaming Receiver

This sample shows how to connect to a broadcasting device.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 
## Run the program
To run the program, use the following command in your terminal:
```bash
python streaming_receiver.py --ip_address <ip_adress>
```
Arguments: 
  - --ip_address IP address or hostname of the sender. Should be in format a.b.c.d:p or hostname:p


### Features
 - Connects to a network ZED device.
 - Display image from broadcast with Open-CV 

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