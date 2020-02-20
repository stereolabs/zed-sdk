# Stereolabs ZED - Streaming Receiver

This sample shows how to open a stream from an other ZED and display the left image.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

### Prerequisites

- Windows 10, Ubuntu LTS
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [ZED Python API](https://www.stereolabs.com/docs/app-development/python/)
- OpenCV python (`python -m pip install opencv-python`)

## Run the example
    
This sample need a sender to operate (See streaming sender sample)

```
python streaming_receiver.py <sender_IP>
```