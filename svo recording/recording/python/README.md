# Stereolabs ZED - SVO Recording utilities

This sample shows how to record video in Stereolabs SVO format.
SVO video files can be played with the ZED API and used with its different modules.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

## Run the sample

The default recording compression is H264 which provides fast encoding and efficently compressed SVO file. It can be changed to H265 or LOSSLESS (image based PNG compression) depending on the hardware capabilities. [See API Documentation for more information](https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1SVO__COMPRESSION__MODE.html)

```
python svo_recording.py svo_file.svo
```

Use Ctrl-C to stop the recording.