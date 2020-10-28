# Stereolabs ZED - Using multiple ZED

This sample shows how to use multiple ZED cameras in a single application.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/latest).

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads), [OpenCV](https://github.com/opencv/opencv/releases))

## How it works

- Video capture for each camera is done in a separate thread for optimal performance. You can specify the number of ZED used by changing the `NUM_CAMERAS` parameter.
- Each camera has its own timestamp (uncomment a line to display it). These timestamps can be used for device synchronization.
- OpenCV is used to display the images and depth maps. To stop the application, simply press 'q'.

### Limitations

- This sample works on Windows with the latest firmware v.1523
- USB bandwidth: The ZED  in 1080p30 mode generates around 250MB/s of image data. USB 3.0 maximum bandwidth is around 400MB/s, so the number of cameras, resolutions and framerates you can use on a single machine will be limited by the USB 3.0 controller on the motherboard. When bandwidth limit is exceeded, corrupted frames (green or purple frames, tearing) can appear.
- Using a single USB 3.0 controller, here are configurations that we tested:
  - 2 ZEDs in HD1080 @ 15fps and HD720 @ 30fps
  - 3 ZEDs in HD720 @ 15fps
  - 4 ZEDs in VGA @ 30fps
- To use multiple ZED at full speed on a single computer, we recommend adding USB3.0 PCIe expansion cards.
- You can also use multiple GPUs to load-balance computations (use `param.device` to select a GPU for a ZED) and improve performance.
