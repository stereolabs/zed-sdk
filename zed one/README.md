# ZED One

These samples demonstrate how to use the ZED SDK with the **ZED One** mono camera. The ZED One is a compact monocular camera that provides high-quality video capture, streaming, and recording capabilities. You can find additional information in our [Documentation](https://www.stereolabs.com/docs/).

## Overview

This section contains the following code samples:

- [Live](./live/): Shows how to capture and display live video from a ZED One camera with camera settings control.
- [SVO Recording](./svo_recording/): Shows how to **record** video to SVO format for later playback with the ZED SDK.
- [SVO Playback](./svo_playback/): Shows how to **play back** recorded SVO files and control playback.
- [SVO External Data](./svo_external_data/): Shows how to **embed and retrieve custom data** within SVO files.
- [Streaming Sender](./streaming_sender/): Shows how to **stream** live video from a ZED One over the network.
- [Streaming Receiver](./streaming_receiver/): Shows how to **receive and display** a video stream from a remote ZED One.
- [Custom Inference](./custom_inference/): Shows how to **retrieve** a tensor to run an AI model directly from the ZED SDK.

## Requirements

- ZED SDK 5.0 or later
- ZED One camera
- CMake 3.5+
- OpenCV (for visualization)
- CUDA

## Build

### Linux

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Windows

```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
