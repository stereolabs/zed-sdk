# Streaming

This sample shows how to create a **video stream** from a ZED camera which can be read by a remote application for viewing or further processing. You can find additional information on the Video Streaming module in our [Documentation](https://www.stereolabs.com/docs/video/streaming/) and [API Reference](https://www.stereolabs.com/docs/api/structsl_1_1StreamingParameters.html).

## Overview

This repository contains the following code samples:

- [ZED Stream Sender](./sender): This sample demonstrates how to use the ZED SDK to establish a network connection, encode a live video stream, and transmit it to a remote client. The ZED SDK handles all aspects of network communication, video encoding, and transmission, making it easy to integrate the ZED camera into applications that require **remote video capture and processing**, such as computer vision, remote monitoring, or teleoperation.

- [ZED Stream Receiver](./receiver/): This sample demonstrates how to use the ZED SDK to **receive and decode** video data sent from a remote ZED stream.

