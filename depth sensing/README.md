# Depth Sensing Samples

This repository contains samples demonstrating how to use the [ZED]("https://www.stereolabs.com/store/") camera's **Depth Sensing features** using the ZED SDK.

<p align="center">
<img src="https://user-images.githubusercontent.com/32394882/230639409-356b8dfa-df66-4bc2-84d8-a25fd0229779.gif" />
</p>

## Overview

This section contains the following code samples:

- [Depth Sensing sample](./depth%20sensing): This sample demonstrates how to extract depth information from a single ZED camera and visualize it in an OpenGL window.
- [Depth Sensing with multiple cameras](./multi%20camera): This sample provides an example of how to design an app that use **multiple ZED cameras** in separated threads and displays RGB and depth maps in OpenCV.
- [Region Of Interest](./region%20of%20interest)  This sample showcases how to define a Region of Interest (ROI), pixels outside of this area are discard from the all modules (depth, positional tracking, detections ...).
- [Image Refocus](./image%20refocus) This sample illustrates how to apply depth-dependent blur to an image, allowing users to adjust the focal point after the image has been taken.

