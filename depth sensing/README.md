# Depth Sensing Samples

This repository contains samples demonstrating how to use the [ZED]("https://www.stereolabs.com/store/") camera's **Depth Sensing features** using the ZED SDK.

<p align="center">
<img src="https://user-images.githubusercontent.com/32394882/230639409-356b8dfa-df66-4bc2-84d8-a25fd0229779.gif" />
</p>

## Overview

This repository contains multiple samples, including:
- **depth sensing:** This sample demonstrates how to extract depth information from a single ZED camera and visualize it in an OpenGL window.
- **image refocus:** This sample illustrates how to apply depth-dependent blur to an image, allowing users to adjust the focal point after the image has been taken.
- **multi camera:** This sample provides an example of how to design an app that use multiple ZED cameras in separated threads.
- **region of interest:**  This sample showcases how to define a Region of Interest (ROI), pixels outside of this area are discard from the all modules (depth, positional tracking, detections ...).
