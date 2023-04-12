 <h1 align="center">
  <br>
  ZED SDK - Object Detection
  <br>
</h1>

<p align="center">
These samples show how to use ZED SDK for performing object detection
</p>

![Object Detection](https://user-images.githubusercontent.com/32394882/230630901-9d53502a-f3f9-45b6-bf57-027148bb18ad.gif)

<p align="center">
  <a href="https://www.stereolabs.com/docs/object-detection/">Documentation</a>
  ·
  <a href="https://www.stereolabs.com/docs/api/classsl_1_1Camera.html">API Reference</a>
</p>

# Overview
This repository contains multiple samples, including:
 - **Birds eye viewer**: Detected objects are presented in an immersive 3D view alongside the current point cloud, providing an intuitive perspective on object placement.
 - **concurrent detections:** This sample demonstrates how to simultaneously run Object detection and Body Tracking, this allows you to take advantage of both detectors.
 - **Image viewer**: Detected objects are displayed in a clear, easily digestible 2D format alongside the current image, making it simple to identify and track objects of interest.
 - **Custom detector**: This sample shows how the user can use its own detector on the ZED images and then use the ZED API to compute 3D information and perform tracking on the detected objects.
