

<h1 align="center">
  <br>
  Fusion Samples
  <br>
</h1>

<p align="center">
The ZED SDK's fusion API is specifically designed to combine data from multiple cameras, resulting in higher quality data. The API can fuse data from several cameras to improve the accuracy and robustness of tracking systems. For instance, the fusion API can be used in outdoor robot tracking with GNSS to provide real-time fusion of the 3D position and orientation of the robot, even in challenging environments. Additionally, the API can be used with the ZED Camera's body tracking feature to fuse data from multiple cameras to track an entire space with much higher quality. This capability enables a range of applications that require accurate spatial tracking, such as robotics, autonomous vehicles, augmented reality, and virtual reality.
</p>

<p align="center">
  <a href="https://www.stereolabs.com/docs/">Documentation</a>
  ·
  <a href="https://www.stereolabs.com/docs/api/classsl_1_1Fusion.html">API Reference</a>
</p>

# Overview
This repository serves as a hub for finding other repositories related to the ZED SDK's fusion API. It provides a convenient way to discover and access additional resources related to the fusion API, including examples, tutorials, and integrations with other software platforms. These resources can be used to further explore the capabilities of the fusion API and to build more sophisticated applications that leverage the data fusion capabilities of the ZED Camera.

## Body tracking
![Body Tracking](https://user-images.githubusercontent.com/32394882/230631989-24dd2b58-2c85-451b-a4ed-558d74d1b922.gif)
[The multi camera Body Tracking sample](/body%20tracking/multi-camera/), demonstrates how to combine multiple body detections from an array of cameras to create a more accurate and robust representation of the detected bodies. By fusing data from multiple cameras, the sample can improve the accuracy and robustness of the body tracking system, especially in challenging environments with occlusions or complex motions. The sample showcases the capabilities of the ZED SDK's fusion API and provides a starting point for building more sophisticated applications that require multi-camera body tracking.

## GeoTracking
![GeoTracking](https://user-images.githubusercontent.com/32394882/230602944-ed61e6dd-e485-4911-8a4c-d6c9e4fab0fd.gif)
[The GeoTracking sample](/geotracking/), demonstrates how to combine data from the ZED Camera and a Global Navigation Satellite System (GNSS) receiver for outdoor tracking applications. The sample showcases the fusion API of the ZED SDK and provides an example of how to use it to integrate data from multiple sources, such as the camera and GNSS receiver. By fusing data from these sources, the sample can improve the accuracy and robustness of the tracking system, especially in challenging outdoor environments. The sample provides a starting point for building more sophisticated applications that require outdoor tracking with the ZED Camera and GNSS.
