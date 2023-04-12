# Fusion Samples

The ZED SDK's Fusion API is designed to combine data from multiple cameras, resulting in higher quality data. The API can fuse data from several cameras to improve the accuracy and robustness of tracking systems.

For instance, the Fusion API can be used in outdoor robot tracking with GNSS to provide real-time fusion of the 3D position and orientation of the robot, even in challenging environments. Additionally, the API can be used with the ZED Camera's body tracking feature to fuse data from multiple cameras to track an entire space with much higher quality. This capability enables a range of applications that require accurate spatial tracking, such as robotics, autonomous vehicles, augmented reality, and virtual reality.


## Overview

This section lists the available modules available in the **Fusion API**. It provides a convenient way to discover and access additional resources related to the Fusion API, including examples, tutorials, and integrations with other software platforms. These resources can be used to further explore the capabilities of the Fusion API and to build more sophisticated applications that leverage the data fusion capabilities of the ZED Camera.

## Body tracking

<p align="center">
  <img src="https://user-images.githubusercontent.com/32394882/230631989-24dd2b58-2c85-451b-a4ed-558d74d1b922.gif" />
</p>


The [Multi camera Body Tracking sample](/body%20tracking/multi-camera/) demonstrates how to combine multiple body detections from an array of cameras to create a more accurate and robust representation of the detected bodies. By fusing data from multiple cameras, the sample can improve the accuracy and robustness of the body tracking system, especially in challenging environments with occlusions or complex motions. The sample showcases the capabilities of the ZED SDK's Fusion API and provides a starting point for building more sophisticated applications that require multi-camera body tracking.

## GeoTracking

<p align="center">
  <img src="https://user-images.githubusercontent.com/32394882/230602944-ed61e6dd-e485-4911-8a4c-d6c9e4fab0fd.gif" />
</p>

The [GeoTracking sample](/geotracking/) demonstrates how to combine data from the ZED Camera and a Global Navigation Satellite System (GNSS) receiver for outdoor tracking applications. The sample showcases the Fusion API of the ZED SDK and provides an example of how to use it to integrate data from multiple sources, such as the camera and GNSS receiver. By fusing data from these sources, the sample can improve the accuracy and robustness of the tracking system, especially in challenging outdoor environments. The sample provides a starting point for building more sophisticated applications that require outdoor tracking with the ZED Camera and GNSS.
