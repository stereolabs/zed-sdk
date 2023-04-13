# Object Detection

These samples show how to use ZED SDK for performing object detection. You can find additional information on the Object Detection module in our [Documentation](https://www.stereolabs.com/docs/object-detection/) and [API Reference](https://www.stereolabs.com/docs/api/group__Object__group.html).

<p align="center">
  <img src="https://user-images.githubusercontent.com/32394882/230630901-9d53502a-f3f9-45b6-bf57-027148bb18ad.gif" />
</p>

## Overview

This section contains the following code samples:

 - [Birds Eye Viewer](./birds%20eye%20viewer/) Detected objects are presented in a top-down view alongside the 3D point cloud, providing an intuitive perspective on object placement.

 - [Concurrent Detections](./concurrent%20detections/): This sample demonstrates how to simultaneously run **Object detection** and **Body Tracking** modules, which allow to use of both detectors in a single application.

 - [Image Viewer](./image%20viewer/): Detected objects are displayed in a 2D view making it simple to identify and track objects of interest.

 - [Custom Detector](./custom%20detector/): This sample shows how to use a custom object detector on ZED images. The ZED SDK then computes 3D information and performs object tracking on detected objects.
