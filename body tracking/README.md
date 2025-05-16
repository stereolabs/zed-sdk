# Body Tracking Samples

This repository contains samples demonstrating how to use the [ZED](https://www.stereolabs.com/store/) camera's **Body Tracking** features using the ZED SDK. You can find additional information on the Body Tracking module in our [Documentation](https://www.stereolabs.com/docs/body-tracking/) and [API Reference](https://www.stereolabs.com/docs/api/group__Body__group.html).

<p align="center">
  <img src="https://user-images.githubusercontent.com/32394882/230631989-24dd2b58-2c85-451b-a4ed-558d74d1b922.gif" />
</p>

## Overview

This section contains the following code samples:

- [Body Tracking](./body%20tracking/): This sample shows how to use the Body Tracking module, using a single camera and a simple 3D display.

- [Tracking Data Export](./export/): This sample shows how to export **human body tracking data** into a JSON format. You can adapt the code to fit your needs.

- [Integrations](./integrations) This folder contains links to other repositories that provide Body Tracking **integration examples** and tutorials with Unreal Engine 5, Unity, and Livelink.

- [Multi Camera Fusion](./multi-camera): This sample demonstrates how to use the ZED SDK **Fusion API** to track people in an entire space, with data from multiple cameras which produces higher quality results than using a single camera. The sample goes through the full process of setting up your cameras, calibrating your system with ZED360, fusing and visualizing the data.
