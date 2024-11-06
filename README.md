<h1 align="center">
  <!--- Stereolabs Banner --->
  <!--- a href="http://www.stereolabs.com/docs"><img src="https://user-images.githubusercontent.com/32394882/228559403-1da06352-9bac-4279-a5b4-e68bafcb2b1c.jpg" alt="Stereolabs"></a --->
  ZED SDK
  <br>
</h1>

<p align="center">
  The ZED SDK is a cross-platform library designed to get the best out of the <a href="https://www.stereolabs.com/store/">ZED</a> cameras. 
  <br />
  In this project, we provide tutorials and code samples to get started using the ZED SDK API.
</p>

<p align="center">
  <a href="https://www.stereolabs.com">Website</a>
  ·
  <a href="https://store.stereolabs.com/">Store</a>
  ·
  <a href="https://www.stereolabs.com/docs/api/">API Reference</a>
  ·
  <a href="https://community.stereolabs.com/">Community</a>
  ·
  <a href="https://www.stereolabs.com/blog/">Blog</a>
</p>

<p align="center">
  <a href="https://www.stereolabs.com/developers/release"><img src="https://img.shields.io/github/v/release/stereolabs/zed-sdk?color=%2300aeec&label=ZED%20SDK" alt="SDK Version"></a>
  <a href="https://community.stereolabs.com/"><img src="https://img.shields.io/discourse/posts?server=https%3A%2F%2Fcommunity.stereolabs.com%2F" alt="ZED Discourse"></a>
  <a href="https://hub.docker.com/u/stereolabs"><img src="https://img.shields.io/docker/pulls/stereolabs/zed" alt="Docker Pulls"></a>
  <a href="https://github.com/stereolabs/zed-examples/stargazers"><img src="https://img.shields.io/github/stars/stereolabs/zed-sdk?style=social" alt="Github Stars"></a>
</p>

---

:tada: The **ZED SDK 4.2** is released! We support the [**ZED X**](https://www.stereolabs.com/zed-x/) and [**ZED X Mini**](https://www.stereolabs.com/zed-x/) cameras, added the **Fusion API** for multi-camera Body Tracking, and more! Please check the [Release Notes](https://www.stereolabs.com/developers/release/) of the latest version for more details.

## Overview

Depth Sensing | Object Detection | Body Tracking |
:------------: |  :----------: | :-------------:  |
[![Depth Sensing](https://user-images.githubusercontent.com/32394882/230639409-356b8dfa-df66-4bc2-84d8-a25fd0229779.gif)](https://www.stereolabs.com/docs/depth-sensing)  | [![Object Detection](https://user-images.githubusercontent.com/32394882/230630901-9d53502a-f3f9-45b6-bf57-027148bb18ad.gif)](https://www.stereolabs.com/docs/object-detection)  | [![Body Tracking](https://user-images.githubusercontent.com/32394882/230631989-24dd2b58-2c85-451b-a4ed-558d74d1b922.gif)](https://www.stereolabs.com/docs/body-tracking)  |

Positional Tracking | Global Localization | Spatial Mapping |
:------------: |  :----------: | :-------------:  |
[![Positional Tracking](https://user-images.githubusercontent.com/32394882/229093429-a445e8ae-7109-4995-bc1d-6a27a61bdb60.gif)](https://www.stereolabs.com/docs/positional-tracking/) | [![Global Localization](https://user-images.githubusercontent.com/32394882/230602944-ed61e6dd-e485-4911-8a4c-d6c9e4fab0fd.gif)](/global%20localization) | [![Spatial Mapping](https://user-images.githubusercontent.com/32394882/229099549-63ca7832-b7a2-42eb-9971-c1635d205b0c.gif)](https://www.stereolabs.com/docs/spatial-mapping) |

Camera Control | Plane Detection | Multi Camera Fusion |
:------------: |  :----------: | :-------------:  |
[![Camera Control](https://user-images.githubusercontent.com/32394882/230602616-6b57c351-09c4-4aba-bdec-842afcc3b2ea.gif)](https://www.stereolabs.com/docs/video/camera-controls/) | [![Plane Detection](https://user-images.githubusercontent.com/32394882/229093072-d9d70e92-07d5-46cb-bde7-21f7c66fd6a1.gif)](https://www.stereolabs.com/docs/spatial-mapping/plane-detection/)  | [![Multi Camera Fusion](https://user-images.githubusercontent.com/32394882/228791106-a5f971d8-8d6f-483b-9f87-7f0f0025b8be.gif)](/fusion) |


## Why ZED?

- 🎯 End-to-end spatial perception platform for human-like sensing capabilities.
- ⚡ Real-time performance: all algorithms of the ZED SDK are designed and optimized to run in real-time. 
- 📷 Reduce time-to-market with our comprehensive, ready-to-use hardware and software designed for multiple applications.
- 📖 User-friendly and intuitive, with easy-to-use integrations and well-documented API for streamlined development.
- 🛠️ Wide range of supported platforms, from desktop to embedded PCs.

## Getting started

The ZED SDK contains all the libraries that power your camera along with tools that let you experiment with its features and settings.

To get started:
- [Get a ZED from the Stereolabs Store](https://store.stereolabs.com/)
- [Download the ZED SDK](https://www.stereolabs.com/developers/release/#downloads)
- [Install the ZED SDK](https://www.stereolabs.com/docs/installation/) on [Windows](https://www.stereolabs.com/docs/installation/windows/), [Linux](https://www.stereolabs.com/docs/installation/linux/) or [Jetson](https://www.stereolabs.com/docs/installation/jetson/)
- [Start experimenting with the ZED SDK's tutorials](/tutorials)

The [documentation](https://www.stereolabs.com/docs/) and [API reference](https://www.stereolabs.com/docs/api/) are great starting points to learn more about the ZED SDK and its many modules.

## Samples

This repository contains ready-to-use and samples to start using the ZED SDK with only a few lines of code. They are organized by ZED SDK module: 

* [**Tutorials**](/tutorials) - A series of basic tutorials that demonstrate the use of each API module.

* [**Camera Control**](/camera%20control) - This sample shows how to adjust the **ZED camera parameters**.

* [**Camera Streaming**](/camera%20streaming) - This sample shows how to **stream** and receive on local network the ZED's video feed.

* [**Depth Sensing**](/depth%20sensing) - This sample shows how to capture a **3D point cloud** and display with OpenGL. It also shows how to save depth data in different formats.

* [**Positional Tracking**](/positional%20tracking) - This sample shows how to use **positional tracking** and display the result with *OpenGL*.

* [**Global Localization**](/global%20localization) - This sample shows how to fuse the ZED SDK's **positional tracking with GNSS data** for global positioning.

* [**Spatial Mapping**](/spatial%20mapping) - This sample shows how to capture **3D meshes** with the ZED and display it with *OpenGL*. Classic Mesh and Point Cloud fusion are available.

* [**Object Detection**](/object%20detection) - This sample shows how to use the **Object Detection API** module with the ZED.

* [**Body Tracking**](/body%20tracking) - This sample shows how to use the **Body Tracking API** with the ZED.

* [**Recording**](/recording) - This sample shows how to **record** and **playback** video files in SVO format. SVO files let you use all the ZED SDK features without having a ZED connected.

## Supported platforms

Here is the list of all supported operating systems for the latest version of the ZED SDK. Please find the [recommended specifications](https://www.stereolabs.com/docs/installation/specifications/) to make sure your configuration is compatible with the ZED SDK.

| Ubuntu LTS | Windows | Jetson |
| -------- | ------------------------- | ----------------- |
| <div align="center"><a href="https://www.stereolabs.com/docs/installation/linux"><img src="https://user-images.githubusercontent.com/32394882/230619268-bdf66472-8bf5-41e7-9efa-ca3698ff271a.png" width="40%" alt="" /></a></div>  | <div align="center"><a href="https://www.stereolabs.com/docs/installation/windows"><img  src="https://user-images.githubusercontent.com/32394882/230619282-fe2f84fb-2130-4164-a193-db2893b58272.png" width="40%" alt="" /></a></div> | <div align="center"><a href="https://www.stereolabs.com/docs/installation/jetson/"><img src="https://user-images.githubusercontent.com/32394882/230619273-feeee52b-209b-48da-b990-06630cabe323.png" width="40%" alt="" /></a></div>

The ZED SDK requires the use of an **NVIDIA GPU** with a **Compute Capability > 5**.

If you are not familiar with the corresponding versions between NVIDIA JetPack SDK and Jetson Linux, please take a look at our [blog post](https://www.stereolabs.com/blog/nvidia-jetson-l4t-and-jetpack-support/). 


## Integrations

The ZED SDK can be easily integrated into projects using the following programming languages:

| C++ | Python | C# | C |
| -------- | ------------------------- | ----------------- | -------- | 
| <div align="center"><a href="https://www.stereolabs.com/docs/api"><img src="https://user-images.githubusercontent.com/32394882/229499695-c71857a2-eded-4171-8185-4e522d5b6c71.png" width="50%" alt="" /></a></div>  | <div align="center"><a href="https://www.stereolabs.com/docs/api/python/"><img src="https://user-images.githubusercontent.com/32394882/229499718-c66c3649-d139-48e5-8523-65b23a120440.png" width="50%" alt="" /></a></div> | <div align="center"><a href="https://www.stereolabs.com/docs/api/csharp"><img src="https://user-images.githubusercontent.com/32394882/229499667-5e4c4d72-1140-4eda-b206-d9f95c93c15c.png" width="50%" alt="" /></a></div> | <div align="center"><a href="https://www.stereolabs.com/docs/api/c/"><img src="https://user-images.githubusercontent.com/32394882/229499672-9b5308a2-3294-4818-84c5-07f84834a6d9.png" width="50%" alt="" /></a></div>

<br />

Thanks to its comprehensive API, ZED cameras can be interfaced with **multiple third-party libraries** and environments.

| Unity | Unreal Engine 5 | OpenCV | ROS | ROS 2
| -------- | ------------------------- | ----------------- | ----- | ----- |
| <div align="center"><a href="https://www.stereolabs.com/docs/unity/"><img src="https://user-images.githubusercontent.com/32394882/229497186-d77d9d1f-5eb8-420a-851e-3513d982d78d.png" width="70%" alt="" /></a></div>  | <div align="center"><a href="https://www.stereolabs.com/docs/ue5/"><img  src="https://user-images.githubusercontent.com/32394882/229497196-19fb4d4c-423d-4ae3-abba-b26ca384e7e4.png" width="70%" alt="" /></a></div> | <div align="center"><a href="https://www.stereolabs.com/docs/opencv/"><img src="https://user-images.githubusercontent.com/32394882/229497204-09a267bd-cbcf-4d3f-b9a2-f95349b4b7a9.png" width="70%" alt="" /></a></div> | <div align="center"><a href="https://www.stereolabs.com/docs/ros/"><img src="https://user-images.githubusercontent.com/32394882/229475890-452f2cc0-1a9a-4b2a-87ad-cacc44a7435e.png" width="70%" alt="" /></a></div> | <div align="center"><a href="https://www.stereolabs.com/docs/ros2/"><img src="https://user-images.githubusercontent.com/32394882/230614232-ce7a208e-7cf0-47a2-9c6f-5203e6034a5d.png" width="70%" alt="" /></a></div>

| Pytorch | YOLO | Matlab | Isaac SIM | Touch Designer |  
| -------- | ------------------------- | ----------------- | ----- | ----- |
| <div align="center"><a href="https://www.stereolabs.com/docs/pytorch"><img src="https://user-images.githubusercontent.com/32394882/229475918-1add790d-b10e-4529-a1d7-097f015a481f.png" width="70%" alt="" /></a></div>  | <div align="center"><a href="https://www.stereolabs.com/docs/yolo/"><img src="https://user-images.githubusercontent.com/32394882/230623781-3c87a5c9-b6af-4f93-bcc7-5ec381acf5d7.png" width="70%" alt="" /></a></div> | <div align="center"><a href="https://www.stereolabs.com/docs/matlab/"><img src="https://user-images.githubusercontent.com/32394882/229472074-4747f789-4ce6-4aef-b4f7-eab3bb77ab52.png" width="70%" alt="" /></a></div> | <div align="center"><a href="https://www.stereolabs.com/"><img src="https://user-images.githubusercontent.com/32394882/229472012-fe8d4458-219b-4825-8e87-9a3e9bc55e62.png" width="70%" alt="" /></a></div> | <div align="center"><a href="https://derivative.ca/UserGuide/ZED"><img src="https://user-images.githubusercontent.com/32394882/230623653-630e7bd2-1300-47ad-8133-39543470b2b1.png" width="70%" alt="" /></a></div>


<br />

## Community

Join the conversation and connect with other ZED SDK users to share ideas, solve problems, and help make the ZED SDK awesome. Our aim is to make it extremely convenient for everyone to communicate with us.

- **Discourse** is our forum where all ZED users can connect. This is the best place to brainstorm and exchange about ZED cameras, ZED SDK software, and other Stereolabs products. Feel free to create an account and ask your questions, or even share your awesome projects!

- **Twitter** Follow Stereolabs [@Stereolabs3D](https://twitter.com/stereolabs3d) for official news and release announcements.
- **GitHub** If you come across a bug, please raise an issue in this [**GitHub repository**](https://github.com/stereolabs/zed-examples/issues).

- **Email** To talk to Stereolabs directly, the easiest way is by email. Get in touch with us at support@stereolabs.com.

<br />
<br />

<div align="center">
  <a href="https://github.com/stereolabs" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/32394882/228892870-fbac3f33-49d9-4575-9a2b-10fc2ba26091.svg" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/32394882/228893668-ce93aa6e-0867-406f-9481-d1cb307a7dcf.png" width="3%" alt="" />
  <a href="https://www.linkedin.com/company/stereolabs" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/32394882/228892887-d12a8d98-4245-4121-8d23-52bd61431b29.svg" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/32394882/228893668-ce93aa6e-0867-406f-9481-d1cb307a7dcf.png" width="3%" alt="" />
  <a href="https://twitter.com/stereolabs3d" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/32394882/228892805-93d657be-a54c-4e12-83c6-6e7b15a256e2.svg" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/32394882/228893668-ce93aa6e-0867-406f-9481-d1cb307a7dcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/Stereolabs3d" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/32394882/228892815-f04bb1ce-aa42-49b0-bfe7-d1d051ead830.svg" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/32394882/228893668-ce93aa6e-0867-406f-9481-d1cb307a7dcf.png" width="3%" alt="" />
  <a href="https://community.stereolabs.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/32394882/228892794-8840d6c5-54bf-44d3-a95b-d9c51927914f.svg" width="3%" alt="" /></a>
</div>
