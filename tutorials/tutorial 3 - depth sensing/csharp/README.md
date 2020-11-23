# Tutorial 3: Depth sensing with the ZED

This tutorial shows how to get the depth or XYZ value from the ZED C# SDK. The program will loop until 1000 frames are grabbed.
We assume that you have followed previous tutorials (opening the ZED and image capture).

### Prerequisites

- Windows 10, Ubuntu LTS
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

## Build for Windows

- Create a "build" folder in the source folder
- Open cmake-gui and select the source and build folders
- Generate the Visual Studio `Win64` solution
- Open the resulting solution and change configuration to `Release`
- Build solution

# Code overview
## Create a camera

As in other tutorials, we create, configure and open the ZED.
We set the ZED in HD720 mode at 60fps and enable depth in PERFORMANCE mode. The ZED SDK provides different depth modes: PERFORMANCE, QUALITY, ULTRA. For more information, see online documentation.

## Capture data

Now that the ZED is opened, we can capture images and depth. Retrieving the depth map is as simple as retrieving an image:
* We create a Mat to store the depth map.
* We call retrieveMeasure() to get the depth map.
* We call retrieveMeasure() to get the point cloud.

```csharp
Camera.RetrieveMeasure(depth_map, MEASURE.XYZRGBA);
float4 xyz_value;
depth_map.GetValue(i, j, out xyz_value, MEM.CPU);
```

In the above code, (i,j) are the pixel coordinates. (0,0) refers to the top left corner of the image.

