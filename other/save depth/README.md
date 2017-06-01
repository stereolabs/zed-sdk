# Stereolabs ZED - Saving depth maps

This sample allows you to save depth information (depth map image and point cloud) provided by the ZED camera or an SVO file in different formats (PNG 16bit, PFM, PGM, XYZ, PCD, PLY, VTK).

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

### Prerequisites

- Windows 7 64bits or later, Ubuntu 16.04
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads), [OpenCV 3.1](http://opencv.org/downloads.html))

## Build the program

Download the sample and follow these instructions:

#### Build for Windows

- Create a "build" folder in the source folder
- Open cmake-gui and select the source and build folders
- Generate the Visual Studio `Win64` solution
- Open the resulting solution and change configuration to `Release`
- Build solution

#### Build for Linux

Open a terminal in the sample directory and execute the following command:

    mkdir build
    cd build
    cmake ..
    make

## Run the program

Open a terminal in build directory and execute the following command:

     ./ZED\ Save\ depth [option]=[arg]

Example :

     ./ZED\ Save\ depth --resolution=3 --mode=2


### Launch options

Option                    |               Descriptions             |                 Available Arguments                 |         Default Argument
 -----------------------------------------|----------------------------------------|-----------------------------------------------------|------------------------------
 --filename                              | SVO input filename (optional).                          | Path to an input SVO file                                 | <none>
 --path                                  | Specify the location to save image, depth and point cloud.                            | Output path                                         | ./
 --resolution                            | Specify ZED video resolution.   | "0": HD2K, "1" : HD1080, "2" : HD720, "3" : VGA     | 2
 --mode                               | Specify depth map quality mode.      | "1": PERFORMANCE, "2": MEDIUM, "3": QUALITY         | 1
 --device                                | If multiple GPUs are available, select a GPU device for depth computation.	By default, (-1) will select the GPU with the highest number of CUDA cores.                            |  GPU ID                                        | -1
 --help, -h , -?, --usage                | Display help message.                   |                                                     |

### Keyboard shortcuts

This table lists keyboard shortcuts that you can use in the sample application.


Feature                    |           Description            | Hotkeys                   
-----------------------------|----------------------------------|-------------
Save side-by-side images     | Save side-by-side color image in PNG file format.        |'w'                                                         
Depth map file format      | Specify file format of the saved depth map: 'PNG' (16-Bit, in mm), 'PFM', 'PGM'.     |'n'         
Save depth         | Save the depth image in the selected format.            |'d'                   
Point cloud file format    | Specify file format of the saved point cloud: 'XYZ', 'PCD' (ASCII), 'PLY', 'VTK'.   |'m'              
Save point cloud         | Save the point cloud in the selected format.            |'p'                                                      
Quit                         | Quit the application.                  |'q'         
