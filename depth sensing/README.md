# Stereolabs ZED - Depth Sensing

This sample demonstrates how to grab the current point cloud with the ZED SDK and how to display it in a 3D view with OpenGL / freeGLUT.
It also allows you to save depth information (depth map image and point cloud) in different formats (PNG 16bit, PFM, PGM, XYZ, PCD, PLY, VTK).

## Build the program

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

- Navigate to the build directory and launch the executable file
- Or open a terminal in the build directory and run the sample :

        ./ZED_Depth_Sensing

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
