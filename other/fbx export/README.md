# ZED SDK - FBX Export


This sample shows how to export ZED Camera data or skeleton data as FBX file compatible with 3D software like Blender or Maya.
It is using the [FBX SDK](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-2-1), make sure it is installed on your computer.

____

## Export Camera Path

This sample shows how to export the camera path as an FBX file. The trajectory will be saved as an animation and can be imported later in a 3D software such as Blender or Maya.

## Export Skeleton data

This sample shows how to export skeleton data detected by the ZED SDK to an FBX file. The skeleton data will be saved as animations and can be imported later in a 3D software such as Blender or Maya.
____

## Build the program
 - Make sure you have installed the latest ZED SDK and the FBX SDK (C++ API).

 - Set the FBX install directory in the [cmake](CMakeLists.txt#L18) file. By default, it is installed at :
*C:/Program Files/Autodesk/FBX/FBX SDK/yourFBXSDKversion*.


and then build the sample (build both samples at once) :

```
mkdir build
cd build
cmake ..
make -j
```

## Run the program

- Navigate to the build directory and launch the executable
- Or open a terminal in the build directory and run the sample :

      ./ZED_FBX_Camera
      ./ZED_FBX_Skeletons

## Support

If you need assistance go to our Community site at https://community.stereolabs.com/
