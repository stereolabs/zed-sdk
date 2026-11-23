# ZED SDK - Multi Camera Positional Tracking

## This sample shows how to track a rig of several ZED cameras as one rigid body

### Features
 - The cameras of the rig feed a single positional tracking and build a single shared map, through the Sensors API
 - ZED stereo and ZED One (ZED X One) cameras can be mixed in one rig
 - An OpenGL window displays the rig in 3D: every camera at its pose, the path of the rig, the map and the image of any camera
 - The rig trajectory, every camera trajectory and the map can be saved

See the [C++ sample](./cpp/) to describe your rig and run it.
