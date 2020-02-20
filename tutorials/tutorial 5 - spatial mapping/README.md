# Tutorial 5: Spatial mapping with the ZED

This tutorial shows how to use the spatial mapping module with the ZED. It will loop until 500 frames are grabbed, extract a mesh, filter it and save it as a obj file.<br/>
We assume that you have followed previous tutorials.

### Prerequisites

- Windows 10, Ubuntu LTS
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

# Code overview

## Create a camera

In this example, we choose to have a right-handed coordinate system  with Y axis up, since it is the most common system chosen in 3D viewing software (meshlab for example).

## Enable positional tracking

The spatial mapping needs the positional tracking to be activated. Therefore, as with tutorial 4 - Positional tracking, we need to enable the tracking module first.

## Enable spatial mapping

Now that tracking is enabled, we need to enable the spatial mapping module. You will see that it is very close to the positional tracking: We create a spatial mapping parameters and call `enableSpatialMapping()` function with this parameter.

It is not the purpose of this tutorial to go into the details of `SpatialMappingParameters` class, but you will find mode information in the API documentation.

## Capture data

The spatial mapping does not require any function call in the grab process. the ZED SDK handles and checks that a new image,depth and position can be ingested in the mapping module and will automatically launch the calculation asynchronously.

It means that you just simply have to grab images to have a mesh creating in background.

## Extract mesh

We have now grabbed 500 frames and the mesh has been created in background. Now we need to extract it.

First, we need to create a mesh object to manipulate it: a `sl::Mesh`. Then launch the extraction with Camera::extractWholeMesh(). This function will block until the mesh is available.

This mesh can be filtered (if needed) to remove duplicate vertices and unneeded faces. This will make the mesh lighter to manipulate.
Since we are manipulating the mesh, this function is a function member of `sl::Mesh`.<br/>


You can see that filter takes a filtering parameter. This allows you to fine tuning the processing. Likewise, more information are given in the API documentation regarding filtering parameters.


You can now save the mesh as an obj file for external manipulation.

## Disable modules and exit

Once the mesh is extracted and saved, don't forget to disable the modules and close the camera before exiting the program.

Since spatial mapping requires positional tracking, always disable spatial mapping before disabling tracking.