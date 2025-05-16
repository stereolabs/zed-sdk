# Tutorial 4: Spatial mapping with the ZED

This tutorial shows how to use the spatial mapping module with the ZED. It will loop until 500 frames are grabbed, extract a mesh, filter it and save it as a obj file.<br/>
We assume that you have followed previous tutorials.

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

## Build for Windows

- Create a "build" folder in the source folder
- Open cmake-gui and select the source and build folders
- Generate the Visual Studio `Win64` solution
- Open the resulting solution and change configuration to `Release`
- Build solution

# Code overview
## Create a camera

As in previous tutorials, we create, configure and open the ZED. In this example, we choose to have a right-handed coordinate system  with Y axis up, since it is the most common system chosen in 3D viewing software (meshlab for example).

```csharp
// Create a ZED camera object
Camera zed = new Camera(0);

// Set configuration parameters
InitParameters init_params;
init_params.resolution = RESOLUTION.HD720; // Use HD720 video mode (default fps: 60)
init_params.coordinateSystem = COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP; // Use a right-handed Y-up coordinate system
init_params.coordinateUnits = UNIT.METER; // Set units in meters

// Open the camera
ERROR_CODE err = zed.Open(ref init_params);
if (err != ERROR_CODE.SUCCESS)
    Enviromnent.Exit(-1);
```

## Enable positional tracking

The spatial mapping needs the positional tracking to be activated. Therefore, as with tutorial 4 - Positional tracking, we need to enable the tracking module first.


```csharp
sl.PositionalTrackingParameters tracking_parameters = new sl.PositionalTrackingParameters();
err = zed.EnablePositionalTracking(ref tracking_parameters);
if (err != ERROR_CODE.SUCCESS)
    exit(-1);
```

## Enable spatial mapping

Now that tracking is enabled, we need to enable the spatial mapping module. You will see that it is very close to the positional tracking: We create a spatial mapping parameters and call `enableSpatialMapping()` function with this parameter.

```csharp
sl.SpatialMappingParameters mapping_parameters = new sl.SpatialMappingParameters();
err = zed.EnableSpatialMapping(ref mapping_parameters);
if (err != ERROR_CODE.SUCCESS)
    exit(-1);
```

It is not the purpose of this tutorial to go into the details of `SpatialMappingParameters` class, but you will find mode information in the API documentation.

The spatial mapping is now activated.

## Capture data

The spatial mapping does not require any function call in the grab process. the ZED SDK handles and checks that a new image,depth and position can be ingested in the mapping module and will automatically launch the calculation asynchronously.<br/>
It means that you just simply have to grab images to have a mesh creating in background.<br/>
In this tutorial, we grab 500 frames and then stop the loop to extract mesh.

```csharp
// Grab data during 500 frames
int i = 0;
RuntimeParameters runtimeParameters = new RuntimeParameters();
while (i < 500) {
	if (zed.Grab(ref runtimeParameters) == ERROR_CODE.SUCCESS) {
		// In background, spatial mapping will use new images, depth and pose to create and update the mesh. No specific functions are required here.
		SPATIAL_MAPPING_STATE mapping_state = zed.GetSpatialMappingState();
		// Print spatial mapping state
		Console.WriteLine( "Images captured: " + i + "/ 500  |  Spatial mapping state: " + mapping_state.ToString());
		i++;
	}
}
```

## Extract mesh

We have now grabbed 500 frames and the mesh has been created in background. Now we need to extract it.<br/>
First, we need to create a mesh object to manipulate it: a `sl::Mesh`. Then launch the extraction with Camera::extractWholeMesh(). This function will block until the mesh is available.

```csharp
zed.ExtractWholeMesh(); // Extract the whole mesh
```

We have now a mesh. This mesh can be filtered (if needed) to remove duplicate vertices and unneeded faces. This will make the mesh lighter to manipulate.<br/>
Since we are manipulating the mesh, this function is a function member of `sl::Mesh`.<br/>

```csharp
// Filter the mesh (remove unnecessary vertices and faces)
zed.FilterMesh(FILTER.LOW, ref mesh); 

 ```

You can see that filter takes a filtering parameter. This allows you to fine tuning the processing. Likewise, more information are given in the API documentation regarding filtering parameters.


You can now save the mesh as an obj file for external manipulation:

```csharp
// Save the mesh in an obj file
zed.SaveMesh("mesh.obj", MESH_FILE_FORMAT.OBJ);
```

## Disable modules and exit

Once the mesh is extracted and saved, don't forget to disable the modules and close the camera before exiting the program.<br/>
Since spatial mapping requires positional tracking, always disable spatial mapping before disabling tracking.

```csharp
// Disable tracking and mapping and close the camera
 zed.DisableSpatialMapping();
 zed.DisablePositionalTracking();
 zed.Close();
```

And this is it!<br/>

You can now map your environment with the ZED.
