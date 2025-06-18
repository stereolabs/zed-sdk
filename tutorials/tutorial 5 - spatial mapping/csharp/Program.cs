//======= Copyright (c) Stereolabs Corporation, All rights reserved. ===============
using System;
using System.Runtime.InteropServices;
using System.Numerics;

namespace sl
{
    class Program
    {
        static void Main(string[] args)
        {
            // Set configuration parameters
            InitParameters init_params = new InitParameters();
            init_params.resolution = RESOLUTION.HD1080;
            init_params.cameraFPS = 30;
            Camera zed = new Camera(0);
            // Open the camera
            ERROR_CODE err = zed.Open(ref init_params);
            if (err != ERROR_CODE.SUCCESS)
                Environment.Exit(-1);

            // Configure spatial mapping parameters
            SpatialMappingParameters mappingParams = new SpatialMappingParameters();
            mappingParams.resolutionMeter = SpatialMappingParameters.get(MAPPING_RESOLUTION.LOW);
            mappingParams.rangeMeter = SpatialMappingParameters.get(MAPPING_RANGE.FAR);
            mappingParams.saveTexture = false;

            //Enable tracking and mapping
            PositionalTrackingParameters trackingParams = new PositionalTrackingParameters();
            zed.EnablePositionalTracking(ref trackingParams);
            zed.EnableSpatialMapping(ref mappingParams);

            RuntimeParameters runtimeParameters = new RuntimeParameters();

            int i = 0;
            Mesh mesh = new Mesh();

            // Grab 500 frames and stop
            while (i < 500)
            {
                if (zed.Grab(ref runtimeParameters) == ERROR_CODE.SUCCESS)
                {
                    SPATIAL_MAPPING_STATE state = zed.GetSpatialMappingState();
                    Console.WriteLine("Images captures: " + i + " /500 || Mapping state: " + state);
                    i++;
                }
            }
            // Retrieve the spatial map
            Console.WriteLine("Extracting mesh...");
            zed.ExtractWholeSpatialMap();
            // Filter the mesh
            Console.WriteLine("Filtering mesh...");
            zed.FilterMesh(MESH_FILTER.LOW, ref mesh); // not available for fused point cloud
                                                       // Apply the texture
            Console.WriteLine("Saving mesh...");
            zed.SaveMesh("mesh.obj", MESH_FILE_FORMAT.OBJ);
            zed.Close();
        }
    }
}
