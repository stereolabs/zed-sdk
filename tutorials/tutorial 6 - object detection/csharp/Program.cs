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
            // Set Initialization parameters
            InitParameters init_params = new InitParameters();
            init_params.resolution = RESOLUTION.HD2K;
            init_params.coordinateUnits = UNIT.METER;
            init_params.coordinateSystem = COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP;
            init_params.depthMode = DEPTH_MODE.PERFORMANCE;
            init_params.sdkVerbose = true;

            Camera zedCamera = new Camera(0);
            // Open the camera
            ERROR_CODE err = zedCamera.Open(ref init_params);
            if (err != ERROR_CODE.SUCCESS)
                Environment.Exit(-1);

            // Enable positional tracking
            PositionalTrackingParameters trackingParams = new PositionalTrackingParameters();
            err = zedCamera.EnablePositionalTracking(ref trackingParams);
            if (err != ERROR_CODE.SUCCESS)
            {
                Console.WriteLine("ERROR in Enable Tracking. Exiting...");
                Environment.Exit(-1);
            }


            // Enable Object Detection
            ObjectDetectionParameters object_detection_parameters = new ObjectDetectionParameters();
            object_detection_parameters.detectionModel = sl.DETECTION_MODEL.MULTI_CLASS_BOX;
            object_detection_parameters.enableObjectTracking = true;
            object_detection_parameters.maxRange = -1;
            object_detection_parameters.batchParameters = new BatchParameters();
            object_detection_parameters.enable2DMask = false;
            object_detection_parameters.enableObjectTracking = true;

            err = zedCamera.EnableObjectDetection(ref object_detection_parameters);
            if (err != ERROR_CODE.SUCCESS)
            {
                Console.WriteLine("ERROR in Enable OD. Exiting...");
                Environment.Exit(-1);
            }
            // Create Runtime parameters
            RuntimeParameters runtimeParameters = new RuntimeParameters();

            // Create Object Detection frame handle (contains all the objects data)
            sl.Objects objects = new sl.Objects();
            // Create object detection runtime parameters (confidence, ...)
            ObjectDetectionRuntimeParameters obj_runtime_parameters = new ObjectDetectionRuntimeParameters();
            obj_runtime_parameters.detectionConfidenceThreshold = 50;

            int i = 0;
            while (i < 1000)
            {
                if (zedCamera.Grab(ref runtimeParameters) == ERROR_CODE.SUCCESS)
                {
                    //Retrieve Objects from Object detection
                    err  = zedCamera.RetrieveObjects(ref objects, ref obj_runtime_parameters);
                    //Console.WriteLine("Retrieve objects state : " + (ERROR_CODE)err);
                    // Display the data each 10 frames
                    if (i % 10 == 0)
                    {
                        Console.WriteLine("Nb Objects Detection : " + objects.numObject);
                        for (int p = 0; p < objects.numObject; p++)
                        {
                            Console.WriteLine("Position of object " + p + " : " + objects.objectData[p].position + "Tracked? : " + objects.objectData[p].objectTrackingState);
                        }
                    }
                    i++;
                }
            }

            // Disable object detection, positional tracking and close the camera
            zedCamera.DisableObjectDetection();
            zedCamera.DisablePositionalTracking("");
            zedCamera.Close();
        }
    }
}