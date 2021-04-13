//======= Copyright (c) Stereolabs Corporation, All rights reserved. ===============
using System;
using System.Runtime.InteropServices;
using System.Numerics;

namespace sl
{
    class Program
    {
        static ObjectDetectionParameters object_detection_parameters;

        static void Main(string[] args)
        {
            // Set Initialization parameters
            InitParameters init_params = new InitParameters();
            init_params.resolution = RESOLUTION.HD720;
            init_params.coordinateUnits = UNIT.METER;
            init_params.sdkVerbose = true;

            Camera zedCamera = new Camera(0);
            // Open the camera
            ERROR_CODE err = zedCamera.Open(ref init_params);
            if (err != ERROR_CODE.SUCCESS)
                Environment.Exit(-1);

            // Enable positional tracking
            PositionalTrackingParameters trackingParams = new PositionalTrackingParameters();
            // If you want to have object tracking you need to enable positional tracking first
            err = zedCamera.EnablePositionalTracking(ref trackingParams);
            if (err != ERROR_CODE.SUCCESS)
                Environment.Exit(-1);

            // Enable Object Detection
            object_detection_parameters = new ObjectDetectionParameters();
            // Different model can be chosen, optimizing the runtime or the accuracy
            object_detection_parameters.detectionModel = sl.DETECTION_MODEL.HUMAN_BODY_FAST;
            // track detects object across time and space
            object_detection_parameters.enableObjectTracking = true;
            // run detection for every Camera grab
            object_detection_parameters.imageSync = true;
            err = zedCamera.EnableObjectDetection(ref object_detection_parameters);
            if (err != ERROR_CODE.SUCCESS)
                Environment.Exit(-1);

            // Create Runtime parameters
            RuntimeParameters runtimeParameters = new RuntimeParameters();

            // Create Object Detection frame handle (contains all the objects data)
            sl.Objects objects = new sl.Objects();
            // Create object detection runtime parameters (confidence, ...)
            ObjectDetectionRuntimeParameters obj_runtime_parameters = new ObjectDetectionRuntimeParameters();
            obj_runtime_parameters.detectionConfidenceThreshold = 40;


            int nbDetection = 0;
            while (nbDetection < 100)
            {
                if (zedCamera.Grab(ref runtimeParameters) == ERROR_CODE.SUCCESS)
                {
                    // Retrieve Objects from Object detection
                    zedCamera.RetrieveObjects(ref objects, ref obj_runtime_parameters);
                    
                    if (Convert.ToBoolean(objects.isNew))
                    {
                        Console.WriteLine(objects.numObject + " Person(s) detected");
                        Console.WriteLine();
                        if (objects.numObject > 0)
                        {
                            sl.ObjectData firstObject = objects.objectData[0];

                            Console.WriteLine("First Person attributes :");
                            Console.WriteLine(" Confidence (" + firstObject.confidence);

                            if (object_detection_parameters.enableObjectTracking)
                            {
                                Console.WriteLine(" Tracking ID: " + firstObject.id + " tracking state: " + firstObject.objectTrackingState +
                                    " / " + firstObject.actionState);
                            }

                            Console.WriteLine(" 3D Position: " + firstObject.position +
                                              " Velocity: " + firstObject.velocity);

                            Console.WriteLine(" Keypoints 2D");
                            // The body part meaning can be obtained by casting the index into a BODY_PARTS
                            // to get the BODY_PARTS index the getIdx function is available
                            for (int i = 0; i < firstObject.keypoints2D.Length; i++)
                            {
                                var kp = firstObject.keypoints2D[i];
                                Console.WriteLine("     " + (sl.BODY_PARTS)i + " " + kp.X + ", " + kp.Y);
                            }

                            // The BODY_PARTS can be link as bones, using sl::BODY_BONES which gives the BODY_PARTS pair for each
                            Console.WriteLine(" Keypoints 3D ");
                            for (int i = 0; i < firstObject.keypoints.Length; i++)
                            {
                                var kp = firstObject.keypoints[i];
                                Console.WriteLine("     " + (sl.BODY_PARTS)i + " " + kp.X + ", " + kp.Y + ", " + kp.Z);
                            }

                            Console.WriteLine();
                            Console.WriteLine("Press 'Enter' to continue...");
                            Console.ReadLine();
                        }
                    }             
                }
            }

            // Disable object detection, positional tracking and close the camera
            zedCamera.DisableObjectDetection();
            zedCamera.DisablePositionalTracking("");
            zedCamera.Close();
        }
    }
}