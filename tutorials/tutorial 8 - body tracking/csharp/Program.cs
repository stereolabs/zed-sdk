//======= Copyright (c) Stereolabs Corporation, All rights reserved. ===============
using System;
using System.Runtime.InteropServices;
using System.Numerics;

namespace sl
{
    class Program
    {
        static BodyTrackingParameters body_tracking_parameters;

        static void Main(string[] args)
        {
            // Set Initialization parameters
            InitParameters init_params = new InitParameters();
            init_params.resolution = RESOLUTION.HD720;
            init_params.coordinateUnits = UNIT.METER;
            init_params.sdkVerbose = 1;

            Camera zedCamera = new Camera(0);
            // Open the camera
            ERROR_CODE err = zedCamera.Open(ref init_params);
            if (err != ERROR_CODE.SUCCESS)
            {
                Console.WriteLine("ERROR in Open. Exiting...");
                Environment.Exit(-1);
            }


            // Enable positional tracking
            PositionalTrackingParameters trackingParams = new PositionalTrackingParameters();
            // If you want to have body tracking you need to enable positional tracking first
            err = zedCamera.EnablePositionalTracking(ref trackingParams);
            if (err != ERROR_CODE.SUCCESS)
            {
                Console.WriteLine("ERROR in Enable Tracking. Exiting...");
                Environment.Exit(-1);
            }


            // Enable Body Tracking
            body_tracking_parameters = new BodyTrackingParameters();
            // Different model can be chosen, optimizing the runtime or the accuracy
            body_tracking_parameters.detectionModel = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST;
            // Choose the appropriate body format for your application
            // (38 to have simplified hands, 70 for detailed hands, but more resource-heavy.)
            body_tracking_parameters.bodyFormat = sl.BODY_FORMAT.BODY_38;
            // track detects object across time and space
            body_tracking_parameters.enableObjectTracking = true;
            // run detection for every Camera grab
            body_tracking_parameters.imageSync = true;
            err = zedCamera.EnableBodyTracking(ref body_tracking_parameters);
            if (err != ERROR_CODE.SUCCESS)
            {
                Console.WriteLine("ERROR in EnableBodyTracking. Exiting...");
                Environment.Exit(-1);
            }


            // Create Runtime parameters
            RuntimeParameters runtimeParameters = new RuntimeParameters();

            // Create Body Tracking frame handle (contains all the objects data)
            sl.Bodies bodies = new sl.Bodies();
            // Create Body Tracking runtime parameters (confidence, ...)
            BodyTrackingRuntimeParameters bt_runtime_parameters = new BodyTrackingRuntimeParameters();
            bt_runtime_parameters.detectionConfidenceThreshold = 40;


            int nbDetection = 0;
            while (nbDetection < 100)
            {
                if (zedCamera.Grab(ref runtimeParameters) == ERROR_CODE.SUCCESS)
                {
                    // Retrieve Objects from Body Tracking
                    zedCamera.RetrieveBodies(ref bodies, ref bt_runtime_parameters);

                    if (Convert.ToBoolean(bodies.isNew))
                    {
                        Console.WriteLine(bodies.nbBodies + " Person(s) detected");
                        Console.WriteLine();
                        if (bodies.nbBodies > 0)
                        {
                            sl.BodyData firstBody = bodies.bodiesList[0];

                            Console.WriteLine("First Person attributes :");
                            Console.WriteLine(" Confidence (" + firstBody.confidence);

                            if (body_tracking_parameters.enableObjectTracking)
                            {
                                Console.WriteLine(" Tracking ID: " + firstBody.id + " tracking state: " + firstBody.trackingState +
                                    " / " + firstBody.actionState);
                            }

                            Console.WriteLine(" 3D Position: " + firstBody.position +
                                              " Velocity: " + firstBody.velocity);

                            Console.WriteLine(" Keypoints 2D");
                            // The body part meaning can be obtained by casting the index into a BODY_PARTS
                            // to get the BODY_PARTS index the getIdx function is available
                            for (int i = 0; i < (int)sl.BODY_38_PARTS.LAST; i++)
                            {
                                var kp = firstBody.keypoints2D[i];
                                Console.WriteLine("     " + (sl.BODY_38_PARTS)i + " " + kp.X + ", " + kp.Y);
                            }

                            // The BODY_PARTS can be link as bones, using sl::BODY_BONES which gives the BODY_PARTS pair for each
                            Console.WriteLine(" Keypoints 3D ");
                            for (int i = 0; i < (int)sl.BODY_38_PARTS.LAST; i++)
                            {
                                var kp = firstBody.keypoints[i];
                                Console.WriteLine("     " + (sl.BODY_38_PARTS)i + " " + kp.X + ", " + kp.Y + ", " + kp.Z);
                            }

                            nbDetection++;
                            Console.WriteLine();
                            Console.WriteLine("Press 'Enter' to continue...");
                            Console.ReadLine();
                        }
                    }
                }
            }

            // Disable Body Tracking, positional tracking and close the camera
            zedCamera.DisableBodyTracking();
            zedCamera.DisablePositionalTracking("");
            zedCamera.Close();
        }
    }
}
