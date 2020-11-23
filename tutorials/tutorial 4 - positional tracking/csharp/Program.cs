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
            init_params.resolution = RESOLUTION.HD720;
            init_params.cameraFPS = 60;
            init_params.coordinateUnits = UNIT.METER;
            init_params.coordinateSystem = COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP;
            init_params.depthMode = DEPTH_MODE.PERFORMANCE;

            Camera zedCamera = new Camera(0);
            // Open the camera
            ERROR_CODE err = zedCamera.Open(ref init_params);
            if (err != ERROR_CODE.SUCCESS)
                Environment.Exit(-1);

            PositionalTrackingParameters positionalTrackingParameters = new PositionalTrackingParameters();
            err = zedCamera.EnablePositionalTracking(ref positionalTrackingParameters);
            if (err != ERROR_CODE.SUCCESS)
                Environment.Exit(-1);

            int i = 0;
            sl.Pose pose = new Pose();

            RuntimeParameters runtimeParameters = new RuntimeParameters();
            while (i < 1000)
            {
                if (zedCamera.Grab(ref runtimeParameters) == ERROR_CODE.SUCCESS)
                {
                    // Get the pose of the left eye of the camera with reference to the world frame
                    zedCamera.GetPosition(ref pose,REFERENCE_FRAME.WORLD);

                    // Display the translation and timestamp each 10 frames
                    if (i%10==0)
                    Console.WriteLine("Translation : " + pose.translation + ", Rotation : " + pose.rotation + ", Timestamp : " + pose.timestamp);

                    i++;
                }
            }

            // Disable positional tracking and close the camera
            zedCamera.DisablePositionalTracking("");
            zedCamera.Close();
        }
    }
}
