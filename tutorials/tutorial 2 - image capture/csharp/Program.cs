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
            init_params.resolution = RESOLUTION.HD1080;

            Camera zedCamera = new Camera(0);
            // Open the camera
            ERROR_CODE err = zedCamera.Open(ref init_params);
            if (err != ERROR_CODE.SUCCESS)
                Environment.Exit(-1);

            // Get resolution of camera
            uint mWidth = (uint)zedCamera.ImageWidth;
            uint mHeight = (uint)zedCamera.ImageHeight;

            // Initialize the Mat that will contain the left image
            Mat image = new Mat();
            image.Create(mWidth, mHeight, MAT_TYPE.MAT_8U_C4, MEM.CPU); // Mat need to be created before use.

            // defin default Runtime parameters
            RuntimeParameters runtimeParameters = new RuntimeParameters();

            // Initialize runtime parameters and frame counter
            int i = 0;
            while (i < 1000)
            {
                if (zedCamera.Grab(ref runtimeParameters) == ERROR_CODE.SUCCESS)
                {
                    zedCamera.RetrieveImage(image, VIEW.LEFT); // Get the left image
                    ulong timestamp = zedCamera.GetCameraTimeStamp(); // Get image timestamp
                    Console.WriteLine("Image resolution: " + image.GetWidth() + "x" + image.GetHeight() +"|| Image timestamp: " + timestamp);
                    // increment frame count
                    i++;
                }
            }

            // Disable positional tracking and close the camera
            zedCamera.DisablePositionalTracking("");
            zedCamera.Close();
        }
    }
}
