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
            // Create the camera
            Camera zedCamera = new Camera(0);
            // Create default configuration parameters
            InitParameters init_params = new InitParameters();
            ERROR_CODE err = zedCamera.Open(ref init_params);
            if (err != ERROR_CODE.SUCCESS)
                Environment.Exit(-1);

            // Get camera information (serial number)
            int zed_serial = zedCamera.GetZEDSerialNumber();
            Console.WriteLine("Hello! This is my serial number: " + zed_serial);
            Console.ReadLine();

            zedCamera.Close();
        }
    }
}
