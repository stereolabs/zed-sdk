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

            SensorsData sensors_data = new SensorsData();
            ulong last_imu_timestamp = 0;

            RuntimeParameters runtimeParameters = new RuntimeParameters();
            while (zed.Grab(ref runtimeParameters) == ERROR_CODE.SUCCESS)
            {
                zed.GetSensorsData(ref sensors_data, TIME_REFERENCE.CURRENT);
                if (sensors_data.imu.timestamp > last_imu_timestamp)
                {
                    // Show Sensors Data
                    Console.WriteLine("IMU Orientation : " + sensors_data.imu.fusedOrientation);
                    Console.WriteLine("Angular Velocity : " + sensors_data.imu.angularVelocity);
                    Console.WriteLine("Magnetometer Magnetic field : " + sensors_data.magnetometer.magneticField);
                    Console.WriteLine("Barometer Atmospheric pressure : " + sensors_data.barometer.pressure);
                    last_imu_timestamp = sensors_data.imu.timestamp;
                    // Wait for the [ENTER] key to be pressed
                    Console.ReadLine();
                }
            }
            zed.Close();
        }
    }
}
