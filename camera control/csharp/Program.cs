///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/********************************************************************************
 ** This sample demonstrates how to grab images and change the camera settings **
 ** with the ZED SDK                                                           **
 ********************************************************************************/

using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Numerics;
using OpenCvSharp;
using sl;
using System.Net;

class Program
{
    // Sample variables
    static VIDEO_SETTINGS camera_settings_ = VIDEO_SETTINGS.BRIGHTNESS;
    static string str_camera_settings = "BRIGHTNESS";
    static int step_camera_setting = 1;
    static bool led_on = true;

    static void Main(string[] args)
    {
        Camera zed = new Camera(0);

        InitParameters initParameters = new InitParameters()
        {
            sdkVerbose = true,
            resolution = RESOLUTION.HD720,
            depthMode = DEPTH_MODE.NONE
        };

        parseArgs(args, ref initParameters);

        ERROR_CODE returnedState = zed.Open(ref initParameters);
        if (returnedState != ERROR_CODE.SUCCESS)
        {
            Environment.Exit(-1);
        }

        string winName = "Camera control";
        Cv2.NamedWindow(winName);

        Console.WriteLine("ZED Model             : " + zed.GetCameraModel());
        Console.WriteLine("ZED Serial Number     : " + zed.GetZEDSerialNumber());
        Console.WriteLine("ZED Camera Firmware   : " + zed.GetCameraFirmwareVersion());
        Console.WriteLine("ZED Camera Resolution : " + zed.GetInitParameters().resolution);
        Console.WriteLine("ZED Camera FPS        : " + zed.GetInitParameters().cameraFPS);

        // Print help control
        printHelp();

        sl.Mat zedImage = new sl.Mat();
        zedImage.Create(new Resolution((uint)zed.ImageWidth, (uint)zed.ImageHeight), MAT_TYPE.MAT_8U_C4);

        // Initialise camera setting
        switchCameraSettings();

        char key = ' ';

        RuntimeParameters rtParams = new RuntimeParameters();
        while (key != 'q')
        {
            // Check that a new image is successfully acquired
            returnedState = zed.Grab(ref rtParams);
            if (returnedState == ERROR_CODE.SUCCESS)
            {
                //Retrieve left image
                zed.RetrieveImage(zedImage, VIEW.LEFT);

                // Convert to cvMat
                OpenCvSharp.Mat cvImage = new OpenCvSharp.Mat(zedImage.GetHeight(), zedImage.GetWidth(), OpenCvSharp.MatType.CV_8UC4, zedImage.GetPtr());

                Cv2.ImShow(winName, cvImage);
            }
            else
            {
                Console.WriteLine("ERROR during capture");
                break;
            }

            key = (char)Cv2.WaitKey(10);
            // Change camera settings with keyboard
            updateCameraSettings(key, ref zed);
        }
    }

    static void updateCameraSettings(char key, ref Camera zed)
    {
        int current_value;

        // Keyboard shortcuts
        switch (key)
        {

            // Switch to the next camera parameter
            case 's':
                switchCameraSettings();
                current_value = zed.GetCameraSettings(camera_settings_);
                break;

            // Increase camera settings value ('+' key)
            case '+':
                current_value = zed.GetCameraSettings(camera_settings_);
                zed.SetCameraSettings(camera_settings_, current_value + step_camera_setting);
                Console.WriteLine(str_camera_settings + ": " + zed.GetCameraSettings(camera_settings_));
                break;

            // Decrease camera settings value ('-' key)
            case '-':
                current_value = zed.GetCameraSettings(camera_settings_);
                current_value = current_value > 0 ? current_value - step_camera_setting : 0;
                zed.SetCameraSettings(camera_settings_, current_value);
                Console.WriteLine(str_camera_settings + ": " + zed.GetCameraSettings(camera_settings_));
                break;

            //switch LED On :
            case 'l':
                led_on = !led_on;
                zed.SetCameraSettings(sl.VIDEO_SETTINGS.LED_STATUS, Convert.ToInt32(led_on));
                break;

            // Reset to default parameters
            case 'r':
                Console.WriteLine("Reset all settings to default\n");
                ResetCameraSettings(ref zed);
                break;

            case 'f':
                Console.WriteLine("reset AEC_AGC_ROI");
                zed.SetCameraSettings(VIDEO_SETTINGS.AEC_AGC, 1);
                break;

        }
    }

    static void switchCameraSettings()
    {
        camera_settings_ = (VIDEO_SETTINGS)((int)camera_settings_ + 1);

        // reset to 1st setting
        if (camera_settings_ == VIDEO_SETTINGS.LED_STATUS)
            camera_settings_ = VIDEO_SETTINGS.BRIGHTNESS;

        // increment if AEC_AGC_ROI since it using the overloaded function
        if (camera_settings_ == VIDEO_SETTINGS.AEC_AGC_ROI)
            camera_settings_ = (VIDEO_SETTINGS)((int)camera_settings_ + 1);

        // select the right step
        step_camera_setting = (camera_settings_ == VIDEO_SETTINGS.WHITEBALANCE) ? 100 : 1;

        // get the name of the selected SETTING
        str_camera_settings = camera_settings_.ToString();

        Console.WriteLine("Switch to camera settings : "  + str_camera_settings);
    }

    static private void parseArgs(string[] args, ref sl.InitParameters param)
    {
        if (args.Length > 0 && args[0].IndexOf(".svo") != -1)
        {
            // SVO input mode
            param.inputType = INPUT_TYPE.SVO;
            param.pathSVO = args[0];
            Console.WriteLine("[Sample] Using SVO File input: " + args[0]);
        }
        else if (args.Length > 0 && args[0].IndexOf(".svo") == -1)
        {
            IPAddress ip;
            string arg = args[0];
            if (IPAddress.TryParse(arg, out ip))
            {
                // Stream input mode - IP + port
                param.inputType = INPUT_TYPE.STREAM;
                param.ipStream = ip.ToString();
                Console.WriteLine("[Sample] Using Stream input, IP : " + ip);
            }
            else if (args[0].IndexOf("HD2K") != -1)
            {
                param.resolution = sl.RESOLUTION.HD2K;
                Console.WriteLine("[Sample] Using Camera in resolution HD2K");
            }
            else if (args[0].IndexOf("HD1080") != -1)
            {
                param.resolution = sl.RESOLUTION.HD1080;
                Console.WriteLine("[Sample] Using Camera in resolution HD1080");
            }
            else if (args[0].IndexOf("HD720") != -1)
            {
                param.resolution = sl.RESOLUTION.HD720;
                Console.WriteLine("[Sample] Using Camera in resolution HD720");
            }
            else if (args[0].IndexOf("VGA") != -1)
            {
                param.resolution = sl.RESOLUTION.VGA;
                Console.WriteLine("[Sample] Using Camera in resolution VGA");
            }
        }
        else
        {
            //
        }
    }

    /**
    This function displays help
    **/
    static void printHelp()
    {
        Console.WriteLine("Camera controls hotkeys:");
        Console.WriteLine("* Increase camera settings value:  '+'");
        Console.WriteLine("* Decrease camera settings value:  '-'");
        Console.WriteLine("* Toggle camera settings:          's'");
        Console.WriteLine("* Toggle camera LED:               'l' (lower L)");
        Console.WriteLine("* Reset all parameters:            'r'");
        Console.WriteLine("* Reset AEC_AGC:                  'f'");
        Console.WriteLine("* Exit :                           'q'");
        Console.WriteLine("");
    }

    /// <summary>
    /// Reset camera settings to default
    /// </summary>
    static void ResetCameraSettings(ref Camera zed)
    {

        zed.SetCameraSettings(sl.VIDEO_SETTINGS.BRIGHTNESS, sl.Camera.brightnessDefault);
        zed.SetCameraSettings(sl.VIDEO_SETTINGS.CONTRAST, sl.Camera.contrastDefault);
        zed.SetCameraSettings(sl.VIDEO_SETTINGS.HUE, sl.Camera.hueDefault);
        zed.SetCameraSettings(sl.VIDEO_SETTINGS.SATURATION, sl.Camera.saturationDefault);
        zed.SetCameraSettings(sl.VIDEO_SETTINGS.SHARPNESS, sl.Camera.sharpnessDefault);
        zed.SetCameraSettings(sl.VIDEO_SETTINGS.GAMMA, sl.Camera.gammaDefault);
        zed.SetCameraSettings(sl.VIDEO_SETTINGS.AUTO_WHITEBALANCE, 1);
        zed.SetCameraSettings(sl.VIDEO_SETTINGS.LED_STATUS, 1);
        zed.SetCameraSettings(sl.VIDEO_SETTINGS.AEC_AGC, 1);

    }
}
