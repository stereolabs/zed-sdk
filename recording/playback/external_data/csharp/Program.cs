///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2024, STEREOLABS.
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

/************************************************************
** This sample demonstrates how to read a SVO video file. **
** We use OpenCV to display the video.					   **
*************************************************************/

using System;
using sl;
using System.Collections.Generic;
using System.Net;

class Program
{

    static void Main(string[] args)
    {
        if (args.Length != 1)
        {
            Console.WriteLine("Usage: ");
            Console.WriteLine("    ZED_SVO_Playback <SVO_file> ");
            Console.WriteLine("* *SVO file is mandatory in the application * *");

            Environment.Exit(-1);
        }

        // Create ZED Camera
        Camera zed = new Camera(0);

        //Specify SVO path parameters
        InitParameters initParameters = new InitParameters()
        {
            inputType = INPUT_TYPE.SVO,
            pathSVO = args[0],
            svoRealTimeMode = false,
            depthMode = DEPTH_MODE.PERFORMANCE,
            sdkVerbose = 1,
        };

        parseArgs(args, ref initParameters);

        ERROR_CODE state = zed.Open(ref initParameters);
        if (state != ERROR_CODE.SUCCESS)
        {
            Environment.Exit(-1);
        }

        char key = ' ';
        RuntimeParameters rtParams = new RuntimeParameters();

        string s = "";

        List<string> keys = zed.GetSVODataKeys();

        foreach (var piece in keys)
        {
            s += piece + " ;";
        }
        Console.WriteLine("Channels that are in the SVO: " + s);

        ulong last_timestamp_ns = 0;

        List<SVOData> data = new List<SVOData>();
        zed.RetrieveSVOData("TEST", ref data, 0, 0);

        foreach(var d in data)
        {
            Console.WriteLine(d.GetContent());
        }

        Console.WriteLine("############\n");

        while (key != 'q')
        {
            state = zed.Grab(ref rtParams);
            if (state == ERROR_CODE.SUCCESS)
            {
                List<SVOData> svoData = new List<SVOData>();
                Console.WriteLine("Reading between " + last_timestamp_ns + " and " + zed.GetCameraTimeStamp());
                state = zed.RetrieveSVOData("TEST", ref svoData, last_timestamp_ns, zed.GetCameraTimeStamp());

                if (state == ERROR_CODE.SUCCESS)
                {
                    foreach (var d in svoData)
                    {
                        Console.WriteLine(zed.GetCameraTimeStamp() + " // " + d.GetContent());
                    }
                }

                last_timestamp_ns = zed.GetCameraTimeStamp();
            }
            else if (state == ERROR_CODE.END_OF_SVO_FILE_REACHED)
            {
                Console.WriteLine("SVO end has been reached. Looping back to 0");
                zed.SetSVOPosition(0);
            }
            else
            {
                Console.WriteLine("Grab Error : " + state);
                break;
            }
        }
        zed.Close();
    }

    static void parseArgs(string[] args, ref sl.InitParameters param)
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
}