///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, STEREOLABS.
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

/****************************************************************************************
** This sample shows how to record video in Stereolabs SVO format.					   **
** SVO video files can be played with the ZED API and used with its different modules  **
*****************************************************************************************/

using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Numerics;
using sl;

class Program
{
    [STAThread]
    static void Main(string[] args)
    {

        if (args.Length < 1)
        {
            Console.WriteLine("Usage : Only the path of the output SVO file should be passed as argument.");
            Environment.Exit(-1);
        }
        // Create ZED Camera
        Camera zed = new Camera(0);

        Console.CancelKeyPress += delegate {
            Console.WriteLine("close");
            zed.DisableRecording();
            zed.Close();
        };

        //Specify SVO path parameters
        InitParameters initParameters = new InitParameters()
        {
            resolution = RESOLUTION.HD2K,
            depthMode = DEPTH_MODE.NONE,
        };

        ERROR_CODE state = zed.Open(ref initParameters);
        if (state != ERROR_CODE.SUCCESS)
        {
            Environment.Exit(-1);
        }

        string pathOutput = args[0];

        RecordingParameters recordingParams = new RecordingParameters(pathOutput, SVO_COMPRESSION_MODE.H264_BASED, 8000, 15, false);
        state = zed.EnableRecording(recordingParams);
        if (state != ERROR_CODE.SUCCESS)
        {
            zed.Close();
            Environment.Exit(-1);
        }

        // Start recording SVO, stop with Q
        Console.WriteLine("SVO is recording, press Q to stop");
        int framesRecorded = 0;
        
        RuntimeParameters rtParams = new RuntimeParameters();

        while (true)
        {
            if (zed.Grab(ref rtParams) == ERROR_CODE.SUCCESS){
                // Each new frame is added to the SVO file
                framesRecorded++;
                Console.WriteLine("Frame count: " + framesRecorded);
            }

            bool State = (System.Windows.Input.Keyboard.IsKeyDown(System.Windows.Input.Key.Q) == true);
            if (State) break;
        }

        // Stop recording
        zed.DisableRecording();
        zed.Close();
    }
}
