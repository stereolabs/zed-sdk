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

/************************************************************
** This sample demonstrates how to read a SVO video file. **
** We use OpenCV to display the video.					   **
*************************************************************/

using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Numerics;
using OpenCvSharp;
using sl;

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
            svoRealTimeMode = true,
            depthMode = DEPTH_MODE.PERFORMANCE
        };

        ERROR_CODE state = zed.Open(ref initParameters);
        if (state != ERROR_CODE.SUCCESS)
        {
            Environment.Exit(-1);
        }

        Resolution resolution = zed.GetCalibrationParameters().leftCam.resolution;
        // Define OpenCV window size (resize to max 720/404)
        Resolution lowResolution = new Resolution((uint)Math.Min(720, (int)resolution.width) * 2, (uint)Math.Min(404, (int)resolution.height));

        sl.Mat svoImage = new sl.Mat();
        svoImage.Create(lowResolution, MAT_TYPE.MAT_8U_C4);
        OpenCvSharp.Mat svoImageOCV = SLMat2CVMat(ref svoImage, MAT_TYPE.MAT_8U_C4);

        //Setup key, images, times
        char key = ' ';
        Console.WriteLine("Press 's' to save SVO image as PNG");
        Console.WriteLine("Press 'f' to jump forward in the video");
        Console.WriteLine("Press 'b' to jump backard in the video");
        Console.WriteLine("Press 'q' to exit...");

        int svoFrameRate = zed.GetInitParameters().cameraFPS;
        int nbFrames = zed.GetSVONumberOfFrames();
        Console.WriteLine("[INFO] SVO contains " + nbFrames + " frames");

        RuntimeParameters rtParams = new RuntimeParameters();
        // Start SVO Playback

        while (key != 'q')
        {
            state = zed.Grab(ref rtParams);
            if (state == ERROR_CODE.SUCCESS)
            {
                //Get the side by side image
                zed.RetrieveImage(svoImage, VIEW.SIDE_BY_SIDE, MEM.CPU, lowResolution);
                int svoPosition = zed.GetSVOPosition();

                //Display the frame
                Cv2.ImShow("View", svoImageOCV);
                key = (char)Cv2.WaitKey(10);

                switch (key)
                {
                    case 's':
                        svoImage.Write("capture" + svoPosition + ".png");
                        break;
                    case 'f':
                        zed.SetSVOPosition(svoPosition + svoFrameRate);
                        break;
                    case 'b':
                        zed.SetSVOPosition(svoPosition - svoFrameRate);
                        break;
                }
                ProgressBar((float)svoPosition / (float)nbFrames, 30);
            }
            else if (zed.GetSVOPosition() >= nbFrames - (zed.GetInitParameters().svoRealTimeMode ? 2 : 1))
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


    /// <summary>
    ///  Creates an OpenCV version of a ZED Mat. 
    /// </summary>
    /// <param name="zedmat">Source ZED Mat.</param>
    /// <param name="zedmattype">Type of ZED Mat - data type and channel number.
    /// <returns></returns>
    private static OpenCvSharp.Mat SLMat2CVMat(ref sl.Mat zedmat, MAT_TYPE zedmattype)
    {
        int cvmattype = SLMatType2CVMatType(zedmattype);
        OpenCvSharp.Mat cvmat = new OpenCvSharp.Mat(zedmat.GetHeight(), zedmat.GetWidth(), cvmattype, zedmat.GetPtr());

        return cvmat;
    }

    /// <summary>
    /// Returns the OpenCV type that corresponds to a given ZED Mat type. 
    /// </summary>
    private static int SLMatType2CVMatType(MAT_TYPE zedmattype)
    {
        switch (zedmattype)
        {
            case sl.MAT_TYPE.MAT_32F_C1:
                return OpenCvSharp.MatType.CV_32FC1;
            case sl.MAT_TYPE.MAT_32F_C2:
                return OpenCvSharp.MatType.CV_32FC2;
            case sl.MAT_TYPE.MAT_32F_C3:
                return OpenCvSharp.MatType.CV_32FC3;
            case sl.MAT_TYPE.MAT_32F_C4:
                return OpenCvSharp.MatType.CV_32FC4;
            case sl.MAT_TYPE.MAT_8U_C1:
                return OpenCvSharp.MatType.CV_8UC1;
            case sl.MAT_TYPE.MAT_8U_C2:
                return OpenCvSharp.MatType.CV_8UC2;
            case sl.MAT_TYPE.MAT_8U_C3:
                return OpenCvSharp.MatType.CV_8UC3;
            case sl.MAT_TYPE.MAT_8U_C4:
                return OpenCvSharp.MatType.CV_8UC4;
            default:
                return -1;
        }
    }

    // Display progress bar
    static void ProgressBar(float ratio, uint w)
    {
        uint c = (uint)(ratio * w);
        for (uint x = 0; x < c; x++) Console.Write("=");
        for (uint x = c; x < w; x++) Console.Write(" ");
        Console.Write((int)(ratio * 100) + "% ");
        Console.Write("\r");
    }
}