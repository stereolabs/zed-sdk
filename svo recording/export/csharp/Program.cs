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

 /***********************************************************************
  ** This sample demonstrates how to read a SVO file 		    		  **
  ** and convert it into an AVI file (LEFT + RIGHT) or (LEFT + DEPTH)   **
  ************************************************************************/
  
using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Numerics;
using OpenCvSharp;
using sl;

class Program
{

    public enum APP_TYPE
    {
        LEFT_AND_RIGHT,
        LEFT_AND_DEPTH,
        LEFT_AND_DEPTH_16
    };

    static bool exit_app = false;

    [STAThread]
    static void Main(string[] args)
    {
        if (args.Length != 3)
        {
            Console.WriteLine("Usage: ");
            Console.WriteLine("    ZED_SVO_Export A B C ");
            Console.WriteLine("Please use the following parameters from the command line:");
            Console.WriteLine(" A - SVO file path (input) : \"path/to/file.svo\"");
            Console.WriteLine(" B - AVI file path (output) or image sequence folder(output) : \"path/to/output/file.avi\" or \"path/to/output/folder\"");
            Console.WriteLine(" C - Export mode:  0=Export LEFT+RIGHT AVI.");
            Console.WriteLine("                   1=Export LEFT+DEPTH_VIEW AVI.");
            Console.WriteLine("                   2=Export LEFT+RIGHT image sequence.");
            Console.WriteLine("                   3=Export LEFT+DEPTH_VIEW image sequence.");
            Console.WriteLine("                   4=Export LEFT+DEPTH_16Bit image sequence.");
            Console.WriteLine(" A and B need to end with '/' or '\\'\n\n");
            Console.WriteLine("Examples: \n");
            Console.WriteLine("  (AVI LEFT+RIGHT)   ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/file.avi\" 0");
            Console.WriteLine("  (AVI LEFT+DEPTH)   ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/file.avi\" 1");
            Console.WriteLine("  (SEQUENCE LEFT+RIGHT)   ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/folder\" 2");
            Console.WriteLine("  (SEQUENCE LEFT+DEPTH)   ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/folder\" 3");
            Console.WriteLine("  (SEQUENCE LEFT+DEPTH_16Bit)   ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/folder\" 4");

            Environment.Exit(-1);
        }
        string svoInputPath = args[0];
        string outputPath = args[1];
        bool outputAsVideo = true;
        APP_TYPE appType = APP_TYPE.LEFT_AND_RIGHT;

        if (args[2].Equals("1") || args[2].Equals("3"))
            appType = APP_TYPE.LEFT_AND_DEPTH;
        if (args[2].Equals("4"))
            appType = APP_TYPE.LEFT_AND_DEPTH_16;
        // Check if exporting to AVI or SEQUENCE
        if (!args[2].Equals("0") && !args[2].Equals("1"))
            outputAsVideo = false;

        if (!outputAsVideo && !Directory.Exists(outputPath))
        {
            Console.WriteLine("Input directory doesn't exist. Check permissions or create it. " + outputPath);
            Environment.Exit(-1);
        }
        if (!outputAsVideo && outputPath.Substring(outputPath.Length - 1) != "/" && outputPath.Substring(outputPath.Length - 1) != "\\")
        {
            Console.WriteLine("Error: output folder needs to end with '/' or '\\'." + outputPath);
            Environment.Exit(-1);
        }

        // Create ZED Camera
        Camera zed = new Camera(0);

        //Specify SVO path parameters
        InitParameters initParameters = new InitParameters()
        {
            inputType = INPUT_TYPE.SVO,
            pathSVO = svoInputPath,
            svoRealTimeMode = true,
            coordinateUnits = UNIT.MILLIMETER
        };

        ERROR_CODE zedOpenState = zed.Open(ref initParameters);
        if (zedOpenState != ERROR_CODE.SUCCESS)
        {
            Environment.Exit(-1);
        }

        Resolution imageSize = zed.GetCalibrationParameters().leftCam.resolution;

        sl.Mat leftImage = new sl.Mat();
        leftImage.Create(imageSize, MAT_TYPE.MAT_8U_C4);
        OpenCvSharp.Mat leftImageOCV = SLMat2CVMat(ref leftImage, MAT_TYPE.MAT_8U_C4);

        sl.Mat rightImage = new sl.Mat();
        rightImage.Create(imageSize, MAT_TYPE.MAT_8U_C4);
        OpenCvSharp.Mat rightImageOCV = SLMat2CVMat(ref rightImage, MAT_TYPE.MAT_8U_C4);

        sl.Mat depthImage = new sl.Mat();
        depthImage.Create(imageSize, MAT_TYPE.MAT_32F_C1);
        OpenCvSharp.Mat depthImageOCV = SLMat2CVMat(ref depthImage, MAT_TYPE.MAT_8U_C4);

        OpenCvSharp.Mat imageSideBySide = new OpenCvSharp.Mat();
        if (outputAsVideo)
        {
            imageSideBySide = new OpenCvSharp.Mat((int)imageSize.height, (int)imageSize.width * 2, OpenCvSharp.MatType.CV_8UC3);
        }

        OpenCvSharp.VideoWriter videoWriter = new OpenCvSharp.VideoWriter();

        //Create Video writter
        if (outputAsVideo)
        {
            int fourcc = OpenCvSharp.VideoWriter.FourCC('M', '4', 'S', '2'); // MPEG-4 part 2 codec

            int frameRate = Math.Max(zed.GetInitParameters().cameraFPS, 25); // Minimum write rate in OpenCV is 25
            Console.WriteLine(outputPath);
            videoWriter.Open(outputPath, fourcc, frameRate, new OpenCvSharp.Size((int)imageSize.width * 2, (int)imageSize.height));
            if (!videoWriter.IsOpened())
            {
                Console.WriteLine("Error: OpenCV video writer cannot be opened. Please check the .avi file path and write permissions.");
                zed.Close();
                Environment.Exit(-1);
            }
        }

        RuntimeParameters rtParams = new RuntimeParameters();
        rtParams.sensingMode = SENSING_MODE.FILL;

        // Start SVO conversion to AVI/SEQUENCE
        Console.WriteLine("Converting SVO... press Q to interupt conversion");

        int nbFrames = zed.GetSVONumberOfFrames();
        int svoPosition = 0;
        zed.SetSVOPosition(svoPosition);


        while (!exit_app)
        {
            exit_app = (System.Windows.Input.Keyboard.IsKeyDown(System.Windows.Input.Key.Q) == true);
            ERROR_CODE err = zed.Grab(ref rtParams);
            if (err == ERROR_CODE.SUCCESS)
            {
                svoPosition = zed.GetSVOPosition();

                // Retrieve SVO images
                zed.RetrieveImage(leftImage, VIEW.LEFT);

                switch (appType)
                {
                    case APP_TYPE.LEFT_AND_RIGHT:
                        zed.RetrieveImage(rightImage, VIEW.RIGHT);
                        break;
                    case APP_TYPE.LEFT_AND_DEPTH:
                        zed.RetrieveImage(rightImage, VIEW.DEPTH);
                        break;
                    case APP_TYPE.LEFT_AND_DEPTH_16:
                        zed.RetrieveMeasure(depthImage, MEASURE.DEPTH);
                        break;
                    default:
                        break;
                }

                if (outputAsVideo)
                {
                    // Convert SVO image from RGBA to RGB
                    Cv2.CvtColor(leftImageOCV, imageSideBySide[new OpenCvSharp.Rect(0,0, (int)imageSize.width, (int)imageSize.height)], ColorConversionCodes.BGRA2BGR);
                    Cv2.CvtColor(rightImageOCV, imageSideBySide[new OpenCvSharp.Rect((int)imageSize.width, 0, (int)imageSize.width, (int)imageSize.height)], ColorConversionCodes.BGRA2BGR);
                    // Write the RGB image in the video
                    videoWriter.Write(imageSideBySide);
                }
                else
                {
                    // Generate filenames
                    string filename1 = "";
                    filename1 = outputPath + "/left" + svoPosition + ".png";
                    string filename2 = "";
                    filename2 = outputPath + (appType == APP_TYPE.LEFT_AND_RIGHT ? "/right" : "/depth") +svoPosition + ".png";

                    // Save Left images
                    Cv2.ImWrite(filename1, leftImageOCV);

                    //Save depth
                    if (appType != APP_TYPE.LEFT_AND_DEPTH_16)
                        Cv2.ImWrite(filename2, rightImageOCV);
                    else
                    {
                        //Convert to 16 bit
                        OpenCvSharp.Mat depth16 = new OpenCvSharp.Mat();
                        depthImageOCV.ConvertTo(depth16, MatType.CV_16UC1);
                        Cv2.ImWrite(filename2, depth16);
                    }
                }

                // Display Progress
                ProgressBar((float)svoPosition / (float)nbFrames, 30);
            }
            else if (zed.GetSVOPosition() >= nbFrames - (zed.GetInitParameters().svoRealTimeMode ? 2 : 1))
            {
                Console.WriteLine("SVO end has been reached. Exiting now.");
                Environment.Exit(-1);
                exit_app = true;
            }
            else
            {
                Console.WriteLine("Grab Error : " + err);
                exit_app = true;
            }
        }
        if (outputAsVideo)
        {
            //Close the video writer
            videoWriter.Release();
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
