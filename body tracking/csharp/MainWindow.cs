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

/*****************************************************************************************
 ** This sample demonstrates how to detect human bodies and retrieves their 3D position **
 **         with the ZED SDK and display the result in an OpenGL window.                **
 *****************************************************************************************/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using System.Net;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenGL;
using OpenGL.CoreUI;
using OpenCvSharp;

namespace sl
{
    class MainWindow
    {
        Resolution pcRes;
        Resolution displayRes;
        GLViewer viewer;
        Camera zedCamera;
        ObjectDetectionRuntimeParameters obj_runtime_parameters;
        RuntimeParameters runtimeParameters;
        sl.Mat pointCloud;
        sl.Mat imageLeft;
        OpenCvSharp.Mat imageLeftOcv;
        Objects objects;
        sl.float2 imgScale;
        sl.Pose camPose;
        string window_name;
        int key;
        bool isTrackingON = false;
        bool isPlayback = false;

        public MainWindow(string[] args)
        {
            // Set configuration parameters
            InitParameters init_params = new InitParameters();
            init_params.resolution = RESOLUTION.HD1080;
            init_params.cameraFPS = 30;
            init_params.depthMode = DEPTH_MODE.ULTRA;
            init_params.coordinateUnits = UNIT.METER;
            init_params.coordinateSystem = COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP;

            parseArgs(args, ref init_params);
            // Open the camera
            zedCamera = new Camera(0);
            ERROR_CODE err = zedCamera.Open(ref init_params);

            if (err != ERROR_CODE.SUCCESS)
                Environment.Exit(-1);

            if (!(zedCamera.CameraModel == sl.MODEL.ZED2 || zedCamera.CameraModel == sl.MODEL.ZED2i))
            {
                Console.WriteLine(" ERROR : Use ZED2/ZED2i Camera only");
                return;
            }

            // Enable tracking (mandatory for object detection)
            PositionalTrackingParameters positionalTrackingParameters = new PositionalTrackingParameters();
            zedCamera.EnablePositionalTracking(ref positionalTrackingParameters);

            runtimeParameters = new RuntimeParameters();

            // Enable the Objects detection module
            ObjectDetectionParameters obj_det_params = new ObjectDetectionParameters();
            obj_det_params.enableObjectTracking = true; // the object detection will track objects across multiple images, instead of an image-by-image basis
            isTrackingON = obj_det_params.enableObjectTracking;
            obj_det_params.enable2DMask = false;
            obj_det_params.enableBodyFitting = true; // smooth skeletons moves
            obj_det_params.imageSync = true; // the object detection is synchronized to the image
            obj_det_params.detectionModel = sl.DETECTION_MODEL.HUMAN_BODY_ACCURATE;

            zedCamera.EnableObjectDetection(ref obj_det_params);

            // Create ZED Objects filled in the main loop
            camPose = new sl.Pose();
            objects = new Objects();
            int Height = zedCamera.ImageHeight;
            int Width = zedCamera.ImageWidth;

            imageLeft = new Mat();
            displayRes = new Resolution(Math.Min((uint)Width, 1280), Math.Min((uint)Height, 720));
            imgScale = new sl.float2((int)displayRes.width / (float)Width, (int)displayRes.height / (float)Height);
            imageLeft.Create(displayRes, MAT_TYPE.MAT_8U_C4, MEM.CPU);

            imageLeftOcv = new OpenCvSharp.Mat((int)displayRes.height, (int)displayRes.width, OpenCvSharp.MatType.CV_8UC4, imageLeft.GetPtr());

            pointCloud = new sl.Mat();
            pcRes = new Resolution(Math.Min((uint)Width, 720), Math.Min((uint)Height, 404));
            pointCloud.Create(pcRes, MAT_TYPE.MAT_32F_C4, MEM.CPU);

            // Create OpenGL Viewer
            viewer = new GLViewer(new Resolution((uint)Width, (uint)Height));

            // Configure object detection runtime parameters
            obj_runtime_parameters = new ObjectDetectionRuntimeParameters();
            obj_runtime_parameters.detectionConfidenceThreshold = 40;

            window_name = "ZED| 2D View";
            Cv2.NamedWindow(window_name, WindowMode.Normal);// Create Window

            // Create OpenGL window
            CreateWindow();
        }

        // Create Window
        public void CreateWindow()
        {
            using (OpenGL.CoreUI.NativeWindow nativeWindow = OpenGL.CoreUI.NativeWindow.Create())
            {
                nativeWindow.ContextCreated += NativeWindow_ContextCreated;
                nativeWindow.Render += NativeWindow_Render;
                nativeWindow.MouseMove += NativeWindow_MouseEvent;
                nativeWindow.Resize += NativeWindow_Resize;
                nativeWindow.KeyDown += (object obj, NativeWindowKeyEventArgs e) =>
                {
                    switch (e.Key)
                    {
                        case KeyCode.Escape:
                            close();
                            nativeWindow.Stop();
                            break;

                        case KeyCode.F:
                            nativeWindow.Fullscreen = !nativeWindow.Fullscreen;
                            break;
                    }

                    viewer.keyEventFunction(e);
                };
                nativeWindow.CursorVisible = false;
                nativeWindow.MultisampleBits = 4;

                int wnd_h = Screen.PrimaryScreen.Bounds.Height;
                int wnd_w = Screen.PrimaryScreen.Bounds.Width;

                int height = (int)(wnd_h * 0.9f);
                int width = (int)(wnd_w * 0.9f);

                if (width > zedCamera.ImageWidth && height > zedCamera.ImageHeight)
                {
                    width = zedCamera.ImageWidth;
                    height = zedCamera.ImageHeight;
                }

                nativeWindow.Create((int)(zedCamera.ImageWidth * 0.05f), (int)(zedCamera.ImageHeight * 0.05f), 1200, 700, NativeWindowStyle.Resizeable);
                nativeWindow.Show();
                nativeWindow.Run();
            }
        }

        private void NativeWindow_Resize(object sender, EventArgs e)
        {
            OpenGL.CoreUI.NativeWindow nativeWindow = (OpenGL.CoreUI.NativeWindow)sender;

            viewer.resizeCallback((int)nativeWindow.Width, (int)nativeWindow.Height);
        }

        private void NativeWindow_MouseEvent(object sender, NativeWindowMouseEventArgs e)
        {
            viewer.mouseEventFunction(e);
            viewer.computeMouseMotion(e.Location.X, e.Location.Y);
        }

        // Init Window
        private void NativeWindow_ContextCreated(object sender, NativeWindowEventArgs e)
        {
            OpenGL.CoreUI.NativeWindow nativeWindow = (OpenGL.CoreUI.NativeWindow)sender;

            Gl.ReadBuffer(ReadBufferMode.Back);
            Gl.ClearColor(0.2f, 0.19f, 0.2f, 1.0f);

            Gl.Enable(EnableCap.Blend);
            Gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

            Gl.Enable(EnableCap.LineSmooth);
            Gl.Hint(HintTarget.LineSmoothHint, HintMode.Nicest);

            viewer.init(zedCamera.GetCalibrationParameters().leftCam, isTrackingON);
        }

        // Render loop
        private void NativeWindow_Render(object sender, NativeWindowEventArgs e)
        {
            OpenGL.CoreUI.NativeWindow nativeWindow = (OpenGL.CoreUI.NativeWindow)sender;
            Gl.Viewport(0, 0, (int)nativeWindow.Width, (int)nativeWindow.Height);
            Gl.Clear(ClearBufferMask.ColorBufferBit);

            ERROR_CODE err = ERROR_CODE.FAILURE;
            if (viewer.isAvailable() && zedCamera.Grab(ref runtimeParameters) == ERROR_CODE.SUCCESS)
            {
                if (imageLeft.IsInit())
                {
                    // Retrieve left image
                    zedCamera.RetrieveMeasure(pointCloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, pcRes);
                    zedCamera.RetrieveImage(imageLeft, sl.VIEW.LEFT, sl.MEM.CPU, displayRes);
                    zedCamera.GetPosition(ref camPose, REFERENCE_FRAME.WORLD);

                    // Retrieve Objects
                    zedCamera.RetrieveObjects(ref objects, ref obj_runtime_parameters);

                    TrackingViewer.render_2D(ref imageLeftOcv, imgScale, ref objects, isTrackingON);

                    //Update GL View
                    viewer.update(pointCloud, objects, camPose);
                    viewer.render();

                    if (isPlayback && zedCamera.GetSVOPosition() == zedCamera.GetSVONumberOfFrames()) return;

                    Cv2.ImShow(window_name, imageLeftOcv);
                }
            }
        }

        private void close()
        {
            zedCamera.DisablePositionalTracking();
            zedCamera.DisableObjectDetection();
            zedCamera.Close();
            viewer.exit();
            pointCloud.Free();
            imageLeft.Free();
        }

        private void parseArgs(string[] args , ref sl.InitParameters param)
        {
            if (args.Length > 0 && args[0].IndexOf(".svo") != -1)
            {
                // SVO input mode
                param.inputType = INPUT_TYPE.SVO;
                param.pathSVO = args[0];
                isPlayback = true;
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
}
