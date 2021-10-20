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

/*********************************************************************************
 ** This sample demonstrates how to capture 3D point cloud and detected objects **
 **      with the ZED SDK and display the result in an OpenGL window. 	        **
 *********************************************************************************/

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
using sl;

class MainWindow
{
    // Flag to enable/disable the batch option in Object Detection module
    // Batching system allows to reconstruct trajectories from the object detection module by adding Re-Identification / Appareance matching.
    // For example, if an object is not seen during some time, it can be re-ID to a previous ID if the matching score is high enough
    // Use with caution if image retention is activated (See BatchSystemhandler.cs) :
    //   --> Images will only appears if a object is detected since the batching system is based on OD detection.
    static bool USE_BATCHING = false;

    bool isTrackingON = false;
    bool isPlayback = false;
    GLViewer viewer;
    Camera zedCamera;
    ObjectDetectionRuntimeParameters obj_runtime_parameters;
    RuntimeParameters runtimeParameters;
    BatchParameters batchParameters;
    sl.Mat pointCloud;
    sl.Mat imageLeft;
    OpenCvSharp.Mat imageRenderLeft;
    OpenCvSharp.Mat imageTrackOcv;
    OpenCvSharp.Mat imageLeftOcv;
    OpenCvSharp.Mat globalImage;
    sl.float2 imgScale;
    Pose camWorldPose;
    Pose camCameraPose;
    Resolution pcRes;
    Resolution displayRes;
    Objects objects;
    float maxDepthDistance;
    TrackingViewer trackViewGenerator;
    BatchSystemHandler batchHandler;
    int detection_confidence;
    string window_name;
    char key = ' ';

    public MainWindow(string[] args)
    {
        // Set configuration parameters
        InitParameters init_params = new InitParameters();
        init_params.resolution = RESOLUTION.HD1080;
        init_params.depthMode = DEPTH_MODE.ULTRA;
        init_params.coordinateUnits = UNIT.METER;
        init_params.coordinateSystem = COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP;
        init_params.depthMaximumDistance = 10f;
        init_params.cameraDisableSelfCalib = true;

        maxDepthDistance = init_params.depthMaximumDistance;
        parseArgs(args, ref init_params);
        // Open the camera
        zedCamera = new Camera(0);
        ERROR_CODE err = zedCamera.Open(ref init_params);

        if (err != ERROR_CODE.SUCCESS)
            Environment.Exit(-1);

        if (zedCamera.CameraModel == sl.MODEL.ZED)
        {
            Console.WriteLine(" ERROR : not compatible camera model");
            return;
        }

        // Enable tracking (mandatory for object detection)
        PositionalTrackingParameters trackingParams = new PositionalTrackingParameters();
        zedCamera.EnablePositionalTracking(ref trackingParams);

        runtimeParameters = new RuntimeParameters();

        // Enable the Objects detection module
        ObjectDetectionParameters obj_det_params = new ObjectDetectionParameters();
        obj_det_params.enableObjectTracking = true; // the object detection will track objects across multiple images, instead of an image-by-image basis
        isTrackingON = obj_det_params.enableObjectTracking;
        obj_det_params.enable2DMask = false;
        obj_det_params.imageSync = true; // the object detection is synchronized to the image
        obj_det_params.detectionModel = sl.DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE;

        if (USE_BATCHING)
        {
            batchParameters = new BatchParameters();
            batchParameters.latency = 2.0f;
            batchParameters.enable = true;
            batchHandler = new BatchSystemHandler((int)batchParameters.latency * 2);
            obj_det_params.batchParameters = batchParameters;
        }

        zedCamera.EnableObjectDetection(ref obj_det_params);

        // Configure object detection runtime parameters
        obj_runtime_parameters = new ObjectDetectionRuntimeParameters();
        detection_confidence = 60;
        obj_runtime_parameters.detectionConfidenceThreshold = detection_confidence;
        obj_runtime_parameters.objectClassFilter = new int[(int)OBJECT_CLASS.LAST];
        obj_runtime_parameters.objectClassFilter[(int)sl.OBJECT_CLASS.PERSON] = Convert.ToInt32(true);
        //obj_runtime_parameters.objectClassFilter[(int)sl.OBJECT_CLASS.VEHICLE] = Convert.ToInt32(true);
        // To set a specific threshold
        obj_runtime_parameters.objectConfidenceThreshold = new int[(int)OBJECT_CLASS.LAST];
        obj_runtime_parameters.objectConfidenceThreshold[(int)sl.OBJECT_CLASS.PERSON] = detection_confidence;
        //obj_runtime_parameters.object_confidence_threshold[(int)sl.OBJECT_CLASS.VEHICLE] = detection_confidence;

        // Create ZED Objects filled in the main loop
        objects = new Objects();
        imageLeft = new sl.Mat();
        int Height = zedCamera.ImageHeight;
        int Width = zedCamera.ImageWidth;

        displayRes = new Resolution(Math.Min((uint)Width, 1280), Math.Min((uint)Height, 720));
        Resolution tracksRes = new Resolution(400, (uint)displayRes.height);

        // create a global image to store both image and tracks view
        globalImage = new OpenCvSharp.Mat((int)displayRes.height, (int)displayRes.width + (int)tracksRes.width, OpenCvSharp.MatType.CV_8UC4);
        // retrieve ref on image part
        imageLeftOcv = new OpenCvSharp.Mat(globalImage, new OpenCvSharp.Rect(0, 0, (int)displayRes.width, (int)displayRes.height));
        // retrieve ref on tracks part
        imageTrackOcv = new OpenCvSharp.Mat(globalImage, new OpenCvSharp.Rect((int)displayRes.width, 0, (int)tracksRes.width, (int)tracksRes.height));
        // init an sl::Mat from the ocv image ref (which is in fact the memory of global_image)
        imageLeft.Create(displayRes, MAT_TYPE.MAT_8U_C4, MEM.CPU);
        imageRenderLeft = new OpenCvSharp.Mat((int)displayRes.height, (int)displayRes.width, OpenCvSharp.MatType.CV_8UC4, imageLeft.GetPtr());
        imgScale = new sl.float2((int)displayRes.width / (float)Width, (int)displayRes.height / (float)Height);

        // Create OpenGL Viewer
        viewer = new GLViewer();

        camWorldPose = new Pose();
        camCameraPose = new Pose();
        pointCloud = new sl.Mat();
        pcRes = new Resolution(Math.Min((uint)Width, 720), Math.Min((uint)Height, 404));
        pointCloud.Create(pcRes, MAT_TYPE.MAT_32F_C4, MEM.CPU);

        // 2D tracks
        trackViewGenerator = new TrackingViewer(tracksRes, (int)zedCamera.GetCameraFPS(), maxDepthDistance,3);
        trackViewGenerator.setCameraCalibration(zedCamera.GetCalibrationParameters());

        window_name = "ZED| 2D View and Birds view";
        Cv2.NamedWindow(window_name, WindowMode.Normal);// Create Window
        Cv2.CreateTrackbar("Confidence", window_name, ref detection_confidence, 100);

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

            //nativeWindow.MultisampleBits = 4;

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
            try
            {
                nativeWindow.Run();
            }
            catch (Exception e)
            {
                Console.WriteLine("Mouse wheel is broken in the current OPENGL .NET VERSION. Please do not use it.");
            }
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

        Gl.Enable(EnableCap.DepthTest);

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
        Gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

        ERROR_CODE err = ERROR_CODE.FAILURE;
        if (viewer.isAvailable() && zedCamera.Grab(ref runtimeParameters) == ERROR_CODE.SUCCESS)
        {
            foreach( var it in obj_runtime_parameters.objectClassFilter)
            {
                    obj_runtime_parameters.objectConfidenceThreshold[it] = detection_confidence;
            }

            // Retrieve Objects
            err = zedCamera.RetrieveObjects(ref objects, ref obj_runtime_parameters);

            if (err == ERROR_CODE.SUCCESS && objects.isNew != 0)
            {
                // Retrieve left image
                zedCamera.RetrieveMeasure(pointCloud, MEASURE.XYZRGBA, MEM.CPU, pcRes);
                zedCamera.GetPosition(ref camWorldPose, REFERENCE_FRAME.WORLD);
                zedCamera.GetPosition(ref camCameraPose, REFERENCE_FRAME.CAMERA);
                zedCamera.RetrieveImage(imageLeft, VIEW.LEFT, MEM.CPU, displayRes);

                bool update_render_view = true;
                bool update_3d_view = true;
                bool update_tracking_view = true;
                int nbBatches = 0;

                if (USE_BATCHING)
                {
                    List<ObjectsBatch> objectsBatch = new List<ObjectsBatch>();
                    zedCamera.UpdateObjectsBatch(out nbBatches);
                    for (int i = 0; i < nbBatches; i++)
                    {
                        ObjectsBatch obj_batch = new ObjectsBatch();
                        zedCamera.GetObjectsBatch(i, ref obj_batch);
                        objectsBatch.Add(obj_batch);
                    }
                    batchHandler.push(camCameraPose, camWorldPose, imageLeft, pointCloud, ref objectsBatch);
                    batchHandler.pop(ref camCameraPose, ref camWorldPose, ref imageLeft, ref pointCloud, ref objects);
                    update_render_view = BatchSystemHandler.WITH_IMAGE_RETENTION ? Convert.ToBoolean(objects.isNew) : true;
                    update_3d_view = BatchSystemHandler.WITH_IMAGE_RETENTION ? Convert.ToBoolean(objects.isNew) : true;
                }
                if (update_render_view)
                {
                    imageRenderLeft.CopyTo(imageLeftOcv);
                    TrackingViewer.render_2D(ref imageLeftOcv, imgScale, ref objects, true, isTrackingON);
                }
                if (update_3d_view)
                {
                    //Update GL View
                    viewer.update(pointCloud, objects, camWorldPose);
                    viewer.render();
                }
                if (update_tracking_view)
                    trackViewGenerator.generate_view(ref objects, camCameraPose, ref imageTrackOcv, Convert.ToBoolean(objects.isTracked));
            }

            if (isPlayback && zedCamera.GetSVOPosition() == zedCamera.GetSVONumberOfFrames()) return;

            Cv2.ImShow(window_name, globalImage);

        }
    }

    private void close()
    {
        if (USE_BATCHING)
        {
            batchHandler.clear();
        }
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

