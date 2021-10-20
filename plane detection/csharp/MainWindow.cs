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
using System.Diagnostics;

namespace sl
{
    class MainWindow
    {
        GLViewer viewer;
        Camera zedCamera;
        RuntimeParameters runtimeParameters;
        POSITIONAL_TRACKING_STATE tracking_state; 

        // Positional tracking data
        Pose cam_pose;
        // Current left image
        Mat zedMat;
        // Detected plane
        PlaneData plane;
        // Plane mesh
        Vector3[] planeMeshVertices;
        int[] planeMeshTriangles;
        UserAction userAction;

        int timer = 0;
        int nbVertices = 0, nbTriangles = 0;

        ERROR_CODE findPlaneStatus;

        bool hasIMU = false;

        public MainWindow(string[] args)
        {
            // Set configuration parameters
            InitParameters init_params = new InitParameters();
            init_params.resolution = RESOLUTION.HD720;
            init_params.cameraFPS = 60;
            init_params.depthMode = DEPTH_MODE.ULTRA;
            init_params.coordinateUnits = UNIT.METER;
            init_params.coordinateSystem = COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP;
            init_params.sdkVerbose = true;

            parseArgs(args, ref init_params);
            // Open the camera
            zedCamera = new Camera(0);
            ERROR_CODE err = zedCamera.Open(ref init_params);

            if (err != ERROR_CODE.SUCCESS)
                Environment.Exit(-1);

            if (zedCamera.CameraModel != sl.MODEL.ZED2)
            {
                Console.WriteLine(" ERROR : Use ZED2 Camera only");
                return;
            }

            findPlaneStatus = ERROR_CODE.FAILURE;
            tracking_state = POSITIONAL_TRACKING_STATE.OFF;

            hasIMU = zedCamera.GetSensorsConfiguration().gyroscope_parameters.isAvailable;
            userAction = new UserAction();
            // Enable tracking
            PositionalTrackingParameters positionalTrackingParameters = new PositionalTrackingParameters();
            zedCamera.EnablePositionalTracking(ref positionalTrackingParameters);

            runtimeParameters = new RuntimeParameters();
            runtimeParameters.measure3DReferenceFrame = REFERENCE_FRAME.WORLD;
            // Create ZED Objects filled in the main loop
            zedMat = new Mat();
            cam_pose = new Pose();

            //Create mesh.
            planeMeshTriangles = new int[65000];
            planeMeshVertices = new Vector3[65000];
            plane = new PlaneData();
            int Height = zedCamera.ImageHeight;
            int Width = zedCamera.ImageWidth;

            Resolution res = new Resolution((uint)Width, (uint)Height);
            zedMat.Create(res, MAT_TYPE.MAT_8U_C4, MEM.CPU);

            // Create OpenGL Viewer
            viewer = new GLViewer(new Resolution((uint)Width, (uint)Height));

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
                nativeWindow.MouseDown += NativeWindow_MouseDown;
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
                        case KeyCode.Space:
                            userAction.pressSpace = true;
                            break;
                    }
                };


                int wnd_h = Screen.PrimaryScreen.Bounds.Height;
                int wnd_w = Screen.PrimaryScreen.Bounds.Width;

                int height = (int)(wnd_h * 0.9f);
                int width = (int)(wnd_w * 0.9f);

                if (width > zedCamera.ImageWidth && height > zedCamera.ImageHeight)
                {
                    width = zedCamera.ImageWidth;
                    height = zedCamera.ImageHeight;
                }

                nativeWindow.Create((int)(zedCamera.ImageWidth * 0.05f), (int)(zedCamera.ImageHeight * 0.05f), (uint)width, (uint)height, NativeWindowStyle.Resizeable);
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

        private void NativeWindow_MouseDown(object sender, NativeWindowMouseEventArgs e)
        {
            if (e.Buttons == MouseButton.Left)
            {
                userAction.hit = true;
                Vector2 screenPos = new Vector2((float)e.Location.X / (float)zedCamera.ImageWidth, ((float)zedCamera.ImageHeight - (float)e.Location.Y) / (float)zedCamera.ImageHeight);
                userAction.hitCoord = screenPos;
            }
        }

        // Init Window
        private void NativeWindow_ContextCreated(object sender, NativeWindowEventArgs e)
        {
            OpenGL.CoreUI.NativeWindow nativeWindow = (OpenGL.CoreUI.NativeWindow)sender;

            Gl.ReadBuffer(ReadBufferMode.Back);
            Gl.ClearColor(0.0f, 0.0f, 0.0f, 1.0f);

            Gl.Enable(EnableCap.Blend);
            Gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

            Gl.Enable(EnableCap.LineSmooth);
            Gl.Hint(HintTarget.LineSmoothHint, HintMode.Nicest);

            viewer.init(zedCamera.GetCalibrationParameters().leftCam);
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
                if (zedMat.IsInit())
                {
                    // Retrieve left image
                    err = zedCamera.RetrieveImage(zedMat, sl.VIEW.LEFT, sl.MEM.CPU);
                    // Update pose data (used for projection of the mesh over the current image)
                    tracking_state = zedCamera.GetPosition(ref cam_pose);

                    if (tracking_state == POSITIONAL_TRACKING_STATE.OK)
                    {
                        timer++;
                        if (userAction.hit)
                        {                      
                            Vector2 imageClick = new Vector2((float)userAction.hitCoord.X * (float)zedCamera.ImageWidth, (float)userAction.hitCoord.Y * (float)zedCamera.ImageHeight);
                            findPlaneStatus = zedCamera.findPlaneAtHit(ref plane, imageClick);
                            if (findPlaneStatus == ERROR_CODE.SUCCESS)
                            {
                                zedCamera.convertHitPlaneToMesh(planeMeshVertices, planeMeshTriangles, out nbVertices, out nbTriangles);
                            }
                            userAction.clear();
                        }
                        //if 500ms have spend since last request (for 60fps)
                        if (timer % 30 == 0 && userAction.pressSpace)
                        {
                            // Update pose data (used for projection of the mesh over the current image)
                            Quaternion priorQuat = Quaternion.Identity;
                            Vector3 priorTrans = Vector3.Zero;
                            findPlaneStatus = zedCamera.findFloorPlane(ref plane, out float playerHeight, priorQuat, priorTrans);

                            if (findPlaneStatus == ERROR_CODE.SUCCESS)
                            {
                               zedCamera.convertFloorPlaneToMesh(planeMeshVertices, planeMeshTriangles, out nbVertices, out nbTriangles);
                            }
                            userAction.clear();
                        }
                    }
                    if (findPlaneStatus == ERROR_CODE.SUCCESS)
                    {
                        viewer.updateMesh(planeMeshVertices, planeMeshTriangles, nbVertices, nbTriangles, plane.Type, plane.Bounds, userAction);
                    }
                    viewer.updateImageAndState(zedMat, cam_pose, tracking_state);
                    viewer.render();
                }
            }
        }

        private void close()
        {
            zedCamera.DisablePositionalTracking();
            zedCamera.Close();
            viewer.exit();
        }

        private void parseArgs(string[] args , ref sl.InitParameters param)
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
}
