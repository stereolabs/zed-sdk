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

/*****************************************************************************************
 ** This sample demonstrates how to detect human bodies and retrieves their 3D position **
 **         with the ZED SDK and display the result in an OpenGL window.                **
 *****************************************************************************************/

using System;
using System.Collections.Generic;
using System.Net;
using System.Numerics;
using System.Text;
using System.Windows.Forms;
using OpenGL;
using OpenGL.CoreUI;

using sl;

class MainWindow
{
    GLViewer viewer;
    Fusion fusion;
    BodyTrackingFusionParameters bodyTrackingFusionParameters;
    BodyTrackingFusionRuntimeParameters bodyTrackingFusionRuntimeParameters;
    Bodies bodies;
    List<sl.CameraIdentifier> cameras;
    sl.Resolution pointCloudResolution;
    Dictionary<ulong, sl.Mat> pointClouds;
    Dictionary<ulong, sl.Bodies> camRawData;
    public MainWindow(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("Requires a configuration file as argument");
            return;
        }

        const sl.COORDINATE_SYSTEM COORDINATE_SYSTEM = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP;
        const sl.UNIT UNIT = sl.UNIT.METER;

        // Read json file containing the configuration of your multicamera setup. 
        List<FusionConfiguration> fusionConfigurations = sl.Fusion.ReadConfigurationFile(new StringBuilder(args[0]), COORDINATE_SYSTEM, UNIT);

        if (fusionConfigurations.Count == 0 )
        {
            Console.WriteLine("Empty configuration file");
            return;
        }

        List<ClientPublisher> clients =  new List<ClientPublisher>();
        int id = 0;
        // Check if the ZED camera should run within the same process or if they are running on the edge.
        foreach (FusionConfiguration config in fusionConfigurations)
        {
            if (config.commParam.communicationType == sl.COMM_TYPE.INTRA_PROCESS)
            {
                Console.WriteLine("Try to open ZED " + config.serialNumber);
                var client = new ClientPublisher(id);
                bool state = client.Open(config.inputType);
                clients.Add(client);

                if (!state)
                {
                    Console.WriteLine("Error while opening ZED " + config.serialNumber);
                    continue;
                }

                Console.WriteLine("ready ! ");
                id++;
            }
        }

        // Starts Camera threads
        foreach(var client in clients) client.Start();

        sl.InitFusionParameters initFusionParameters = new sl.InitFusionParameters();
        initFusionParameters.coordinateUnits = UNIT;
        initFusionParameters.coordinateSystem = COORDINATE_SYSTEM;

        fusion = new sl.Fusion();
        FUSION_ERROR_CODE err = fusion.Init(ref initFusionParameters);

        if (err != sl.FUSION_ERROR_CODE.SUCCESS)
        {
            Console.WriteLine("Error while initializing the fusion. Exiting...");
            Environment.Exit(-1);
        }

        cameras = new List<sl.CameraIdentifier>();

        // subscribe to every cameras of the setup to internally gather their data
        foreach (var config in fusionConfigurations)
        {
            sl.CameraIdentifier uuid = new sl.CameraIdentifier(config.serialNumber);
            // to subscribe to a camera you must give its serial number, the way to communicate with it (shared memory or local network), and its world pose in the setup.
            Vector3 translation = config.position;
            Quaternion rotation = config.rotation;
            err = fusion.Subscribe(ref uuid, config.commParam, ref translation, ref rotation);
            if (err != sl.FUSION_ERROR_CODE.SUCCESS)
            {
                Console.WriteLine("Error while subscribing to camera " + config.serialNumber + " : " + err + ". Exiting...");
                Environment.Exit(-1);
            }
            else
            {
                cameras.Add(uuid);
            }
        }

        if (cameras.Count == 0)
        {
            Console.WriteLine("No camera subscribed. Exiting...");
            Environment.Exit(-1);
        }

        pointCloudResolution = new sl.Resolution(512, 360);
        pointClouds = new Dictionary<ulong, Mat>();

        foreach (var camera in cameras)
        {
            sl.Mat pointCloud = new sl.Mat();
            pointCloud.Create(pointCloudResolution, MAT_TYPE.MAT_32F_C4, MEM.CPU);
            pointClouds[camera.sn] = pointCloud;
        }

        // as this sample shows how to fuse body detection from the multi camera setup
        // we enable the Body Tracking module with its options
        bodyTrackingFusionParameters.enableTracking = true;
        bodyTrackingFusionParameters.enableBodyFitting = true;
        fusion.EnableBodyTracking(ref bodyTrackingFusionParameters);

        bodyTrackingFusionRuntimeParameters.skeletonMinimumAllowedKeypoints = 7;
        bodyTrackingFusionRuntimeParameters.skeletonMinimumAllowedCameras = cameras.Count / 2;

        camRawData = new Dictionary<ulong, Bodies>();

        // Create OpenGL Viewer
        viewer = new GLViewer();

        Console.Write("Viewer shortcuts : \n" +
        "'r': switch on/off for raw skeleton display\n" +
        "'p': switch on/off for live point cloud display");

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

            uint height = (uint)(wnd_h * 0.9f);
            uint width = (uint)(wnd_w * 0.9f);

            nativeWindow.Create((int)(wnd_h * 0.05f), (int)(wnd_w * 0.05f), width, height, NativeWindowStyle.Resizeable);
            nativeWindow.Show();
            try
            {
                nativeWindow.Run();
            }
            catch(Exception e)
            {
                Console.WriteLine("Mouse wheel is broken in the current OPENGL .NET VERSION. Please do not use it. " + e);
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

        Gl.Enable(EnableCap.Blend);
        Gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

        Gl.Enable(EnableCap.LineSmooth);
        Gl.Hint(HintTarget.LineSmoothHint, HintMode.Nicest);

        viewer.Init(bodyTrackingFusionParameters.enableTracking);

        foreach (var camera in cameras)
        {
            viewer.InitPointCloud(camera.sn, pointCloudResolution);
        }
    }

    // Render loop
    private void NativeWindow_Render(object sender, NativeWindowEventArgs e)
    {
        OpenGL.CoreUI.NativeWindow nativeWindow = (OpenGL.CoreUI.NativeWindow)sender;
        Gl.Viewport(0, 0, (int)nativeWindow.Width, (int)nativeWindow.Height);
        Gl.Clear(ClearBufferMask.ColorBufferBit);

        FUSION_ERROR_CODE err = FUSION_ERROR_CODE.FAILURE;
        // run the fusion as long as the viewer is available.
        if (viewer.IsAvailable())
        {
            // run the fusion process (which gather data from all camera, sync them and process them)
            err = fusion.Process();
            if (err == sl.FUSION_ERROR_CODE.SUCCESS)
            {
                // Retrieve fused body
                fusion.RetrieveBodies(ref bodies, ref bodyTrackingFusionRuntimeParameters, new CameraIdentifier(0));
               
                for (int i = 0; i < cameras.Count; i++)
                {
                    CameraIdentifier uuid = cameras[i];

                    sl.Bodies rawBodies = new sl.Bodies();
                    // Retrieve raw skeleton data
                    err = fusion.RetrieveBodies(ref rawBodies, ref bodyTrackingFusionRuntimeParameters, uuid);
                    camRawData[uuid.sn] = rawBodies;

                    // Retrieve camera pose
                    sl.Pose pose = new sl.Pose();
                    sl.POSITIONAL_TRACKING_STATE state = fusion.GetPosition(ref pose, REFERENCE_FRAME.WORLD, ref uuid, POSITION_TYPE.RAW);
                    if (state == sl.POSITIONAL_TRACKING_STATE.OK)
                    {
                        viewer.SetCameraPose(uuid.sn, pose);
                    }

                    // Retrieve point cloud                 
                    err = fusion.RetrieveMeasure(pointClouds[uuid.sn], ref uuid, MEASURE.XYZBGRA, pointCloudResolution);

                    if (err == sl.FUSION_ERROR_CODE.SUCCESS)
                    {
                        viewer.UpdatePointCloud(uuid.sn, pointClouds[uuid.sn]);
                    }              
                }
            }
            //Update GL View
            viewer.UpdateBodies(bodies, camRawData);
            viewer.Render();
        }
    }

    private void close()
    {
        fusion.Close();
        viewer.Exit();
    }
}
