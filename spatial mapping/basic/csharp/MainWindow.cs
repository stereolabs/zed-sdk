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

/*************************************************************************
 ** This sample shows how to capture a real-time 3D reconstruction      **
 ** of the scene using the Spatial Mapping API. The resulting mesh      **
 ** is displayed as a wireframe on top of the left image using OpenGL.  **
 ** Spatial Mapping can be started and stopped with the space bar key   **
 *************************************************************************/

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
        SpatialMappingParameters spatialMappingParameters;
        RuntimeParameters runtimeParameters;
        POSITIONAL_TRACKING_STATE tracking_state; 
        SPATIAL_MAPPING_STATE mapping_state;
        bool mapping_activated;
        Pose cam_pose;
        Mat zedMat;
        Mesh mesh;
        int timer = 0;
        FusedPointCloud fusedPointCloud;

        // Choose between MESH and FUSED_POINT_CLOUD
        const bool CREATE_MESH = true;

        public MainWindow(string[] args)
        {
            // Set configuration parameters
            InitParameters init_params = new InitParameters();
            init_params.resolution = RESOLUTION.HD720;
            init_params.cameraFPS = 60;
            init_params.depthMode = DEPTH_MODE.ULTRA;
            init_params.coordinateUnits = UNIT.METER;
            init_params.coordinateSystem = COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP;
            init_params.depthMaximumDistance = 15f;
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

            tracking_state = POSITIONAL_TRACKING_STATE.OFF;
            mapping_state = SPATIAL_MAPPING_STATE.NOT_ENABLED;
            mapping_activated = false;

            // Enable tracking
            PositionalTrackingParameters positionalTrackingParameters = new PositionalTrackingParameters();
            zedCamera.EnablePositionalTracking(ref positionalTrackingParameters);

            runtimeParameters = new RuntimeParameters();

            spatialMappingParameters = new SpatialMappingParameters();
            spatialMappingParameters.resolutionMeter = SpatialMappingParameters.get(MAPPING_RESOLUTION.MEDIUM);
            spatialMappingParameters.saveTexture = false;
            if (CREATE_MESH)    spatialMappingParameters.map_type = SPATIAL_MAP_TYPE.MESH;
            else   spatialMappingParameters.map_type = SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD;

            // Create ZED Objects filled in the main loop
            zedMat = new Mat();
            cam_pose = new Pose();

            //Create mesh.
            mesh = new Mesh();
            fusedPointCloud = new FusedPointCloud();
            int Height = zedCamera.ImageHeight;
            int Width = zedCamera.ImageWidth;

            Resolution res = new Resolution((uint)Width, (uint)Height);
            zedMat.Create(res, MAT_TYPE.MAT_8U_C4, MEM.CPU);

            // Create OpenGL Viewer
            viewer = new GLViewer(new Resolution((uint)Width, (uint)Height));

            Console.WriteLine("Hit SPACE BAR to start spatial mapping...");

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
                            viewer.change_state = !viewer.change_state;
                            break;
                    }
                };

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

                nativeWindow.Create((int)(zedCamera.ImageWidth * 0.05f), (int)(zedCamera.ImageHeight * 0.05f), (uint)width, (uint)height, NativeWindowStyle.Resizeable);
                nativeWindow.Show();
                nativeWindow.Run();
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

            viewer.init(zedCamera.GetCalibrationParameters().leftCam, CREATE_MESH);
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

                    if (mapping_activated)
                    {
                        mapping_state = zedCamera.GetSpatialMappingState();
                        if (timer % 60 == 0 && viewer.chunksUpdated() == true)
                        {
                            zedCamera.RequestSpatialMap();
                        }
                        if (zedCamera.GetMeshRequestStatus() == ERROR_CODE.SUCCESS && timer > 0)
                        {
                            /// MAP_TYPE == MESH
                            if (CREATE_MESH)
                            {
                                //Retrieves data for mesh visualization only (vertices + triangles);
                                zedCamera.RetrieveChunks(ref mesh);

                                var chunks = new List<Chunk>(mesh.chunks.Values);
                                viewer.updateData(chunks);
                            }

                            /// MAP_TYPE == FUSED_POINT_CLOUD
                            else
                            {
                                zedCamera.RetrieveSpatialMap(ref fusedPointCloud);
                                viewer.updateData(fusedPointCloud.vertices);
                            }
                        }
                    }

                    bool change_state = viewer.updateImageAndState(zedMat, cam_pose, tracking_state, mapping_state);

                    if (change_state)
                    {
                        if (!mapping_activated)
                        {
                            Quaternion quat = Quaternion.Identity; Vector3 vec = Vector3.Zero;
                            zedCamera.ResetPositionalTracking(quat, vec);                       

                            zedCamera.EnableSpatialMapping(ref spatialMappingParameters);

                            Console.WriteLine("Hit SPACE BAR to stop spatial mapping...");
                            // Clear previous Mesh data
                            viewer.clearCurrentMesh();

                            mapping_activated = true;
                        }
                        else
                        {
                            // Filter the mesh (remove unnecessary vertices and faces)
                            if (CREATE_MESH)
                            {
                                zedCamera.ExtractWholeSpatialMap();
                                zedCamera.FilterMesh(MESH_FILTER.MEDIUM, ref mesh);
                            }

                            if (CREATE_MESH && spatialMappingParameters.saveTexture)
                            {
                                zedCamera.ApplyTexture(ref mesh);
                            }

                            bool error_save = false;
                            string saveName = "";
                            //Save mesh as obj file
                            if (CREATE_MESH)
                            {
                                saveName = "mesh_gen.obj";
                                error_save = zedCamera.SaveMesh(saveName, MESH_FILE_FORMAT.OBJ);
                            }
                            else
                            {
                                saveName = "point_cloud.ply";
                                error_save = zedCamera.SavePointCloud(saveName, MESH_FILE_FORMAT.PLY);
                            }


                            if (error_save)
                            {
                                Console.WriteLine("Mesh saved under: " + saveName);
                            }
                            else
                            {
                                Console.WriteLine("Failed to save the mesh under: " + saveName);
                            }

                            mapping_state = SPATIAL_MAPPING_STATE.NOT_ENABLED;
                            mapping_activated = false;
                            zedCamera.DisableSpatialMapping();

                            Console.WriteLine("Hit SPACE BAR to start spatial mapping...");
                        }
                    }
                    timer++;
                    viewer.render();
                }
            }
        }

        private void close()
        {
            zedCamera.DisableSpatialMapping();
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
