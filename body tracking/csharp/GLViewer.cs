using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using System.Runtime.InteropServices;
using OpenGL;
using OpenGL.CoreUI;
using System.Windows.Input;

namespace sl
{
    public struct ObjectClassName
    {
        public Vector3 position;
        public string name;
        public float4 color;
    };

    class GLViewer
    {
        public GLViewer(Resolution res)
        {
            available = false;
            currentInstance = this;

            floorGrid = new Simple3DObject();
            skeleton = new Simple3DObject();
            pointCloud = new PointCloud();
        }

        public bool isAvailable()
        {
            return available;
        }


        public void init(CameraParameters param, bool showOnlyOk)
        {
            Gl.Enable(EnableCap.FramebufferSrgb);

            shaderSK = new ShaderData();
            shaderSK.it = new Shader(Shader.SK_VERTEX_SHADER, Shader.SK_FRAGMENT_SHADER);
            shaderSK.MVP_Mat = Gl.GetUniformLocation(shaderSK.it.getProgramId(), "u_mvpMatrix");

            shaderLine = new ShaderData();
            shaderLine.it = new Shader(Shader.VERTEX_SHADER, Shader.FRAGMENT_SHADER);
            shaderLine.MVP_Mat = Gl.GetUniformLocation(shaderSK.it.getProgramId(), "u_mvpMatrix");

            //Create Camera
            camera_ = new CameraGL(new Vector3(0, 0, 0), new Vector3(0, 0, -1f), Vector3.UnitY);

            showOnlyOK_ = showOnlyOk;

            pointCloud.initialize(param.resolution);

            sphere = new Simple3DObject();
            sphere.init();
            sphere.setDrawingType(PrimitiveType.Quads);
            sphere.createSphere();

            skeleton.init();
            skeleton.setDrawingType(PrimitiveType.Quads);

            floorGrid.init();
            floorGrid.setDrawingType(PrimitiveType.Lines);

            float limit = 20;
            float4 clr_grid = new float4();
            clr_grid.x = 0.35f;
            clr_grid.y = 0.35f;
            clr_grid.z = 0.35f;
            clr_grid.w = 1f;

            float height = -3;
            for (int i = -(int)limit; i <= (int)limit; i++)
            {
                addVert(ref floorGrid, i, limit, height, clr_grid);
            }

            floorGrid.pushToGPU();

            cam_pose = new Matrix4x4();

            available = true;
        }

        public void update(Mat pointCloud_, Objects objects, sl.Pose pose)
        {
            pointCloud.pushNewPC(pointCloud_);

            cam_pose = Matrix4x4.Identity;
            cam_pose = Matrix4x4.Transform(cam_pose, pose.rotation);
            cam_pose = Matrix4x4.Transpose(cam_pose);

            skeleton.clear();

            if (Keyboard.IsKeyDown(Key.O)) showPC = !showPC;

            // For each object
            for (int idx = 0; idx < objects.numObject; idx++)
            {
                sl.ObjectData obj = objects.objectData[idx];

                // Only show tracked objects
                if (renderObject(obj, showOnlyOK_))
                {
                    float4 clr_id = generateColorID(obj.id);
                    Vector3[] keypoints = obj.keypoints;

                    if (keypoints.Length > 0)
                    {
                        foreach (var limb in BODY_BONES)
                        {
                            Vector3 kp_1 = keypoints[getIdx(limb.Item1)];
                            Vector3 kp_2 = keypoints[getIdx(limb.Item2)];

                            float norm_1 = kp_1.Length();
                            float norm_2 = kp_2.Length();

                            if (!float.IsNaN(norm_1) && norm_1 > 0 && !float.IsNaN(norm_2) && norm_2 > 0)
                            {
                                skeleton.addCylinder(new float3(kp_1.X, kp_1.Y, kp_1.Z), new float3(kp_2.X, kp_2.Y, kp_2.Z), clr_id);
                            }
                        }

                        Vector3 spine = (obj.keypoints[getIdx(sl.BODY_PARTS.LEFT_HIP)] + obj.keypoints[getIdx(sl.BODY_PARTS.RIGHT_HIP)]) / 2;
                        Vector3 neck = obj.keypoints[getIdx(sl.BODY_PARTS.NECK)];
                        float norm_spine = spine.Length();
                        float norm_neck = neck.Length();

                        if (!float.IsNaN(norm_spine) && norm_spine > 0 && !float.IsNaN(norm_neck) && norm_neck > 0)
                        {
                            skeleton.addCylinder(new float3(spine.X, spine.Y, spine.Z), new float3(neck.X, neck.Y, neck.Z), clr_id);
                        }

                        for (int i = 0; i < (int)BODY_PARTS.LAST; i++)
                        {
                            Vector3 kp = keypoints[i];
                            float norm = kp.Length();
                            if (!float.IsNaN(norm) && norm > 0)
                            {
                                skeleton.addSphere(sphere, new float3(kp.X, kp.Y, kp.Z), clr_id);
                            }
                        }

                        if (!float.IsNaN(norm_spine) && norm_spine > 0)
                        {
                            skeleton.addSphere(sphere, new float3(spine.X, spine.Y, spine.Z), clr_id);
                        }
                    }
                }
            }
        }

        void addVert(ref Simple3DObject obj, float i_f, float limit, float height, float4 clr)
        {
            float3 p1 = new float3(i_f, height, -limit);
            float3 p2 = new float3(i_f, height, limit);
            float3 p3 = new float3(-limit, height, i_f);
            float3 p4 = new float3(limit, height, i_f);

            obj.addLine(p1, p2, clr);
            obj.addLine(p3, p4, clr);
        }

        /*void createBboxRendering(List<Vector3> bb_, float4 bbox_clr)
        {
            // First create top and bottom full edges
            BBox_edges.addFullEdges(bb_, bbox_clr);
            // Add faded vertical edges
            BBox_edges.addVerticalEdges(bb_, bbox_clr);
            // Add faces
            BBox_faces.addVerticalFaces(bb_, bbox_clr);
            // Add top face
            BBox_faces.addTopFace(bb_, bbox_clr);
        }

        void createIDRendering(Vector3 center, float4 clr, int id)
        {
            ObjectClassName tmp = new ObjectClassName();
            tmp.name = "ID: " + id.ToString();
            tmp.color = clr;
            tmp.position = center; // Reference point
            objectsName.Add(tmp);
        }*/

        public void render()
        {
            camera_.update();
            skeleton.pushToGPU();

            draw();
        }

        public void keyEventFunction(NativeWindowKeyEventArgs e)
        {
            if (e.Key == KeyCode.R)
            {
                camera_.setPosition(new Vector3(0, 0, 1.5f));
                camera_.setDirection(new Vector3(0, 0, -1), Vector3.UnitY);
            }
            if (e.Key == KeyCode.T)
            {
                camera_.setPosition(new Vector3(0, 0.0f, 1.5f));
                camera_.setOffsetFromPosition(new Vector3(0, 0, 6));
                camera_.translate(new Vector3(0, 1.5f, -4));
                camera_.setDirection(new Vector3(0, -1, 0), Vector3.UnitY);
            }
        }

        public void mouseEventFunction(NativeWindowMouseEventArgs e)
        {
            // Rotate camera with mouse
            if (e.Buttons == OpenGL.CoreUI.MouseButton.Left)
            {
                camera_.rotate(Quaternion.CreateFromAxisAngle(camera_.getRight(), (float)mouseMotion_[1] * MOUSE_R_SENSITIVITY));
                camera_.rotate(Quaternion.CreateFromAxisAngle(camera_.getVertical() * -1f, (float)mouseMotion_[0] * MOUSE_R_SENSITIVITY));
            }
            if (e.Buttons == OpenGL.CoreUI.MouseButton.Right)
            {
                camera_.translate(camera_.getUp() * (float)mouseMotion_[1] * MOUSE_T_SENSITIVITY * -1);
                camera_.translate(camera_.getRight() * (float)mouseMotion_[0] * MOUSE_T_SENSITIVITY * -1);
            }
            if (e.Buttons == OpenGL.CoreUI.MouseButton.Middle)
            {
                camera_.translate(camera_.getForward() * (float)mouseMotion_[1] * MOUSE_T_SENSITIVITY * -1);
                camera_.translate(camera_.getForward() * (float)mouseMotion_[0] * MOUSE_T_SENSITIVITY * -1);
            }
        }

        public void resizeCallback(int width, int height)
        {
            Gl.Viewport(0, 0, width, height);
            float hfov = (180.0f / (float)Math.PI) * (float)(2.0 * Math.Atan(width / (2.0f * 500)));
            float vfov = (180.0f / (float)Math.PI) * (float)(2.0f * Math.Atan(height / (2.0f * 500)));

            camera_.setProjection(hfov, vfov, camera_.getZNear(), camera_.getZFar());
        }

        public void computeMouseMotion(int x, int y)
        {
            currentInstance.mouseMotion_[0] = x - currentInstance.previousMouseMotion_[0];
            currentInstance.mouseMotion_[1] = y - currentInstance.previousMouseMotion_[1];
            currentInstance.previousMouseMotion_[0] = x;
            currentInstance.previousMouseMotion_[1] = y;
        }

        public void draw()
        {
            Matrix4x4 vpMatrix = camera_.getViewProjectionMatrix();

            Gl.UseProgram(shaderLine.it.getProgramId());
            Gl.UniformMatrix4f(shaderSK.MVP_Mat, 1, true, vpMatrix);
            Gl.LineWidth(1.0f);
            floorGrid.draw();
            Gl.UseProgram(0);

            Gl.PointSize(1.0f);
            vpMatrix = vpMatrix * cam_pose;
            if (showPC) pointCloud.draw(vpMatrix);

            Gl.Enable(EnableCap.DepthTest);
            Gl.UseProgram(shaderSK.it.getProgramId());
            Gl.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
            Gl.UniformMatrix4f(shaderSK.MVP_Mat, 1, true, vpMatrix);
            skeleton.draw();
            Gl.UseProgram(0);
            Gl.Disable(EnableCap.DepthTest);
        }

        sl.float4 generateColorID(int idx)
        {
            sl.float4 default_color = new float4();
            default_color.x = 236.0f / 255;
            default_color.y = 184.0f / 255;
            default_color.z = 36.0f / 255;
            default_color.z = 255.0f / 255;

            if (idx < 0) return default_color;

            int offset = Math.Max(0, idx % 8);
            sl.float4 color = new float4();
            color.x = id_colors[offset, 2] / 255;
            color.y = id_colors[offset, 1] / 255;
            color.z = id_colors[offset, 0] / 255;
            color.w = 1.0f;
            return color;
        }

        float[,] id_colors = new float[8, 3]{
            { 232.0f, 176.0f ,59.0f },
        { 165.0f, 218.0f ,25.0f },
            { 102.0f, 205.0f ,105.0f},
            { 185.0f, 0.0f   ,255.0f},
            { 99.0f, 107.0f  ,252.0f},
            {252.0f, 225.0f, 8.0f},
            {167.0f, 130.0f, 141.0f},
            {194.0f, 72.0f, 113.0f}
        };

        float[,] class_colors = new float[6, 3]{
            { 44.0f, 117.0f, 255.0f}, // PEOPLE
            { 255.0f, 0.0f, 255.0f}, // VEHICLE
            { 0.0f, 0.0f, 255.0f},
            { 0.0f, 255.0f, 255.0f},
            { 0.0f, 255.0f, 0.0f},
            { 255.0f, 255.0f, 255.0f}
        };

        float4 getColorClass(int idx)
        {
            idx = Math.Min(5, idx);
            sl.float4 color = new float4();
            color.x = class_colors[idx, 0];
            color.y = class_colors[idx, 1];
            color.z = class_colors[idx, 2];
            color.w = 1.0f;
            return color;
        }

        bool renderObject(ObjectData i, bool showOnlyOK)
        {
            if (showOnlyOK)
                return (i.objectTrackingState == sl.OBJECT_TRACKING_STATE.OK);
            else
                return (i.objectTrackingState == sl.OBJECT_TRACKING_STATE.OK || i.objectTrackingState == sl.OBJECT_TRACKING_STATE.OFF);
        }

        private void setRenderCameraProjection(CameraParameters camParams, float znear, float zfar)
        {
            float PI = 3.141592653f;
            // Just slightly up the ZED camera FOV to make a small black border
            float fov_y = (camParams.vFOV + 0.5f) * PI / 180;
            float fov_x = (camParams.hFOV + 0.5f) * PI / 180;

            projection_.M11 = 1.0f / (float)Math.Tan(fov_x * 0.5f);
            projection_.M22 = 1.0f / (float)Math.Tan(fov_y * 0.5f);
            projection_.M33 = -(zfar + znear) / (zfar - znear);
            projection_.M43 = -1;
            projection_.M34 = -(2.0f * zfar * znear) / (zfar - znear);
            projection_.M44 = 0;

            projection_.M12 = 0;
            projection_.M13 = 2.0f * (((int)camParams.resolution.width - 1.0f * camParams.cx) / (int)camParams.resolution.width) - 1.0f; //Horizontal offset.
            projection_.M14 = 0;

            projection_.M21 = 0;
            projection_.M22 = 1.0f / (float)Math.Tan(fov_y * 0.5f); //Vertical FoV.
            projection_.M23 = -(2.0f * (((int)camParams.resolution.height - 1.0f * camParams.cy) / (int)camParams.resolution.height) - 1.0f); //Vertical offset.
            projection_.M24 = 0;

            projection_.M31 = 0;
            projection_.M32 = 0;

            projection_.M41 = 0;
            projection_.M42 = 0;
        }

        int getIdx(BODY_PARTS part)
        {
            return (int)(part);
        }

        public Tuple<BODY_PARTS, BODY_PARTS>[] BODY_BONES =
        {
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.NOSE, BODY_PARTS.NECK),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.NECK, BODY_PARTS.RIGHT_SHOULDER),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_SHOULDER, BODY_PARTS.RIGHT_ELBOW),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_ELBOW, BODY_PARTS.RIGHT_WRIST),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.NECK, BODY_PARTS.LEFT_SHOULDER),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.LEFT_SHOULDER, BODY_PARTS.LEFT_ELBOW),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.LEFT_ELBOW, BODY_PARTS.LEFT_WRIST),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_HIP, BODY_PARTS.RIGHT_KNEE),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_KNEE, BODY_PARTS.RIGHT_ANKLE),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.LEFT_HIP, BODY_PARTS.LEFT_KNEE),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.LEFT_KNEE, BODY_PARTS.LEFT_ANKLE),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_SHOULDER, BODY_PARTS.LEFT_SHOULDER),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_HIP, BODY_PARTS.LEFT_HIP),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.NOSE, BODY_PARTS.RIGHT_EYE),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_EYE, BODY_PARTS.RIGHT_EAR),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.NOSE, BODY_PARTS.LEFT_EYE),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.LEFT_EYE, BODY_PARTS.LEFT_EAR)
        };

        public void exit()
        {
            if (currentInstance != null)
            {
                available = false;
                skeleton.clear();
            }
        }

        int[] mouseCurrentPosition_ = new int[2];
        int[] mouseMotion_ = new int[2];
        int[] previousMouseMotion_ = new int[2];

        const float MOUSE_R_SENSITIVITY = 0.004f;
        const float MOUSE_T_SENSITIVITY = 0.05f;
        const float MOUSE_UZ_SENSITIVITY = 0.75f;
        const float MOUSE_DZ_SENSITIVITY = 1.25f;

        bool available;
        bool showOnlyOK_ = false;
        Matrix4x4 projection_;
        Matrix4x4 cam_pose;

        PointCloud pointCloud;

        ShaderData shaderSK;
        ShaderData shaderLine;

        Simple3DObject sphere;
        Simple3DObject skeleton;
        Simple3DObject floorGrid;

        GLViewer currentInstance;
        CameraGL camera_;

        bool showPC = false;
    }

    class ImageHandler
    {
        public ImageHandler(Resolution res)
        {
            resolution = res;
        }

        // Initialize Opengl buffers
        public void initialize()
        {
            shaderImage.it = new Shader(Shader.IMAGE_VERTEX_SHADER, Shader.IMAGE_FRAGMENT_SHADER);
            texID = Gl.GetUniformLocation(shaderImage.it.getProgramId(), "texImage");

            float[] g_quad_vertex_buffer_data = new float[18]{
                -1.0f, -1.0f, 0.0f,
                1.0f, -1.0f, 0.0f,
                -1.0f, 1.0f, 0.0f,
                -1.0f, 1.0f, 0.0f,
                1.0f, -1.0f, 0.0f,
                1.0f, 1, 0.0f};

            quad_vb = Gl.GenBuffer();
            Gl.BindBuffer(BufferTarget.ArrayBuffer, quad_vb);
            Gl.BufferData(BufferTarget.ArrayBuffer, (uint)(sizeof(float) * g_quad_vertex_buffer_data.Length), g_quad_vertex_buffer_data, BufferUsage.StaticDraw);
            Gl.BindBuffer(BufferTarget.ArrayBuffer, 0);

            Gl.Enable(EnableCap.Texture2d);
            imageTex = Gl.GenTexture();
            Gl.BindTexture(TextureTarget.Texture2d, imageTex);
            Gl.TexParameter(TextureTarget.Texture2d, TextureParameterName.TextureWrapS, Gl.CLAMP_TO_BORDER);
            Gl.TexParameter(TextureTarget.Texture2d, TextureParameterName.TextureWrapT, Gl.CLAMP_TO_BORDER);
            Gl.TexParameter(TextureTarget.Texture2d, TextureParameterName.TextureWrapR, Gl.CLAMP_TO_BORDER);
            Gl.TexParameter(TextureTarget.Texture2d, TextureParameterName.TextureMinFilter, Gl.LINEAR);
            Gl.TexParameter(TextureTarget.Texture2d, TextureParameterName.TextureMagFilter, Gl.LINEAR);
            Gl.TexImage2D(TextureTarget.Texture2d, 0, InternalFormat.Rgba, (int)resolution.width, (int)resolution.height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, null);
            Gl.BindTexture(TextureTarget.Texture2d, 0);
        }

        public void pushNewImage(Mat zedImage)
        {
            // Update Texture with current zedImage
            Gl.TexSubImage2D(TextureTarget.Texture2d, 0, 0, 0, zedImage.GetWidth(), zedImage.GetHeight(), PixelFormat.Rgba, PixelType.UnsignedByte, zedImage.GetPtr());
        }

        // Draw the Image
        public void draw()
        {
            Gl.UseProgram(shaderImage.it.getProgramId());
            Gl.ActiveTexture(TextureUnit.Texture0);
            Gl.BindTexture(TextureTarget.Texture2d, imageTex);
            Gl.Uniform1(texID, 0);

            Gl.EnableVertexAttribArray(0);
            Gl.BindBuffer(BufferTarget.ArrayBuffer, quad_vb);
            Gl.VertexAttribPointer(0, 3, VertexAttribType.Float, false, 0, IntPtr.Zero);
            Gl.DrawArrays(PrimitiveType.Triangles, 0, 6);
            Gl.DisableVertexAttribArray(0);
            Gl.UseProgram(0);
        }

        public void close()
        {
            Gl.DeleteTextures(imageTex);
        }

        private int texID;
        private uint imageTex;

        private ShaderData shaderImage;
        private Resolution resolution;
        private uint quad_vb;

    };

    class PointCloud
    {
        public PointCloud()
        {
            shader = new ShaderData();
            mat_ = new Mat();
        }

        void close()
        {
            if (mat_.IsInit())
            {
                mat_.Free();
                Gl.DeleteBuffers(1, bufferGLID_);
            }
        }

        public void initialize(Resolution res)
        {
            bufferGLID_ = Gl.GenBuffer();
            Gl.BindBuffer(BufferTarget.ArrayBuffer, bufferGLID_);
            Gl.BufferData(BufferTarget.ArrayBuffer, (uint)res.height * (uint)res.width * sizeof(float) * 4, IntPtr.Zero, BufferUsage.StaticDraw);
            Gl.BindBuffer(BufferTarget.ArrayBuffer, 0);

            shader.it = new Shader(Shader.POINTCLOUD_VERTEX_SHADER, Shader.POINTCLOUD_FRAGMENT_SHADER);
            shader.MVP_Mat = Gl.GetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");

            mat_.Create(res, sl.MAT_TYPE.MAT_32F_C4, MEM.CPU);
            //mat_.Create(new Resolution(Math.Min((uint)res.width, 720), Math.Min((uint)res.height, 404)), sl.MAT_TYPE.MAT_32F_C4, MEM.CPU);
        }

        public void pushNewPC(Mat matXYZRGBA)
        {
            if (mat_.IsInit())
            {
                mat_.SetFrom(matXYZRGBA, COPY_TYPE.CPU_CPU);
            }
        }

        public void draw(Matrix4x4 vp)
        {
            if (mat_.IsInit())
            {
                Gl.UseProgram(shader.it.getProgramId());
                Gl.UniformMatrix4f(shader.MVP_Mat, 1, true, vp);

                Gl.EnableVertexAttribArray(Shader.ATTRIB_VERTICES_POS);
                Gl.BindBuffer(BufferTarget.ArrayBuffer, bufferGLID_);
                Gl.BufferData(BufferTarget.ArrayBuffer, (uint)mat_.GetResolution().height * (uint)mat_.GetResolution().width * sizeof(float) * 4, mat_.GetPtr(), BufferUsage.StaticDraw);
                Gl.VertexAttribPointer(Shader.ATTRIB_VERTICES_POS, 4, VertexAttribType.Float, false, 0, IntPtr.Zero);
                Gl.DrawArrays(PrimitiveType.Points, 0, (int)mat_.GetResolution().height * (int)mat_.GetResolution().width);
                Gl.BindBuffer(BufferTarget.ArrayBuffer, 0);
                Gl.UseProgram(0);

            }
        }
        uint bufferGLID_;

        Mat mat_;
        ShaderData shader;
    };

    class CameraGL
    {
        public CameraGL() { }

        public CameraGL(Vector3 position, Vector3 direction, Vector3 vertical)
        {
            position_ = position;
            setDirection(direction, vertical);
            offset_ = new Vector3(0, 0, 0);
            view_ = Matrix4x4.Identity;
            updateView();
            setProjection(70, 70, 0.2f, 50);
            updateVPMatrix();
        }

        public void update()
        {
            if (Vector3.Dot(vertical_, up_) < 0)
            {
                vertical_ = vertical_ * -1f;
            }
            updateView();
            updateVPMatrix();
        }

        public void setProjection(float horizontalFOV, float verticalFOV, float znear, float zfar)
        {
            horizontalFieldOfView_ = horizontalFOV;
            verticalFieldOfView_ = verticalFOV;
            znear_ = znear;
            zfar_ = zfar;

            float fov_y = verticalFOV * (float)Math.PI / 180.0f;
            float fov_x = horizontalFOV * (float)Math.PI / 180.0f;

            projection_ = Matrix4x4.Identity;

            projection_.M11 = 1.0f / (float)Math.Tan(fov_x * 0.5f);
            projection_.M22 = 1.0f / (float)Math.Tan(fov_y * 0.5f);
            projection_.M33 = -(zfar + znear) / (zfar - znear);
            projection_.M43 = -1;
            projection_.M34 = -(2.0f * zfar * znear) / (zfar - znear);
            projection_.M44 = 0;

        }

        public Matrix4x4 getViewProjectionMatrix() { return vpMatrix_; }

        public float getHorizontalFOV() { return horizontalFieldOfView_; }

        public float getVerticalFOV() { return verticalFieldOfView_; }

        public void setOffsetFromPosition(Vector3 o) { offset_ = o; }

        public Vector3 getOffsetFromPosition() { return offset_; }

        public void setDirection(Vector3 direction, Vector3 vertical)
        {
            Vector3 dirNormalized = Vector3.Normalize(direction);

            // Create rotation
            Vector3 tr1 = Vector3.UnitZ;
            Vector3 tr2 = dirNormalized * -1f;
            float cos_theta = Vector3.Dot(Vector3.Normalize(tr1), Vector3.Normalize(tr2));
            float angle = 0.5f * (float)Math.Acos(cos_theta);
            Vector3 w = Vector3.Cross(tr1, tr2);
            if (Vector3.Zero != w)
            {

                w = Vector3.Normalize(w);
            }

            float half_sin = (float)Math.Sin(angle);
            rotation_.W = (float)Math.Cos(angle);
            rotation_.X = half_sin * w.X;
            rotation_.Y = half_sin * w.Y;
            rotation_.Z = half_sin * w.Z;

            ///////////////////////
            updateVectors();
            vertical_ = vertical;
            if (Vector3.Dot(vertical, up_) < 0) rotate(Quaternion.CreateFromAxisAngle(Vector3.UnitZ, (float)Math.PI));
        }

        public void translate(Vector3 t) { position_ = position_ + t; }

        public void setPosition(Vector3 p) { position_ = p; }

        public void rotate(Quaternion rot)
        {
            rotation_ = rot * rotation_;
            updateVectors();
        }

        public void rotate(Matrix4x4 m)
        {
            rotate(Quaternion.CreateFromRotationMatrix(m));
        }

        public void setRotation(Quaternion rot)
        {
            rotation_ = rot;
            updateVectors();
        }

        public void setRotation(Matrix4x4 m)
        {
            setRotation(Quaternion.CreateFromRotationMatrix(m));
        }

        public Vector3 getPosition() { return position_; }

        public Vector3 getForward() { return forward_; }

        public Vector3 getRight() { return right_; }

        public Vector3 getUp() { return up_; }

        public Vector3 getVertical() { return vertical_; }

        public float getZNear() { return znear_; }

        public float getZFar() { return zfar_; }

        void updateVectors()
        {
            forward_ = Vector3.Transform(Vector3.UnitZ, rotation_);
            up_ = Vector3.Transform(Vector3.UnitY, rotation_);
            right_ = Vector3.Transform(Vector3.UnitX, rotation_);
        }

        void updateView()
        {
            Matrix4x4 transformation = Matrix4x4.Identity;

            transformation = Matrix4x4.Transform(transformation, rotation_);
            transformation.Translation = Vector3.Transform(offset_, rotation_) + position_;
            transformation = Matrix4x4.Transpose(transformation);

            Matrix4x4.Invert(transformation, out view_);
        }

        void updateVPMatrix()
        {
            vpMatrix_ = projection_ * view_;
        }

        public Matrix4x4 projection_;

        Vector3 offset_;
        Vector3 position_;
        Vector3 forward_;
        Vector3 up_;
        Vector3 right_;
        Vector3 vertical_;

        Quaternion rotation_;

        Matrix4x4 view_;
        Matrix4x4 vpMatrix_;

        float horizontalFieldOfView_;
        float verticalFieldOfView_;
        float znear_;
        float zfar_;
    };

    class Shader
    {

        public static readonly string[] IMAGE_VERTEX_SHADER = new string[] {
            "#version 330\n",
            "layout(location = 0) in vec3 vert;\n",
            "out vec2 UV;",
            "void main() {\n",
            "   UV = (vert.xy+vec2(1,1))/2;\n",
            "	gl_Position = vec4(vert, 1);\n",
            "}\n"
        };

        public static readonly string[] IMAGE_FRAGMENT_SHADER = new string[] {
            "#version 330 core\n",
            "in vec2 UV;\n",
            "out vec4 color;\n",
            "uniform sampler2D texImage;\n",
            "void main() {\n",
            "   vec2 scaler = vec2(UV.x,1.f - UV.y);\n",
            "   vec3 rgbcolor = vec3(texture(texImage, scaler).zyx);\n",
            "   float gamma = 1.0/1.65;\n",
            "   vec3 color_rgb = pow(rgbcolor, vec3(1.0/gamma));\n",
            "   color = vec4(color_rgb,1);\n",
            "}"
        };

        public static readonly string[] SK_VERTEX_SHADER = new string[] {
            "#version 330 core\n",
            "layout(location = 0) in vec3 in_Vertex;\n",
            "layout(location = 1) in vec4 in_Color;\n",
            "layout(location = 2) in vec3 in_Normal;\n",
            "out vec4 b_color;\n",
            "out vec3 b_position;\n",
            "out vec3 b_normal;\n",
            "uniform mat4 u_mvpMatrix;\n",
            "uniform vec4 u_color;\n",
            "void main() {\n",
            "   b_color = in_Color;\n",
            "   b_position = in_Vertex;\n",
            "   b_normal = in_Normal;\n",
            "	gl_Position =  u_mvpMatrix * vec4(in_Vertex, 1);\n",
            "}"
        };

        public static readonly string[] SK_FRAGMENT_SHADER = new string[] {
            "#version 330 core\n",
            "in vec4 b_color;\n",
            "in vec3 b_position;\n",
            "in vec3 b_normal;\n",
            "out vec4 out_Color;\n",
            "void main() {\n",
            "	vec3 lightPosition = vec3(0, 10, 0);\n",
            "	vec3 lightColor = vec3(1,1,1);\n",
            "	float ambientStrength = 0.4;\n",
            "	vec3 ambient = ambientStrength * lightColor;\n",
            "	vec3 norm = normalize(b_normal); \n",
            "	vec3 lightDir = normalize(lightPosition - b_position);\n",
            "	float diffuse = (1 - ambientStrength) * max(dot(b_normal, lightDir), 0.0);\n",
            "   out_Color = vec4(b_color.rgb * (diffuse + ambient), 1);\n",
            "}"
        };

        public static readonly string[] POINTCLOUD_VERTEX_SHADER = new string[] {
        "#version 330 core\n",
        "layout(location = 0) in vec4 in_VertexRGBA;\n",
        "uniform mat4 u_mvpMatrix;\n",
        "out vec4 b_color;\n",
        "void main() {\n",
        // Decompose the 4th channel of the XYZRGBA buffer to retrieve the color of the point (1float to 4uint)
        "   uint vertexColor = floatBitsToUint(in_VertexRGBA.w); \n",
        "   vec3 clr_int = vec3((vertexColor & uint(0x000000FF)), (vertexColor & uint(0x0000FF00)) >> 8, (vertexColor & uint(0x00FF0000)) >> 16);\n",
        "   b_color = vec4(clr_int.r / 255.0f, clr_int.g / 255.0f, clr_int.b / 255.0f, 1.f);",
        "	gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);\n",
        "}"
    };

        public static readonly string[] POINTCLOUD_FRAGMENT_SHADER = new string[] {
        "#version 330 core\n",
        "in vec4 b_color;\n",
        "layout(location = 0) out vec4 out_Color;\n",
        "void main() {\n",
        "   out_Color = b_color;\n",
        "}"
    };

        public static readonly string[] VERTEX_SHADER = new string[] {
            "#version 330 core\n",
            "layout(location = 0) in vec3 in_Vertex;\n",
            "layout(location = 1) in vec4 in_Color;\n",
            "uniform mat4 u_mvpMatrix;\n",
            "out vec4 b_color;\n",
            "void main() {\n",
            "   b_color = in_Color;\n",
            "	gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n",
            "}"
        };

        public static readonly string[] FRAGMENT_SHADER = new string[] {
            "#version 330 core\n",
            "in vec4 b_color;\n",
            "layout(location = 0) out vec4 out_Color;\n",
            "void main() {\n",
            "   out_Color = b_color;\n",
            "}"
        };

        public Shader(string[] vs, string[] fs)
        {
            if (!compile(ref verterxId_, ShaderType.VertexShader, vs))
            {
                Console.WriteLine("ERROR: while compiling vertex shader");
            }
            if (!compile(ref fragmentId_, ShaderType.FragmentShader, fs))
            {
                Console.WriteLine("ERROR: while compiling fragment shader");
            }

            programId_ = Gl.CreateProgram();

            Gl.AttachShader(programId_, verterxId_);
            Gl.AttachShader(programId_, fragmentId_);

            Gl.BindAttribLocation(programId_, ATTRIB_VERTICES_POS, "in_vertex");
            Gl.BindAttribLocation(programId_, ATTRIB_COLOR_POS, "in_texCoord");

            Gl.LinkProgram(programId_);

            int errorlk = 0;
            Gl.GetProgram(programId_, ProgramProperty.LinkStatus, out errorlk);
            if (errorlk != Gl.TRUE)
            {
                Console.WriteLine("ERROR: while linking Shader :");
                int errorSize = 0;
                Gl.GetProgram(programId_, ProgramProperty.InfoLogLength, out errorSize);
                StringBuilder error = new StringBuilder(1024);
                Gl.GetShaderInfoLog(programId_, errorSize, out errorSize, error);
                Console.WriteLine(error.ToString());
                error.Clear();
                Gl.DeleteProgram(programId_);
            }
        }

        public uint getProgramId()
        {
            return programId_;
        }

        public static uint ATTRIB_VERTICES_POS = 0;
        public static uint ATTRIB_COLOR_POS = 1;
        public static uint ATTRIB_NORMAL = 2;

        private bool compile(ref uint shaderId, ShaderType type, string[] src)
        {
            int errorcp = 0;

            shaderId = Gl.CreateShader(type);
            if (shaderId == 0)
            {
                Console.WriteLine("ERROR: shader type (" + type + ") does not exist");
            }

            Gl.ShaderSource(shaderId, src);
            Gl.CompileShader(shaderId);
            Gl.GetShader(shaderId, ShaderParameterName.CompileStatus, out errorcp);

            if (errorcp != Gl.TRUE)
            {
                Console.WriteLine("ERROR: while compiling Shader :");
                int errorSize;
                Gl.GetShader(shaderId, ShaderParameterName.InfoLogLength, out errorSize);

                StringBuilder error = new StringBuilder(1024);
                Gl.GetShaderInfoLog(shaderId, errorSize, out errorSize, error);

                Console.WriteLine(error.ToString());
                error.Clear();

                Gl.DeleteShader(shaderId);
                return false;
            }
            return true;
        }

        uint verterxId_;
        uint fragmentId_;
        uint programId_;
    };

    struct ShaderData
    {
        public Shader it;
        public int MVP_Mat;
    };

    class Simple3DObject
    {
        public Simple3DObject()
        {
            is_init = false;
        }

        public void init()
        {
            vaoID_ = 0;
            isStatic_ = false;

            shader.it = new Shader(Shader.SK_VERTEX_SHADER, Shader.SK_FRAGMENT_SHADER);
            shader.MVP_Mat = Gl.GetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");

            vertices_ = new List<float>();
            colors_ = new List<float>();
            indices_ = new List<uint>();
            normals_ = new List<float>();

            is_init = true;
        }

        public bool isInit()
        {
            return is_init;
        }

        public void addPt(float3 pt)
        {
            vertices_.Add(pt.x);
            vertices_.Add(pt.y);
            vertices_.Add(pt.z);
        }

        public void addClr(float4 clr)
        {
            colors_.Add(clr.x);
            colors_.Add(clr.y);
            colors_.Add(clr.z);
            colors_.Add(clr.w);
        }

        public void addNormal(float3 normal)
        {
            normals_.Add(normal.x);
            normals_.Add(normal.y);
            normals_.Add(normal.z);
        }

        void addBBox(List<float3> pts, float4 clr)
        {
            int start_id = vertices_.Count / 3;

            float transparency_top = 0.05f, transparency_bottom = 0.75f;
            for (int i = 0; i < pts.Count; i++)
            {
                addPt(pts[i]);
                clr.w = (i < 4 ? transparency_top : transparency_bottom);
                addClr(clr);
            }

            uint[] boxLinks = new uint[] { 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7 };

            for (int i = 0; i < boxLinks.Length; i += 2)
            {
                indices_.Add((uint)start_id + boxLinks[i]);
                indices_.Add((uint)start_id + boxLinks[i + 1]);
            }
        }

        public void addPoint(float3 pt, float4 clr)
        {
            addPt(pt);
            addClr(clr);
            indices_.Add((uint)indices_.Count());
        }

        public void addLine(float3 p1, float3 p2, float4 clr)
        {
            addPoint(p1, clr);
            addPoint(p2, clr);
        }

        public void addTriangle(float3 p1, float3 p2, float3 p3, float4 clr)
        {
            addPoint(p1, clr);
            addPoint(p2, clr);
            addPoint(p3, clr);
        }

        public void addFullEdges(List<Vector3> pts, float4 clr)
        {
            int start_id = vertices_.Count / 3;

            for (int i = 0; i < pts.Count; i++)
            {
                addPt(new float3(pts[i].X, pts[i].Y, pts[i].Z));
                addClr(clr);
            }

            uint[] boxLinksTop = new uint[] { 0, 1, 1, 2, 2, 3, 3, 0 };
            for (int i = 0; i < boxLinksTop.Length; i += 2)
            {
                indices_.Add((uint)start_id + boxLinksTop[i]);
                indices_.Add((uint)start_id + boxLinksTop[i + 1]);
            }

            uint[] boxLinksBottom = new uint[] { 4, 5, 5, 6, 6, 7, 7, 4 };
            for (int i = 0; i < boxLinksBottom.Length; i += 2)
            {
                indices_.Add((uint)start_id + boxLinksBottom[i]);
                indices_.Add((uint)start_id + boxLinksBottom[i + 1]);
            }
        }

        public void addSingleVerticalLine(Vector3 top_pt, Vector3 bot_pt, float4 clr)
        {
            List<Vector3> current_pts = new List<Vector3>()
            {
                top_pt,
                ((grid_size - 1.0f) * top_pt + bot_pt) / grid_size,
                ((grid_size - 2.0f) * top_pt + bot_pt* 2.0f) / grid_size,
                (2.0f * top_pt + bot_pt* (grid_size - 2.0f)) / grid_size,
                (top_pt + bot_pt* (grid_size - 1.0f)) / grid_size,
                bot_pt
            };

            int start_id = vertices_.Count / 3;
            for (int i = 0; i < current_pts.Count; i++)
            {
                addPt(new float3(current_pts[i].X, current_pts[i].Y, current_pts[i].Z));
                clr.w = (i == 2 || i == 3) ? 0.0f : 0.75f;
                addClr(clr);
            }

            uint[] boxLinks = new uint[] { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5 };
            for (int i = 0; i < boxLinks.Length; i += 2)
            {
                indices_.Add((uint)start_id + boxLinks[i]);
                indices_.Add((uint)start_id + boxLinks[i + 1]);
            }
        }

        public void addVerticalEdges(List<Vector3> pts, float4 clr)
        {

            addSingleVerticalLine(pts[0], pts[4], clr);
            addSingleVerticalLine(pts[1], pts[5], clr);
            addSingleVerticalLine(pts[2], pts[6], clr);
            addSingleVerticalLine(pts[3], pts[7], clr);
        }

        public void addTopFace(List<Vector3> pts, float4 clr)
        {
            clr.w = 0.3f;
            foreach (Vector3 it in pts)
                addPoint(new float3(it.X, it.Y, it.Z), clr);
        }

        void addQuad(List<Vector3> quad_pts, float alpha1, float alpha2, float4 clr)
        { // To use only with 4 points
            for (int i = 0; i < quad_pts.Count; ++i)
            {
                addPt(new float3(quad_pts[i].X, quad_pts[i].Y, quad_pts[i].Z));
                clr.w = (i < 2 ? alpha1 : alpha2);
                addClr(clr);
            }

            indices_.Add((uint)indices_.Count);
            indices_.Add((uint)indices_.Count);
            indices_.Add((uint)indices_.Count);
            indices_.Add((uint)indices_.Count);
        }

        public void addVerticalFaces(List<Vector3> pts, float4 clr)
        {
            // For each face, we need to add 4 quads (the first 2 indexes are always the top points of the quad)
            int[][] quads = new int[4][]
            {
                new int[4]
                {
                    0, 3, 7, 4
                }, // front face
                 new int[4]
                {
                        3, 2, 6, 7
                }, // right face
                new int[4]
                {
                        2, 1, 5, 6
                }, // back face
                new int[4]
                {
                        1, 0, 4, 5
                } // left face
            };
            float alpha = 0.5f;

            foreach (int[] quad in quads)
            {

                // Top quads
                List<Vector3> quad_pts_1 = new List<Vector3> {
                    pts[quad[0]],
                    pts[quad[1]],
                    ((grid_size - 0.5f) * pts[quad[1]] +  0.5f * pts[quad[2]]) / grid_size,
                    ((grid_size - 0.5f) * pts[quad[0]] + 0.5f * pts[quad[3]]) / grid_size };
                addQuad(quad_pts_1, alpha, alpha, clr);

                List<Vector3> quad_pts_2 = new List<Vector3> {
                    ((grid_size - 0.5f) * pts[quad[0]] + 0.5f * pts[quad[3]]) / grid_size,
                    ((grid_size - 0.5f) * pts[quad[1]] +  0.5f * pts[quad[2]]) / grid_size,
                    ((grid_size - 1.0f) * pts[quad[1]] + pts[quad[2]]) / grid_size,
                    ((grid_size - 1.0f) * pts[quad[0]] + pts[quad[3]]) / grid_size};
                addQuad(quad_pts_2, alpha, 2 * alpha / 3, clr);

                List<Vector3> quad_pts_3 = new List<Vector3> {
                    ((grid_size - 1.0f) * pts[quad[0]] + pts[quad[3]]) / grid_size,
                    ((grid_size - 1.0f) * pts[quad[1]] + pts[quad[2]]) / grid_size,
                    ((grid_size - 1.5f) * pts[quad[1]] + 1.5f * pts[quad[2]]) / grid_size,
                    ((grid_size - 1.5f) * pts[quad[0]] + 1.5f * pts[quad[3]]) / grid_size};
                addQuad(quad_pts_3, 2 * alpha / 3, alpha / 3, clr);

                List<Vector3> quad_pts_4 = new List<Vector3> {
                    ((grid_size - 1.5f) * pts[quad[0]] + 1.5f * pts[quad[3]]) / grid_size,
                    ((grid_size - 1.5f) * pts[quad[1]] + 1.5f * pts[quad[2]]) / grid_size,
                    ((grid_size - 2.0f) * pts[quad[1]] + 2.0f * pts[quad[2]]) / grid_size,
                    ((grid_size - 2.0f) * pts[quad[0]] + 2.0f * pts[quad[3]]) / grid_size};
                addQuad(quad_pts_4, alpha / 3, 0.0f, clr);

                // Bottom quads
                List<Vector3> quad_pts_5 = new List<Vector3> {
                    (pts[quad[1]] * 2.0f + (grid_size - 2.0f) * pts[quad[2]]) / grid_size,
                    (pts[quad[0]] * 2.0f + (grid_size - 2.0f) * pts[quad[3]]) / grid_size,
                    (pts[quad[0]] * 1.5f + (grid_size - 1.5f) * pts[quad[3]]) / grid_size,
                    (pts[quad[1]] * 1.5f + (grid_size - 1.5f) * pts[quad[2]]) / grid_size };
                addQuad(quad_pts_5, 0.0f, alpha / 3, clr);

                List<Vector3> quad_pts_6 = new List<Vector3> {
                    (pts[quad[1]] * 1.5f + (grid_size - 1.5f) * pts[quad[2]]) / grid_size,
                    (pts[quad[0]] * 1.5f + (grid_size - 1.5f) * pts[quad[3]]) / grid_size,
                    (pts[quad[0]] + (grid_size - 1.0f) * pts[quad[3]]) / grid_size,
                    (pts[quad[1]] + (grid_size - 1.0f) * pts[quad[2]]) / grid_size};
                addQuad(quad_pts_6, alpha / 3, 2 * alpha / 3, clr);

                List<Vector3> quad_pts_7 = new List<Vector3> {
                    (pts[quad[1]] + (grid_size - 1.0f) * pts[quad[2]]) / grid_size,
                    (pts[quad[0]] + (grid_size - 1.0f) * pts[quad[3]]) / grid_size,
                    (pts[quad[0]] * 0.5f + (grid_size - 0.5f) * pts[quad[3]]) / grid_size,
                    (pts[quad[1]] * 0.5f + (grid_size - 0.5f) * pts[quad[2]]) / grid_size};
                addQuad(quad_pts_7, 2 * alpha / 3, alpha, clr);

                List<Vector3> quad_pts_8 = new List<Vector3> {
                    (pts[quad[0]] * 0.5f + (grid_size - 0.5f) * pts[quad[3]]) / grid_size,
                    (pts[quad[1]] * 0.5f + (grid_size - 0.5f) * pts[quad[2]]) / grid_size,
                    pts[quad[2]],
                    pts[quad[3]]};
                addQuad(quad_pts_8, alpha, alpha, clr);
            }
        }

        public void addCylinder(float3 startPosition, float3 endPosition, float4 clr)
        {

            /////////////////////////////
            /// Compute Rotation Matrix
            /////////////////////////////

            const float PI = 3.1415926f;

            float m_radius = 0.010f;

            float3 dir = endPosition.sub(startPosition);

            float m_height = dir.norm();
            float x = 0.0f, y = 0.0f, z = 0.0f;

            dir.divide(m_height);

            float3 yAxis = new float3(0, 1, 0);

            float3 v = dir.cross(yAxis); ;

            Matrix4x4 rotation;

            float sinTheta = v.norm();
            if (sinTheta < 0.00001f)
            {
                rotation = Matrix4x4.Identity;
            }
            else
            {
                float cosTheta = dir.dot(yAxis);
                float scale = (1.0f - cosTheta) / (1.0f - (cosTheta * cosTheta));

                Matrix4x4 vx = new Matrix4x4(0, v.z, -v.y, 0,
                                            -v.z, 0, v.x, 0,
                                            v.y, -v.x, 0, 0,
                                            0, 0, 0, 1.0f);

                Matrix4x4 vx2 = vx * vx;
                Matrix4x4 vx2Scaled = vx2 * scale;

                rotation = Matrix4x4.Identity;
                rotation = rotation + vx;
                rotation = rotation + vx2Scaled;
            }

            /////////////////////////////
            /// Create Cylinder
            /////////////////////////////

            Matrix3x3 rotationMatrix = new Matrix3x3();
            float[] data = new float[9] { rotation.M11, rotation.M12, rotation.M13, rotation.M21, rotation.M22, rotation.M23, rotation.M31, rotation.M32, rotation.M33 };
            rotationMatrix.m = data;

            float3 v1;
            float3 v2;
            float3 v3;
            float3 v4;
            float3 normal;
            float resolution = 0.1f;
            int NB_SEG = 32;
            float scale_seg = (float)1 / (float)NB_SEG;

            for (int j = 0; j < NB_SEG; j++)
            {
                float i = (float)2 * PI * ((float)j * scale_seg);
                float i1 = (float)2 * PI * ((float)(j + 1) * scale_seg);
                v1 = rotationMatrix.multiply(new float3(m_radius * (float)Math.Cos(i), 0, m_radius * (float)Math.Sin(i))).add(startPosition);
                v2 = rotationMatrix.multiply(new float3(m_radius * (float)Math.Cos(i), m_height, m_radius * (float)Math.Sin(i))).add(startPosition);
                v4 = rotationMatrix.multiply(new float3(m_radius * (float)Math.Cos(i1), m_height, m_radius * (float)Math.Sin(i1))).add(startPosition);
                v3 = rotationMatrix.multiply(new float3(m_radius * (float)Math.Cos(i1), 0, m_radius * (float)Math.Sin(i1))).add(startPosition);

                float3 a = v2.sub(v1);
                float3 b = v3.sub(v1);
                normal = a.cross(b);
                normal.divide(normal.norm());

                addPoint(v1, clr);
                addPoint(v2, clr);
                addPoint(v4, clr);
                addPoint(v3, clr);

                addNormal(normal);
                addNormal(normal);
                addNormal(normal);
                addNormal(normal);
            }
        }

        public void createSphere()
        {
            const float PI = 3.1415926f;

            float m_radius = 0.02f;
            float radiusInv = 1.0f / m_radius;

            int m_stackCount = 16;
            int m_sectorCount = 16;

            float3 v1;
            float3 v2;
            float3 v3;
            float3 v4;
            float3 normal;

            for (int i = 0; i <= m_stackCount; i++)
            {
                double lat0 = PI * (-0.5 + (double)(i - 1) / m_stackCount);
                float z0 = (float)Math.Sin(lat0);
                float zr0 = (float)Math.Cos(lat0);

                double lat1 = PI * (-0.5 + (double)i / m_stackCount);
                float z1 = (float)Math.Sin(lat1);
                float zr1 = (float)Math.Cos(lat1);

                for (int j = 0; j <= m_sectorCount - 1; j++)
                {
                    double lng = 2 * PI * (double)(j - 1) / m_sectorCount;
                    float x = (float)Math.Cos(lng);
                    float y = (float)Math.Sin(lng);

                    v1 = new float3(m_radius * x * zr0, m_radius * y * zr0, m_radius * z0);
                    normal = new float3(x * zr0, y * zr0, z0);
                    addPt(v1);
                    indices_.Add((uint)indices_.Count());
                    addNormal(normal);

                    v2 = new float3(m_radius * x * zr1, m_radius * y * zr1, m_radius * z1);
                    normal = new float3(x * zr1, y * zr1, z1);
                    addPt(v2);
                    indices_.Add((uint)indices_.Count());
                    addNormal(normal);

                    lng = 2 * PI * (double)j / m_sectorCount;
                    x = (float)Math.Cos(lng);
                    y = (float)Math.Sin(lng);

                    v4 = new float3(m_radius * x * zr1, m_radius * y * zr1, m_radius * z1);
                    normal = new float3(x * zr1, y * zr1, z1);
                    addPt(v4);
                    indices_.Add((uint)indices_.Count());
                    addNormal(normal);

                    v3 = new float3(m_radius * x * zr0, m_radius * y * zr0, m_radius * z0);
                    normal = new float3(x * zr0, y * zr0, z0);
                    addPt(v3);
                    indices_.Add((uint)indices_.Count());
                    addNormal(normal);
                }
            }
        }

        public void addSphere(Simple3DObject sphere, float3 position, float4 clr)
        {
            for (int i = 0; i < sphere.vertices_.Count; i += 3)
            {
                float3 point = new float3(sphere.vertices_[i], sphere.vertices_[i + 1], sphere.vertices_[i + 2]).add(position);
                addPoint(point, clr);
                float3 normal = new float3(sphere.normals_[i], sphere.normals_[i + 1], sphere.normals_[i + 2]);
                addNormal(normal);
            }
        }

        public void pushToGPU()
        {
            if (!isStatic_ || vaoID_ == 0)
            {
                if (vaoID_ == 0)
                {
                    vaoID_ = Gl.GenVertexArray();
                    Gl.GenBuffers(vboID_);
                }

                Gl.ShadeModel(ShadingModel.Smooth);
                if (vertices_.Count() > 0)
                {
                    Gl.BindVertexArray(vaoID_);
                    Gl.BindBuffer(BufferTarget.ArrayBuffer, vboID_[0]);
                    Gl.BufferData(BufferTarget.ArrayBuffer, (uint)vertices_.Count() * sizeof(float), vertices_.ToArray(), isStatic_ ? BufferUsage.StaticDraw : BufferUsage.DynamicDraw);
                    Gl.VertexAttribPointer(Shader.ATTRIB_VERTICES_POS, 3, VertexAttribType.Float, false, 0, IntPtr.Zero);
                    Gl.EnableVertexAttribArray(Shader.ATTRIB_VERTICES_POS);
                }
                if (colors_.Count() > 0)
                {

                    Gl.BindBuffer(BufferTarget.ArrayBuffer, vboID_[1]);
                    Gl.BufferData(BufferTarget.ArrayBuffer, (uint)colors_.Count() * sizeof(float), colors_.ToArray(), isStatic_ ? BufferUsage.StaticDraw : BufferUsage.DynamicDraw);
                    Gl.VertexAttribPointer(Shader.ATTRIB_COLOR_POS, 4, VertexAttribType.Float, false, 0, IntPtr.Zero);
                    Gl.EnableVertexAttribArray(Shader.ATTRIB_COLOR_POS);
                }
                if (indices_.Count() > 0)
                {

                    Gl.BindBuffer(BufferTarget.ElementArrayBuffer, vboID_[2]);
                    Gl.BufferData(BufferTarget.ElementArrayBuffer, (uint)indices_.Count() * sizeof(float), indices_.ToArray(), isStatic_ ? BufferUsage.StaticDraw : BufferUsage.DynamicDraw);
                }
                if (normals_.Count() > 0)
                {
                    Gl.BindBuffer(BufferTarget.ArrayBuffer, vboID_[3]);
                    Gl.BufferData(BufferTarget.ArrayBuffer, (uint)normals_.Count() * sizeof(float), normals_.ToArray(), isStatic_ ? BufferUsage.StaticDraw : BufferUsage.DynamicDraw);
                    Gl.VertexAttribPointer(Shader.ATTRIB_NORMAL, 3, VertexAttribType.Float, false, 0, IntPtr.Zero);
                    Gl.EnableVertexAttribArray(Shader.ATTRIB_NORMAL);
                }

                Gl.BindVertexArray(0);
                Gl.BindBuffer(BufferTarget.ElementArrayBuffer, 0);
                Gl.BindBuffer(BufferTarget.ArrayBuffer, 0);
            }
        }

        public void clear()
        {
            vertices_.Clear();
            colors_.Clear();
            indices_.Clear();
            normals_.Clear();
        }

        public void setDrawingType(PrimitiveType type)
        {
            drawingType_ = type;
        }

        public void draw()
        {
            if (indices_.Count() > 0 && vaoID_ != 0)
            {
                Gl.BindVertexArray(vaoID_);
                Gl.DrawElements(drawingType_, indices_.Count(), DrawElementsType.UnsignedInt, IntPtr.Zero);
                Gl.BindVertexArray(0);
            }
        }

        public float grid_size = 9.0f;

        private List<float> vertices_;
        private List<float> colors_;
        private List<uint> indices_;
        private List<float> normals_;

        private bool isStatic_;
        private bool is_init;

        private uint vaoID_;

        /*
        Vertex buffer IDs:
        - [0]: Vertices coordinates;
        - [1]: Colors;
        - [2]: Indices;
        - [3]: Normals
        */
        private uint[] vboID_ = new uint[4];

        private ShaderData shader;

        private float3 position_;
        private Matrix3x3 rotation_;

        PrimitiveType drawingType_;
    };

}
