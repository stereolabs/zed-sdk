using OpenGL;
using OpenGL.CoreUI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace sl
{
    class GLViewer
    {
        public GLViewer()
        {
            floor_grid = new Simple3DObject(true);
            zedModel = new Simple3DObject(false);
            zedPath = new Simple3DObject(false);
            vecPath = new List<float3>();
            cam_pose = new Matrix4x4();

            available = false;
            currentInstance = this;

            clearInputs();
        }

        public bool isAvailable()
        {
            return available;
        }


        public void init(CameraParameters param, MODEL cameraModel)
        {

            mainShader = new ShaderData();
            mainShader.it = new Shader(Shader.VERTEX_SHADER, Shader.FRAGMENT_SHADER);
            mainShader.MVP_Mat = Gl.GetUniformLocation(mainShader.it.getProgramId(), "u_mvpMatrix");

            shaderLine = new ShaderData();
            shaderLine.it = new Shader(Shader.VERTEX_SHADER, Shader.FRAGMENT_SHADER);
            shaderLine.MVP_Mat = Gl.GetUniformLocation(shaderLine.it.getProgramId(), "u_mvpMatrix");

            //Create Camera
            camera_ = new CameraGL(new Vector3(0f, 1f, 1f), new Vector3(0, -1, -4f), Vector3.UnitY);
            camera_.setOffsetFromPosition(new Vector3(0, 0, 1.0f));

            floor_grid.init();
            floor_grid.setDrawingType(PrimitiveType.Lines);

            zedModel.init();
            zedModel.setDrawingType(PrimitiveType.Triangles);
            zedPath.init();

            float limit = 20;
            float4 clr1 = new float4();
            clr1.x = 0.85f;
            clr1.y = 0.87f;
            clr1.z = 0.88f;
            clr1.w = 1f;

            float4 clr2 = new float4();
            clr2.x = 0.42f;
            clr2.y = 0.48f;
            clr2.z = 0.53f;
            clr2.w = 1f;

            float height = 0.0f;
            for (int i = (int)(limit * -5); i <= (int)(limit * 5); i++)
            {
                float i_f = i / 5.0f;
                if ((i % 5) == 0)
                {
                    addVert(ref floor_grid, i_f, limit, height, clr2);
                }
                else
                {
                    addVert(ref floor_grid, i_f, limit, height, clr1);
                }
            }
            floor_grid.pushToGPU();
            zedPath.setDrawingType(PrimitiveType.LineStrip);

            Model3D model = new Model3D();

            switch (cameraModel)
            {
                case MODEL.ZED:
                    model = new Model3D_ZED();
                    break;
                case MODEL.ZED2:
                    model = new Model3D_ZED2();
                    break;
                case MODEL.ZED_M:
                    model = new Model3D_ZED();
                    break;
            }

            foreach(ModelPart part in model.part)
            {
                fillZED(part.nb_triangles, model.vertices, part.triangles, part.color, ref zedModel);
            }

            zedModel.pushToGPU();
            updateZEDposition = false;

            available = true;
        }

        public void updateData(Pose zedRT)
        {
            vecPath.Add(new float3(zedRT.translation.X, zedRT.translation.Y, zedRT.translation.Z));
            zedModel.setRT(zedRT.translation, zedRT.rotation);

            updateZEDposition = true;
        }

        void fillZED(int nb_tri, double[] vertices, List<int> triangles, float3 color, ref Simple3DObject zed_camera)
        {
            for (int p = 0; p < nb_tri * 3; p = p + 3)
            {
                int index = triangles[p] - 1;
                float4 clr = new float4();
                clr.x = color.x; clr.y = color.y; clr.z = color.z; clr.w = 1;
                zed_camera.addPoint(new float3((float)vertices[index * 3], (float)vertices[index * 3 + 1], (float)vertices[index * 3 + 2]), clr);
                index = triangles[p + 1] - 1;
                zed_camera.addPoint(new float3((float)vertices[index * 3], (float)vertices[index * 3 + 1], (float)vertices[index * 3 + 2]), clr);
                index = triangles[p + 2] - 1;
                zed_camera.addPoint(new float3((float)vertices[index * 3], (float)vertices[index * 3 + 1], (float)vertices[index * 3 + 2]), clr);
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

        public void update()
        {
            if (updateZEDposition)
            {
                zedPath.clear();
                float3 clr = new float3(0.1f, 0.5f, 0.9f);
                for (int i = 1; i < vecPath.Count(); i++)
                {
                    float fade = (i * 1.0f) / vecPath.Count();
                    float4 new_color = new float4();
                    new_color.x = clr.x * fade; new_color.y = clr.y * fade; new_color.z = clr.z * fade; new_color.w = fade;
                    zedPath.addPoint(vecPath[i], new_color);
                }
                zedPath.pushToGPU();
                updateZEDposition = false;
            }
        }

        public void render()
        {
            camera_.update();

            update();
            draw();

            //clearInputs();
        }

        public void keyEventFunction(NativeWindowKeyEventArgs e)
        {
            if (e.Key == KeyCode.R)
            {
                camera_.setPosition(new Vector3(0f, 1f, 1f));
                camera_.setDirection(new Vector3(0, -1, -4f), Vector3.UnitY);
            }
            if (e.Key == KeyCode.Up || e.Key == KeyCode.Z)
            {
                camera_.translate(camera_.getForward() * -1f * MOUSE_ZOOM_SENSITIVITY);
            }
            if (e.Key == KeyCode.Down || e.Key == KeyCode.S)
            {
                camera_.translate(camera_.getForward() * 1f * MOUSE_ZOOM_SENSITIVITY);
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
                camera_.translate(camera_.getUp() *  (float)mouseMotion_[1] * MOUSE_T_SENSITIVITY);
                camera_.translate(camera_.getRight() *  -(float)mouseMotion_[0] * MOUSE_T_SENSITIVITY);
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
            currentInstance.mouseMotion_[1] = currentInstance.previousMouseMotion_[1] - y;
            currentInstance.previousMouseMotion_[0] = x;
            currentInstance.previousMouseMotion_[1] = y;
        }

        public void draw()
        {
            Matrix4x4 vpMatrix = camera_.getViewProjectionMatrix();

            Gl.UseProgram(shaderLine.it.getProgramId());
            Gl.UniformMatrix4f(shaderLine.MVP_Mat, 1, true, vpMatrix);
            Gl.LineWidth(1.0f);
            floor_grid.draw();
            Gl.UseProgram(0);

            Gl.UseProgram(mainShader.it.getProgramId());
            Gl.UniformMatrix4f(mainShader.MVP_Mat, 1, true, vpMatrix);

            Gl.LineWidth(2.0f);
            zedPath.draw();

            Matrix4x4 t = Matrix4x4.Transpose(zedModel.getModelMatrix()) * Matrix4x4.Transpose(vpMatrix);
            Gl.UniformMatrix4f(mainShader.MVP_Mat, 1, false, t);
            zedModel.draw();

            Gl.UseProgram(0);
        }

        sl.float4 generateColorClass(int idx)
        {

            int offset = Math.Max(0, idx % 5);
            sl.float4 color = new float4();
            color.x = id_colors[offset, 0];
            color.y = id_colors[offset, 1];
            color.z = id_colors[offset, 2];
            color.w = 1.0f;
            return color;
        }

        float[,] id_colors = new float[5, 3]{

            {.231f, .909f, .69f},
            {.098f, .686f, .816f},
            {.412f, .4f, .804f},
            {1, .725f, .0f},
            {.989f, .388f, .419f}
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

        bool renderObject(ObjectData i) {
            return (i.objectTrackingState == OBJECT_TRACKING_STATE.OK || i.objectTrackingState == OBJECT_TRACKING_STATE.OFF);
        }

        private void setRenderCameraProjection(CameraParameters camParams, float znear, float zfar)
            {
                float PI = 3.141592653f;
                // Just slightly up the ZED camera FOV to make a small black border
                float fov_y = (camParams.vFOV+0.5f) *PI / 180;
                float fov_x = (camParams.hFOV+0.5f) * PI / 180;

                projection_.M11 = 1.0f / (float)Math.Tan(fov_x * 0.5f);
                projection_.M22 = 1.0f / (float)Math.Tan(fov_y * 0.5f);
                projection_.M33 = -(zfar + znear) / (zfar - znear);
                projection_.M43 = -1;
                projection_.M34 = -(2.0f * zfar * znear) / (zfar - znear);
                projection_.M44 = 0;

                projection_.M12 = 0;
                projection_.M13 = 2.0f * (((int)camParams.resolution.width - 1.0f * camParams.cx) / (int)camParams.resolution.width) -1.0f; //Horizontal offset.
                projection_.M14 = 0;

                projection_.M21 = 0;
                projection_.M22 = 1.0f / (float)Math.Tan(fov_y * 0.5f); //Vertical FoV.
                projection_.M23 = -(2.0f * (((int)camParams.resolution.height -1.0f * camParams.cy) / (int)camParams.resolution.height) -1.0f); //Vertical offset.
                projection_.M24 = 0;

                projection_.M31 = 0;
                projection_.M32 = 0;

                projection_.M41 = 0;
                projection_.M42 = 0;
            }


        void clearInputs()
        {
            mouseMotion_[0] = mouseMotion_[1] = 0;
        }

        public void exit()
        {
            if (currentInstance != null)
            {
                available = false;
            }
        }

        bool available;

        int[] mouseCurrentPosition_ = new int[2];
        int[] mouseMotion_ = new int[2];
        int[] previousMouseMotion_ = new int[2];

        const float MOUSE_R_SENSITIVITY = 0.004f;
        const float MOUSE_T_SENSITIVITY = 0.04f;
        const float MOUSE_UZ_SENSITIVITY = 0.75f;
        const float MOUSE_DZ_SENSITIVITY = 1.25f;
        const float MOUSE_ZOOM_SENSITIVITY = 0.065f;

        List<float3> vecPath;
        bool updateZEDposition;

        CameraGL camera_;
        Matrix4x4 projection_;

        ShaderData shaderLine;
        ShaderData mainShader;
        Matrix4x4 cam_pose;

        Simple3DObject floor_grid;
        Simple3DObject zedModel;
        Simple3DObject zedPath;
        
        GLViewer currentInstance;
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
            setProjection(60, 60, 0.1f, 100);
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

        public Matrix4x4 getViewProjectionMatrix(){ return vpMatrix_; }

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
            if (Vector3.Zero != w) {

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

        public Vector3 getVertical(){ return vertical_;}

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
            "uniform bool revert;\n",
            "uniform bool rgbflip;\n",
            "void main() {\n",
            "   vec2 scaler = revert?vec2(UV.x,1.f - UV.y):vec2(UV.x,UV.y);\n",
            "   vec3 rgbcolor = rgbflip?vec3(texture(texImage, scaler).zyx):vec3(texture(texImage, scaler).xyz);\n",
            "   float gamma = 1.0/1.65;\n",
            "   vec3 color_rgb = pow(rgbcolor, vec3(1.0/gamma));;\n",
            "   color = vec4(color_rgb,1);\n",
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
            " float gamma = 2.2;\n",
            "   out_Color = b_color;//pow(b_color, vec4(1.0/gamma));;\n",
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
        public Simple3DObject(bool isStatic)
        {
            is_init = false;
            isStatic_ = isStatic;
            position_ = new Vector3();
            rotation_ = new Quaternion();
        }

        public void init()
        {
            vaoID_ = 0;

            shader.it = new Shader(Shader.VERTEX_SHADER, Shader.FRAGMENT_SHADER);
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
        public Matrix4x4 getModelMatrix()
        {
            Matrix4x4 transform = Matrix4x4.Identity;
            transform = Matrix4x4.CreateFromQuaternion(rotation_);
            transform.Translation = position_;
            return Matrix4x4.Transpose(transform);
        }

        public void setRT(Vector3 translation, Quaternion rotation)
        {
            position_ = translation;
            rotation_ = rotation;
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
            clr.w = 0.4f;
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
                clr.w = (i == 2 || i == 3) ? 0.0f : 0.4f;
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

        public void createFrustum(CameraParameters param)
        {

            // Create 3D axis
            /*Simple3DObject it = new Simple3DObject();
            it.init();*/

            //float Z_ = -150;
            float Z_ = -0.15f;
            float3 cam_0 = new float3(0, 0, 0);
            float3 cam_1, cam_2, cam_3, cam_4;

            float fx_ = 1.0f / param.fx;
            float fy_ = 1.0f / param.fy;

            cam_1.z = Z_;
            cam_1.x = (0 - param.cx) * Z_ * fx_;
            cam_1.y = (0 - param.cy) * Z_ * fy_;

            cam_2.z = Z_;
            cam_2.x = ((int)param.resolution.width - param.cx) * Z_ * fx_;
            cam_2.y = (0 - param.cy) * Z_ * fy_;

            cam_3.z = Z_;
            cam_3.x = ((int)param.resolution.width - param.cx) * Z_ * fx_;
            cam_3.y = ((int)param.resolution.height - param.cy) * Z_ * fy_;

            cam_4.z = Z_;
            cam_4.x = (0 - param.cx) * Z_ * fx_;
            cam_4.y = ((int)param.resolution.height - param.cy) * Z_ * fy_;

            float4 clr = new float4();
            clr.x = 0.2f;
            clr.y = 0.5f;
            clr.z = 0.8f;
            clr.w = 1.0f;

            /*it.addTriangle(cam_0, cam_1, cam_2, clr);
            it.addTriangle(cam_0, cam_2, cam_3, clr);
            it.addTriangle(cam_0, cam_3, cam_4, clr);
            it.addTriangle(cam_0, cam_4, cam_1, clr);


            it.setDrawingType(PrimitiveType.Triangles);
            return it;*/

            addTriangle(cam_0, cam_1, cam_2, clr);
            addTriangle(cam_0, cam_2, cam_3, clr);
            addTriangle(cam_0, cam_3, cam_4, clr);
            addTriangle(cam_0, cam_4, cam_1, clr);
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

        float3 yAxis = new float3(0,1,0);

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

        for (double i = 0; i <= 2 * PI - 1; i += resolution)
        {
            v1 = rotationMatrix.multiply(new float3(m_radius * (float)Math.Cos(i), 0, m_radius * (float)Math.Sin(i))).add(startPosition);
            v2 = rotationMatrix.multiply(new float3(m_radius * (float)Math.Cos(i), m_height, m_radius * (float)Math.Sin(i))).add(startPosition);
            v4 = rotationMatrix.multiply(new float3(m_radius * (float)Math.Cos(i + 1), m_height, m_radius * (float)Math.Sin(i + 1))).add(startPosition);
            v3 = rotationMatrix.multiply(new float3(m_radius * (float)Math.Cos(i + 1), 0, m_radius * (float)Math.Sin(i + 1))).add(startPosition);

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

        public void addSphere(float3 position, float4 clr)
        {
            const float PI = 3.1415926f;

            float m_radius = 0.02f;
            float radiusInv = 1.0f / m_radius;

            int m_stackCount = 20;
            int m_sectorCount = 20;

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

                    v1 = new float3(m_radius * x * zr0, m_radius * y * zr0, m_radius * z0).add(position);
                    normal = new float3(x * zr0, y * zr0, z0);
                    addPoint(v1, clr);
                    addNormal(normal);

                    v2 = new float3(m_radius * x * zr1, m_radius * y * zr1, m_radius * z1).add(position);
                    normal = new float3(x * zr1, y * zr1, z1);
                    addPoint(v2, clr);
                    addNormal(normal);

                    lng = 2 * PI * (double)j / m_sectorCount;
                    x = (float)Math.Cos(lng);
                    y = (float)Math.Sin(lng);

                    v4 = new float3(m_radius * x * zr1, m_radius * y * zr1, m_radius * z1).add(position);
                    normal = new float3(x * zr1, y * zr1, z1);
                    addPoint(v4, clr);
                    addNormal(normal);

                    v3 = new float3(m_radius * x * zr0, m_radius * y * zr0, m_radius * z0).add(position);
                    normal = new float3(x * zr0, y * zr0, z0);
                    addPoint(v3, clr);
                    addNormal(normal);
                }
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

        private Vector3 position_;
        private Quaternion rotation_;

        PrimitiveType drawingType_;
    };
}
