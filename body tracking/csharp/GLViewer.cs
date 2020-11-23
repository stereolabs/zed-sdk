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
            image_handler = new ImageHandler(res);
            available = false;
            currentInstance = this;

            bones = new Simple3DObject();
            joints = new Simple3DObject();

            BBox_edges = new Simple3DObject();
            BBox_faces = new Simple3DObject();
        }

        public bool isAvailable()
        {
            return available;
        }


        public void init(CameraParameters param)
        {
            Gl.Enable(EnableCap.FramebufferSrgb);

            shaderSK = new ShaderData();
            shaderSK.it = new Shader(Shader.SK_VERTEX_SHADER, Shader.SK_FRAGMENT_SHADER);
            shaderSK.MVP_Mat = Gl.GetUniformLocation(shaderSK.it.getProgramId(), "u_mvpMatrix");

            shaderBbox = new ShaderData();
            shaderBbox.it = new Shader(Shader.VERTEX_SHADER, Shader.FRAGMENT_SHADER);
            shaderBbox.MVP_Mat = Gl.GetUniformLocation(shaderBbox.it.getProgramId(), "u_mvpMatrix");

            setRenderCameraProjection(param, 0.5f, 20);

            image_handler.initialize();
            bones.init();
            bones.setDrawingType(PrimitiveType.Quads);
            joints.init();
            joints.setDrawingType(PrimitiveType.Quads);
            BBox_edges.init();
            BBox_edges.setDrawingType(PrimitiveType.Lines);
            BBox_faces.init();
            BBox_faces.setDrawingType(PrimitiveType.Quads);

            objectsName = new List<ObjectClassName>();

            available = true;
        }

        public void update(Mat image, Objects objects)
        {
            image_handler.pushNewImage(image);

            bones.clear();
            joints.clear();

            BBox_faces.clear();
            BBox_edges.clear();

            if (Keyboard.IsKeyDown(Key.B)) showbbox = !showbbox;

            // For each object
            for (int idx = 0; idx < objects.numObject; idx++)
            {
                sl.ObjectData obj = objects.objectData[idx];

                // Only show tracked objects
                if (renderObject(obj))
                {
                    List<Vector3> bb_ = new List<Vector3>();
                    bb_.AddRange(obj.boundingBox);
                    float4 clr_id = generateColorClass(obj.id);
                    float4 clr_class = generateColorClass((int)obj.label);
                    Vector3[] keypoints = obj.keypoints;

                    if (showbbox)
                    {
                        if (obj.objectTrackingState != sl.OBJECT_TRACKING_STATE.OK)
                            clr_id = clr_class;
                        else
                            createIDRendering(obj.position, clr_id, obj.id);

                        createBboxRendering(bb_, clr_id);
                    }

                    if (keypoints.Length > 0)
                    {
                        foreach(var limb in BODY_BONES)
                        {
                            Vector3 kp_1 = keypoints[getIdx(limb.Item1)];
                            Vector3 kp_2 = keypoints[getIdx(limb.Item2)];

                            float norm_1 = kp_1.Length();
                            float norm_2 = kp_2.Length();

                            if (!float.IsNaN(norm_1) && norm_1 > 0 && !float.IsNaN(norm_2) && norm_2 > 0){
                                bones.addCylinder(new float3(kp_1.X, kp_1.Y, kp_1.Z), new float3(kp_2.X, kp_2.Y, kp_2.Z), clr_id);
                            }
                        }
                        for (int i = 0; i < (int)BODY_PARTS.LAST; i++)
                        {
                            Vector3 kp = keypoints[i];
                            float norm = kp.Length();
                            if (!float.IsNaN(norm) && norm > 0)
                            {
                                joints.addSphere(new float3(kp.X, kp.Y, kp.Z), clr_id);
                            }
                        }
                    }
                }
            }
        }

        void createBboxRendering(List<Vector3> bb_, float4 bbox_clr)
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
        }

        public void render()
        {
            BBox_edges.pushToGPU();
            BBox_faces.pushToGPU();
            bones.pushToGPU();
            joints.pushToGPU();

            draw();
        }

        public void draw()
        {
            Gl.Disable(EnableCap.DepthTest);
            image_handler.draw();

            Gl.Enable(EnableCap.DepthTest);
            Gl.UseProgram(shaderSK.it.getProgramId());
            Gl.UniformMatrix4f(shaderSK.MVP_Mat, 1, true, projection_);
            bones.draw();
            joints.draw();

            Gl.UseProgram(shaderBbox.it.getProgramId());
            Gl.UniformMatrix4f(shaderBbox.MVP_Mat, 1, true, projection_);
            Gl.LineWidth(1.5f);
            BBox_edges.draw();
            BBox_faces.draw();

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
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_SHOULDER, BODY_PARTS.RIGHT_HIP),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_HIP, BODY_PARTS.RIGHT_KNEE),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.RIGHT_KNEE, BODY_PARTS.RIGHT_ANKLE),
            Tuple.Create<BODY_PARTS, BODY_PARTS>(BODY_PARTS.LEFT_SHOULDER, BODY_PARTS.LEFT_HIP),
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
                image_handler.close();
                available = false;
                bones.clear();
                joints.clear();
            }
        }

        bool available;

        Matrix4x4 projection_;

        ImageHandler image_handler;

        ShaderData shaderSK;
        ShaderData shaderBbox;

        List<ObjectClassName> objectsName;

        Simple3DObject BBox_edges;
        Simple3DObject BBox_faces;
        Simple3DObject bones;
        Simple3DObject joints;

        GLViewer currentInstance;

        bool showbbox = false;
    }

    class ImageHandler
    {
        public ImageHandler(Resolution res) {
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
            "	float ambientStrength = 0.3;\n",
            "	vec3 ambient = ambientStrength * lightColor;\n",
            "	vec3 norm = normalize(b_normal); \n",
            "	vec3 lightDir = normalize(lightPosition - b_position);\n",
            "	float diffuse = (1 - ambientStrength) * max(dot(b_normal, lightDir), 0.0);\n",
            "   out_Color = vec4(b_color.rgb * (diffuse + ambient), 1);\n",
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
            int NB_SEG = 32;
            float scale_seg = (float)1 / (float)NB_SEG;

            for (int j = 0; j < NB_SEG; j++)
            {
                float i = (float)2 * PI * ((float)j * scale_seg);
                float i1 = (float)2 * PI * ((float)(j+1) * scale_seg);
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

        private float3 position_;
        private Matrix3x3 rotation_;

        PrimitiveType drawingType_;
    };




}
