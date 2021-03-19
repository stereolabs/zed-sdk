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

            shaderBasic = new ShaderData();
            shaderBasic.it = new Shader(Shader.VERTEX_SHADER, Shader.FRAGMENT_SHADER);
            shaderBasic.MVP_Mat = Gl.GetUniformLocation(shaderBasic.it.getProgramId(), "u_mvpMatrix");

            setRenderCameraProjection(param, 0.5f, 20);

            image_handler.initialize();
            BBox_edges.init();
            BBox_edges.setDrawingType(PrimitiveType.Lines);
            BBox_faces.init();
            BBox_faces.setDrawingType(PrimitiveType.Quads);

            objectsName = new List<ObjectClassName>();

            Gl.Disable(EnableCap.DepthTest);

            available = true;

        }

        public void update(Mat image, Objects objects)
        {
            image_handler.pushNewImage(image);

            //if(objects.isNew == 0) return;

            BBox_faces.clear();
            BBox_edges.clear();
            objectsName.Clear();

            // For each object
            for (int idx = 0; idx < objects.numObject; idx++)
            {
                sl.ObjectData obj = objects.objectData[idx];

                // Only show tracked objects
                if (renderObject(obj))
                {
                    List<Vector3> bb_ = new List<Vector3>();
                    bb_.AddRange(obj.boundingBox);
                    if (bb_.Count > 0)
                    {
                        float4 clr_id = generateColorClass(obj.id);
                        float4 clr_class = generateColorClass((int)obj.label);

                        if (obj.objectTrackingState != sl.OBJECT_TRACKING_STATE.OK)
                            clr_id = clr_class;
                        else
                            createIDRendering(obj.position, clr_id, obj.id);

                        createBboxRendering(bb_, clr_id);
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

            draw();
        }

        public void draw()
        {
            image_handler.draw();

            Gl.UseProgram(shaderBasic.it.getProgramId());
            Gl.UniformMatrix4f(shaderBasic.MVP_Mat, 1, true, projection_);
            Gl.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
            Gl.LineWidth(1.5f);
            BBox_edges.draw();
            Gl.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
            BBox_faces.draw();

            Gl.UseProgram(0);
        }

        sl.float4 generateColorClass(int idx) {

            int offset = Math.Max(0, idx % 5);
            sl.float4 color = new float4();
            color.x = id_colors[offset, 0];
            color.y = id_colors[offset, 1];
            color.z = id_colors[offset, 2];
            color.w = 1.0f;
            return color;
        }

        float[,] id_colors = new float[5,3]{

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

        float4 getColorClass(int idx) {
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

        void printText()
        {
            Resolution res = new Resolution(1280, 1080);
            foreach(ObjectClassName obj in objectsName)
            {
                Gl.WindowPos2(obj.position.X, obj.position.Y);

            }
        }

        public void exit()
        {
            if (currentInstance != null)
            {
                image_handler.close();
                available = false;
                BBox_edges.clear();
                BBox_faces.clear();
            }
        }

        bool available;

        Matrix4x4 projection_;

        ImageHandler image_handler;

        ShaderData shaderBasic;

        List<ObjectClassName> objectsName;
        Simple3DObject BBox_edges;
        Simple3DObject BBox_faces;

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

            Gl.Uniform1(Gl.GetUniformLocation(shaderImage.it.getProgramId(), "revert"), 1);
            Gl.Uniform1(Gl.GetUniformLocation(shaderImage.it.getProgramId(), "rgbflip"), 1);

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

            shader.it = new Shader(Shader.VERTEX_SHADER, Shader.FRAGMENT_SHADER);
            shader.MVP_Mat = Gl.GetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");

            vertices_ = new List<float>();
            colors_ = new List<float>();
            indices_ = new List<uint>();

            is_init = true;
        }

        public bool isInit()
        {
            return is_init;
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
            clr.w = 0.2f;
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
            for (int i = 0; i<current_pts.Count; i++)
            {
                addPt(new float3(current_pts[i].X, current_pts[i].Y, current_pts[i].Z));
                clr.w = (i == 2 || i == 3) ? 0.0f : 0.2f;
                addClr(clr);
            }

            uint[] boxLinks = new uint[] { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5 };
            for (int i = 0; i<boxLinks.Length; i += 2)
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
            clr.w = 0.25f;
            foreach (Vector3 it in pts)
                addPoint(new float3(it.X, it.Y, it.Z), clr);
        }

        void addQuad(List<Vector3> quad_pts, float alpha1, float alpha2, float4 clr)
        { // To use only with 4 points
            for (int i = 0; i<quad_pts.Count; ++i)
            {
                addPt(new float3(quad_pts[i].X, quad_pts[i].Y, quad_pts[i].Z));
                clr.w = (i< 2 ? alpha1 : alpha2);
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
            float alpha = 0.25f;

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

        public float grid_size = 15.0f;

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
        */
        private uint[] vboID_ = new uint[3];

        private ShaderData shader;

        private float3 position_;
        private Matrix3x3 rotation_;

        PrimitiveType drawingType_;
    };




}
