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

    public struct UserAction
    {
        public bool pressSpace;
        public bool hit;
        public Vector2 hitCoord;

        public void clear()
        {
            pressSpace = false;
            hit = false;
        }
    };

    class GLViewer
    {
        public GLViewer(Resolution res)
        {
            image_handler = new ImageHandler(res);
            available = false;
            currentInstance = this;
            pose = Matrix4x4.Identity;
            userAction = new UserAction();
            tracking_state = POSITIONAL_TRACKING_STATE.OFF;

            meshObject = new MeshObject();
        }

        public bool isAvailable()
        {
            return available;
        }


        public void init(CameraParameters param)
        {
            image_handler.initialize();

            Gl.Enable(EnableCap.FramebufferSrgb);

            // Create the rendering camera
            setRenderCameraProjection(param, 0.1f, 50);

            meshObject.alloc();

            userAction.hitCoord = new Vector2(0.5f, 0.5f);
            available = true;
        }

        public void updateImageAndState(Mat image, Pose pose_, POSITIONAL_TRACKING_STATE track_state)
        {
            if (available)
            {
                image_handler.pushNewImage(image);
                
                pose = Matrix4x4.Identity;
                pose = Matrix4x4.Transform(pose, pose_.rotation);
                pose.Translation = pose_.translation;
                pose = Matrix4x4.Transpose(pose);
                tracking_state = track_state;

                wnd_h = (int)image.GetResolution().height;
                wnd_w = (int)image.GetResolution().width;
            }
            newData = true;
        }

        public void updateMesh(Vector3[] vertices, int[] triangles, int nbVertices, int nbTriangles, PLANE_TYPE type, Vector3[] bounds, UserAction userAction)
        {
            meshObject.updateMesh(vertices, triangles, nbVertices, nbTriangles, bounds);
            meshObject.type = type;
            this.userAction = userAction;
        }

        public void render()
        {
            if (available)
            {
                update();
                draw();
            }
        }

        public void update()
        {
            if (newData)
            {
                meshObject.pushToGPU();
                newData = false;
            }
        }

        public void draw()
        {
            if (available)
            {
                image_handler.draw();

                // If the Positional tracking is good, we can draw the mesh over the current image
                Gl.Disable(EnableCap.Texture2d);
                if (tracking_state == POSITIONAL_TRACKING_STATE.OK)
                {
                    // Send the projection and the Pose to the GLSL shader to make the projection of the 2D image.
                    Matrix4x4.Invert(pose, out Matrix4x4 pose_inv);
                    Matrix4x4 vpMatrix = projection_ * pose_inv;
                    Gl.UseProgram(meshObject.shader.it.getProgramId());
                    Gl.UniformMatrix4f((int)meshObject.shader.MVP_Mat, 1, true, vpMatrix);

                    float3 clrPlane = getPlaneColor(meshObject.type);
                    Gl.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
                    Gl.Uniform3f(meshObject.shader.shColorLoc, 1, clrPlane);
                    meshObject.draw();
                    
                    Gl.LineWidth(0.5f);
                    Gl.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);
                    Gl.Uniform3f(meshObject.shader.shColorLoc, 1, clrPlane);
                    meshObject.draw();
                    Gl.UseProgram(0);
                    Gl.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
                }


                float cx = userAction.hitCoord.X * 2.0f - 1.0f;
                float cy = (userAction.hitCoord.Y * 2.0f - 1.0f) * -1.0f;

                float lx = 0.02f;
                float ly = lx * (wnd_w/(1.0f * wnd_h));

                Gl.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);
                Gl.LineWidth(2.0f);
                Gl.Color3(0.2f, 0.45f, 0.9f);
                Gl.Begin(PrimitiveType.Lines);
                Gl.Vertex3(cx - lx, cy, 0.0f);
                Gl.Vertex3(cx + lx, cy, 0.0f);
                Gl.Vertex3(cx, cy - ly, 0.0f);
                Gl.Vertex3(cx, cy + ly, 0.0f);
                Gl.End();
                Gl.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
            }
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
            projection_.M23 = -(2.0f * (((int)camParams.resolution.height -1.0f * camParams.cy) / (int)camParams.resolution.height) -1.0f); //Vertical offset.
            projection_.M24 = 0;

            projection_.M31 = 0;
            projection_.M32 = 0;

            projection_.M41 = 0;
            projection_.M42 = 0;          
        }

        public float3 getPlaneColor(PLANE_TYPE type)
        {
            float3 clr = new float3();

            switch (type)
            {
                case PLANE_TYPE.HIT_HORIZONTAL:
                    clr = new float3(0.65f, 0.95f, 0.35f);
                    break;
                case PLANE_TYPE.HIT_VERTICAL:
                    clr = new float3(0.95f, 0.35f, 0.65f);
                    break;
                case PLANE_TYPE.HIT_UNKNOWN:
                    clr = new float3(0.35f, 0.65f, 0.95f);
                    break;
                default:
                    clr = new float3(0.65f, 0.95f, 0.35f);
                    break;
            }
            return clr;
        }

        public void exit()
        {
            if (currentInstance != null)
            {
                image_handler.close();
                available = false;
            }
        }

        bool available;

        // OpenGL camera projection matrix
        Matrix4x4 projection_;

        Matrix4x4 pose;
        sl.POSITIONAL_TRACKING_STATE tracking_state;
        UserAction userAction;

        bool newData;
        int wnd_w = 0;
        int wnd_h = 0;

        MeshObject meshObject;
        ImageHandler image_handler;

        GLViewer currentInstance;
    }

    class MeshObject
    {
        public uint vaoID_;
        public uint[] vboID_;
        public int currentFC;
        public bool needUpdate;

        public PLANE_TYPE type;
        public ShaderData shader;

        public Vector3[] vertices_;
        public float[] edgeDist_;
        public int[] triangles_;

        public MeshObject()
        {
            vaoID_ = 0;
            vboID_ = new uint[3];
            currentFC = 0;
            needUpdate = false;

            type = PLANE_TYPE.FLOOR;
            shader = new ShaderData();

        }

        public void alloc()
        {
            vaoID_ = Gl.GenVertexArray();
            Gl.GenBuffers(vboID_);
            shader.it = new Shader(Shader.MESH_VERTEX_SHADER, Shader.MESH_FRAGMENT_SHADER);
            shader.MVP_Mat = Gl.GetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");
            shader.shColorLoc = Gl.GetUniformLocation(shader.it.getProgramId(), "u_color");
        }

        public void updateMesh(Vector3[] vertices, int[] triangles, int nbVertices, int nbTriangles, Vector3[] bounds)
        {
            if (!needUpdate)
            {
                vertices_ = new Vector3[nbVertices / 3];
                Array.Copy(vertices, vertices_, nbVertices / 3);
                triangles_ = new int[nbTriangles];
                Array.Copy(triangles, triangles_, nbTriangles);

                edgeDist_ = new float[vertices_.Length];

                for (int i = 0; i < edgeDist_.Count(); i++)
                {
                    float dMin = float.MaxValue;
                    Vector3 vCurrent = vertices[i];
                    for (int j = 0; j < bounds.Length; j++)
                    {
                        float distCurrent = Vector3.Distance(vCurrent, bounds[j]);
                        if (distCurrent < dMin)
                        {
                            dMin = distCurrent;
                        }
                    }
                    edgeDist_[i] = dMin >= 0.002f ? 0.4f : 0.0f;
                }
                needUpdate = true;
            }
        }

        public void pushToGPU()
        {
            if (vaoID_ == 0)
            {
                vaoID_ = Gl.GenVertexArray();
                Gl.GenBuffers(vboID_);
            }

            if (needUpdate)
            {
                Gl.BindVertexArray(vaoID_);

                if(vertices_.Count() > 0)
                {
                    Gl.BindBuffer(BufferTarget.ArrayBuffer, vboID_[0]);
                    Gl.BufferData(BufferTarget.ArrayBuffer, (uint)vertices_.Length * sizeof(float) * 3, vertices_, BufferUsage.DynamicDraw);
                    Gl.VertexAttribPointer(Shader.ATTRIB_VERTICES_POS, 3, VertexAttribType.Float, false, 0, IntPtr.Zero);
                    Gl.EnableVertexAttribArray(Shader.ATTRIB_VERTICES_POS);
                }
                if (edgeDist_.Count() > 0)
                {
                    Gl.BindBuffer(BufferTarget.ArrayBuffer, vboID_[1]);
                    Gl.BufferData(BufferTarget.ArrayBuffer, (uint)edgeDist_.Length * sizeof(float), edgeDist_, BufferUsage.DynamicDraw);
                    Gl.VertexAttribPointer(Shader.ATTRIB_VERTICES_DIST, 1, VertexAttribType.Float, false, 0, IntPtr.Zero);
                    Gl.EnableVertexAttribArray(Shader.ATTRIB_VERTICES_DIST);
                }
                if (triangles_.Count() > 0)
                {
                    Gl.BindBuffer(BufferTarget.ElementArrayBuffer, vboID_[2]);
                    Gl.BufferData(BufferTarget.ElementArrayBuffer, (uint)triangles_.Length * sizeof(int), triangles_, BufferUsage.DynamicDraw);

                    currentFC = triangles_.Count();
                }

                Gl.BindVertexArray(0);
                Gl.BindBuffer(BufferTarget.ElementArrayBuffer, 0);
                Gl.BindBuffer(BufferTarget.ArrayBuffer, 0);
                needUpdate = false;
            }
        }

        public void draw()
        {
            if (currentFC > 0 && vaoID_ != 0)
            {
                Gl.BindVertexArray(vaoID_);
                Gl.DrawElements(PrimitiveType.Triangles, currentFC, DrawElementsType.UnsignedInt, IntPtr.Zero);
                Gl.BindVertexArray(0);
            }
        }
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
            Gl.Enable(EnableCap.Texture2d);
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

        public static readonly string[] MESH_VERTEX_SHADER = new string[] {
            "#version 330 core\n",
            "layout(location = 0) in vec3 in_vertex;\n",
            "layout(location = 1) in float in_dist;\n",
            "uniform mat4 u_mvpMatrix;\n",
            "uniform vec3 u_color;\n",
            "out vec3 b_color;\n",
            "out float distance;\n",
            "void main() {\n",
            "   b_color = u_color;\n",
            "   distance = in_dist;\n",
            "   gl_Position = u_mvpMatrix * vec4(in_vertex, 1);\n",
            "}"
        };

        public static readonly string[] MESH_FRAGMENT_SHADER = new string[] {
            "#version 330 core\n",
            "in vec3 b_color;\n",
            "in float distance;\n",
            "layout(location = 0) out vec4 color;\n",
            "void main() {\n",
            "   color = vec4(b_color, distance);\n",
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
        public static uint ATTRIB_VERTICES_DIST = 1;

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
        public int shColorLoc;
    };
}
