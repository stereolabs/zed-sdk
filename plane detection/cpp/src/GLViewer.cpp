#include "GLViewer.hpp"

#include <cuda_gl_interop.h>

using namespace std;

void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    cout <<"[Sample]";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout<<" ";
    cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}

GLchar* MESH_VERTEX_SHADER =
"#version 330 core\n"
"layout(location = 0) in vec3 in_Vertex;\n"
"layout(location = 1) in float in_dist;\n"
"uniform mat4 u_mvpMatrix;\n"
"uniform vec3 u_color;\n"
"out vec3 b_color;\n"
"out float distance;\n"
"void main() {\n"
"   b_color = u_color;\n"
"   distance = in_dist;\n"
"   gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
"}";

GLchar* MESH_FRAGMENT_SHADER =
"#version 330 core\n"
"in vec3 b_color;\n"
"in float distance;\n"
"layout(location = 0) out vec4 color;\n"
"void main() {\n"
"   color = vec4(b_color,distance);\n"
"}";

GLchar* IMAGE_FRAGMENT_SHADER =
"#version 330 core\n"
" in vec2 UV;\n"
" out vec4 color;\n"
" uniform sampler2D texImage;\n"
" uniform bool revert;\n"
" uniform bool rgbflip;\n"
" void main() {\n"
"    vec2 scaler  =revert?vec2(UV.x,1.f - UV.y):vec2(UV.x,UV.y);\n"
"    vec3 rgbcolor = rgbflip?vec3(texture(texImage, scaler).zyx):vec3(texture(texImage, scaler).xyz);\n"
"    color = vec4(rgbcolor,1);\n"
"}";

GLchar* IMAGE_VERTEX_SHADER =
"#version 330\n"
"layout(location = 0) in vec3 vert;\n"
"out vec2 UV;"
"void main() {\n"
"   UV = (vert.xy+vec2(1,1))/2;\n"
"	gl_Position = vec4(vert, 1);\n"
"}\n";

MeshObject::MeshObject() {
    current_fc = 0;
    vaoID_ = 0;
    need_update = false;
}

MeshObject::~MeshObject() {
    need_update = false;
    current_fc = 0;
    vert.clear();
    edge_dist.clear();
    tri.clear();
    if(vaoID_) {
        glDeleteBuffers(3, vboID_);
        glDeleteVertexArrays(1, &vaoID_);
    }
}

void MeshObject::alloc(){
    glGenVertexArrays(1, &vaoID_);
    glGenBuffers(3, vboID_);
    shader.it = Shader(MESH_VERTEX_SHADER, MESH_FRAGMENT_SHADER);
    shader.MVP_Mat = glGetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");
    shader.shColorLoc = glGetUniformLocation(shader.it.getProgramId(), "u_color");
}

void MeshObject::updateMesh(std::vector<sl::float3> &vertices, std::vector<sl::uint3> &triangles, std::vector<int> &border) {
    if(!need_update) {
        vert = vertices;
        tri = triangles;

        edge_dist.resize(vertices.size());

        for (unsigned int i = 0; i < edge_dist.size(); i++) {
            float d_min = std::numeric_limits<float>::max();
            sl::float3 v_current = vertices[i];
            for (unsigned int j = 0; j < border.size(); j++) {
                float dist_current = sl::float3::distance(v_current, vertices[border[j]]);
                if(dist_current < d_min) {
                    d_min = dist_current;
                }
            }
            edge_dist[i] = d_min >= 0.002 ? 0.4 : 0.0f;
        }
        need_update = true;
    }
}

void MeshObject::pushToGPU() {
    if(need_update) {
        glBindVertexArray(vaoID_);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
        glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof(sl::float3), &vert[0], GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
        glBufferData(GL_ARRAY_BUFFER, edge_dist.size() * sizeof(float), &edge_dist[0], GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_DIST, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_DIST);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, tri.size() * sizeof(sl::uint3), &tri[0], GL_DYNAMIC_DRAW);

        current_fc = (int) tri.size() * 3;

        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        need_update = false;
    }
}

void MeshObject::draw() {
    if(current_fc && vaoID_) {
        glBindVertexArray(vaoID_);
        glDrawElements(GL_TRIANGLES, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}

GLViewer* currentInstance_ = nullptr;

GLViewer::GLViewer() :available(false) {
    currentInstance_ = this;
    pose.setIdentity();
    tracking_state = sl::POSITIONAL_TRACKING_STATE::OFF;
    new_data = false;
    user_action.clear();
}

GLViewer::~GLViewer() {
}


bool cudaSafeCall(cudaError_t err) {
    if(err != cudaSuccess) {
        printf("Cuda error [%d]: %s.\n", err, cudaGetErrorString(err));
        return false;
    }
    return true;
}

void GLViewer::exit() {
    if(available) {
        image_handler.close();
    }
    available = false;
}

bool GLViewer::isAvailable() {
    if(available)
        glutMainLoopEvent();
    return available;
}

void CloseFunc(void) { if(currentInstance_) currentInstance_->exit(); }

bool GLViewer::init(int argc, char **argv, sl::CameraParameters &camLeft, bool has_imu) {

    glutInit(&argc, argv);
    int wnd_w = glutGet(GLUT_SCREEN_WIDTH);
    int wnd_h = glutGet(GLUT_SCREEN_HEIGHT);
    int width = wnd_w * 0.9;
    int height = wnd_h * 0.9;
    if (width > camLeft.image_size.width && height > camLeft.image_size.height) {
        width = camLeft.image_size.width;
        height = camLeft.image_size.height;
    }
    
    glutInitWindowSize(width, height);
    glutInitWindowPosition(wnd_w * 0.05, wnd_h * 0.05);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutCreateWindow("ZED Plane Detection");
    
    reshapeCallback(width, height);

    GLenum err = glewInit();
    if (GLEW_OK != err){
        std::cout << "ERROR: glewInit failed: " << glewGetErrorString(err) << "\n";
        return true;
    }

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    bool status_ = image_handler.initialize(camLeft.image_size);
    if (!status_) {
        std::cout << "ERROR: Failed to initialized Image Renderer" << std::endl;
        return true;
    }
    
    // Create the rendering camera
    setRenderCameraProjection(camLeft, 0.1f, 50);

    mesh_object.alloc();

    // Set glut callback before start
    glutDisplayFunc(GLViewer::drawCallback);
    glutKeyboardUpFunc(GLViewer::keyReleasedCallback);
    glutMouseFunc(GLViewer::mouseButtonCallback);
    glutReshapeFunc(GLViewer::reshapeCallback);
    glutCloseFunc(CloseFunc);

    use_imu = has_imu;
    user_action.hit_coord = sl::float2(.5f, .5f);

    available = true;

    return false;
}

void GLViewer::setRenderCameraProjection(sl::CameraParameters params, float znear, float zfar) {
    // Just slightly up the ZED camera FOV to make a small black border
    float fov_y = (params.v_fov + 0.5f) * M_PI / 180.f;
    float fov_x = (params.h_fov + 0.5f) * M_PI / 180.f;

    camera_projection(0, 0) = 1.0f / tanf(fov_x * 0.5f);
    camera_projection(1, 1) = 1.0f / tanf(fov_y * 0.5f);
    camera_projection(2, 2) = -(zfar + znear) / (zfar - znear);
    camera_projection(3, 2) = -1;
    camera_projection(2, 3) = -(2.f * zfar * znear) / (zfar - znear);
    camera_projection(3, 3) = 0;

    camera_projection(0, 0) = 1.0f / tanf(fov_x * 0.5f); //Horizontal FoV.
    camera_projection(0, 1) = 0;
    camera_projection(0, 2) = 2.0f * ((params.image_size.width - 1.0f * params.cx) / params.image_size.width) - 1.0f; //Horizontal offset.
    camera_projection(0, 3) = 0;

    camera_projection(1, 0) = 0;
    camera_projection(1, 1) = 1.0f / tanf(fov_y * 0.5f); //Vertical FoV.
    camera_projection(1, 2) = -(2.0f * ((params.image_size.height - 1.0f * params.cy) / params.image_size.height) - 1.0f); //Vertical offset.
    camera_projection(1, 3) = 0;

    camera_projection(2, 0) = 0;
    camera_projection(2, 1) = 0;
    camera_projection(2, 2) = -(zfar + znear) / (zfar - znear); //Near and far planes.
    camera_projection(2, 3) = -(2.0f * zfar * znear) / (zfar - znear); //Near and far planes.

    camera_projection(3, 0) = 0;
    camera_projection(3, 1) = 0;
    camera_projection(3, 2) = -1;
    camera_projection(3, 3) = 0.0f;
}

void GLViewer::drawCallback() {
    currentInstance_->render();
}

void GLViewer::idle() {
    glutPostRedisplay();
}

void GLViewer::keyReleasedCallback(unsigned char c, int x, int y) {
    // space bar means change spatial mapping state
    if(c == 32)
        currentInstance_->user_action.press_space = true;

    if((c == 'p') || (c == 'P'))
        currentInstance_->user_action.hit = true;

    // Esc or Q hit means exit
    if((c == 'q') || (c == 'Q') || (c == 27))
        currentInstance_->exit();
}

void GLViewer::reshapeCallback(int width, int height) {
    glViewport(0, 0, width, height);
    currentInstance_->wnd_w = width;
    currentInstance_->wnd_h = height;
}

void GLViewer::mouseButtonCallback(int button, int state, int x, int y) {
    if(button < 3) {
        if(state == GLUT_DOWN) {
            currentInstance_->user_action.hit = true;
            currentInstance_->user_action.hit_coord.x = (x / (1.f*currentInstance_->wnd_w));
            currentInstance_->user_action.hit_coord.y = (y / (1.f*currentInstance_->wnd_h));
        }
    }
}

UserAction GLViewer::updateImageAndState(sl::Mat &im, sl::Transform &pose_, sl::POSITIONAL_TRACKING_STATE track_state) {
    if(mtx.try_lock()) {
        if(available) {
            image_handler.pushNewImage(im);
            pose = pose_;
            tracking_state = track_state;
        }
        new_data = true;
        mtx.unlock();
    }

    auto cpy = user_action;
    user_action.clear();
    return cpy;
}

void GLViewer::updateMesh(sl::Mesh &mesh, sl::PLANE_TYPE type) {
    if(mtx.try_lock()) {
        auto edges = mesh.getBoundaries();
        mesh_object.updateMesh(mesh.vertices, mesh.triangles, edges);
        mesh_object.type = type;
        mtx.unlock();
    }
}

void GLViewer::render() {
    if(available) {
        mtx.lock();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0, 0, 0, 1.f);
        update();
        draw();
        printText();
        mtx.unlock();
        glutSwapBuffers();
        glutPostRedisplay();
    }
}

void GLViewer::update() {
    // Update GPU data
    if(new_data) {
        mesh_object.pushToGPU();
        new_data = false;
    }
}

sl::float3 getPlaneColor(sl::PLANE_TYPE type) {
    sl::float3 clr;
    switch(type) {
    case sl::PLANE_TYPE::HORIZONTAL: clr = sl::float3(0.65f, 0.95f, 0.35f);
        break;
    case sl::PLANE_TYPE::VERTICAL: clr = sl::float3(0.95f, 0.35f, 0.65f);
        break;
    case sl::PLANE_TYPE::UNKNOWN: clr = sl::float3(0.35f, 0.65f, 0.95f);
        break;
    default: break;
    }
    return clr;
}

void GLViewer::draw() {
    if(available) {
        image_handler.draw();

        // If the Positional tracking is good, we can draw the mesh over the current image
        if ((tracking_state == sl::POSITIONAL_TRACKING_STATE::OK)) {
            glDisable(GL_TEXTURE_2D);
            // Send the projection and the Pose to the GLSL shader to make the projection of the 2D image.
            sl::Transform vpMatrix = camera_projection * sl::Transform::inverse(pose);
            glUseProgram(mesh_object.shader.it.getProgramId());
            glUniformMatrix4fv(mesh_object.shader.MVP_Mat, 1, GL_TRUE, vpMatrix.m);

            sl::float3 clr_plane = getPlaneColor(mesh_object.type);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glUniform3fv(mesh_object.shader.shColorLoc, 1, clr_plane.v);
            mesh_object.draw();

            glLineWidth(0.5);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glUniform3fv(mesh_object.shader.shColorLoc, 1, clr_plane.v);
            mesh_object.draw();
            glUseProgram(0);
        }

        /// Draw Hit 
        float cx = user_action.hit_coord.x * 2.f - 1.f;
        float cy = (user_action.hit_coord.y * 2.f - 1.f)*-1.f;

        float lx = 0.02f;
        float ly = lx * (wnd_w/(1.f*wnd_h));

        glLineWidth(2);
        glColor3f(0.2f, 0.45f, 0.9f);
        glBegin(GL_LINES);
        glVertex3f(cx - lx, cy, 0.0);
        glVertex3f(cx + lx, cy, 0.0);
        glVertex3f(cx, cy - ly, 0.0);
        glVertex3f(cx, cy + ly, 0.0);
        glEnd();        
    }
}

void printGL(float x, float y, const char *string) {
    glRasterPos2f(x, y);
    int len = (int) strlen(string);
    for(int i = 0; i < len; i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
    }
}

void GLViewer::printText() {
    if(available) {
        if (tracking_state != sl::POSITIONAL_TRACKING_STATE::OK) {
            glColor4f(0.85f, 0.25f, 0.15f, 1.f);
            std::string state_str;
            std::string positional_tracking_state_str("POSITIONAL TRACKING STATE : ");
            state_str = positional_tracking_state_str + toString(tracking_state).c_str();
            printGL(-0.99f, 0.9f, state_str.c_str());
        } else {
            glColor4f(0.82f, 0.82f, 0.82f, 1.f);
            printGL(-0.99f, 0.9f, "Press Space Bar to detect floor PLANE.");
            printGL(-0.99f, 0.85f, "Press 'p' key to get plane at hit.");
        }

        if(use_imu) {
            float y_start = -0.99f;
            for(int t = 0; t < static_cast<int>(sl::PLANE_TYPE::LAST); t++) {
                sl::PLANE_TYPE type = static_cast<sl::PLANE_TYPE> (t);
                sl::float3 clr = getPlaneColor(type);
                glColor4f(clr.r, clr.g, clr.b, 1.f);
                printGL(-0.99f, y_start, sl::toString(type).c_str());
                y_start += 0.05;
            }
            glColor4f(0.22, 0.22, 0.22, 1.f);
            printGL(-0.99f, y_start, "PLANES ORIENTATION :");
        }
    }
}

Shader::Shader(GLchar* vs, GLchar* fs) {
    if(!compile(verterxId_, GL_VERTEX_SHADER, vs)) {
        std::cout << "ERROR: while compiling vertex shader" << std::endl;
    }
    if(!compile(fragmentId_, GL_FRAGMENT_SHADER, fs)) {
        std::cout << "ERROR: while compiling fragment shader" << std::endl;
    }

    programId_ = glCreateProgram();

    glAttachShader(programId_, verterxId_);
    glAttachShader(programId_, fragmentId_);

    glBindAttribLocation(programId_, ATTRIB_VERTICES_POS, "in_vertex");

    glLinkProgram(programId_);

    GLint errorlk(0);
    glGetProgramiv(programId_, GL_LINK_STATUS, &errorlk);
    if(errorlk != GL_TRUE) {
        std::cout << "ERROR: while linking Shader :" << std::endl;
        GLint errorSize(0);
        glGetProgramiv(programId_, GL_INFO_LOG_LENGTH, &errorSize);

        char *error = new char[errorSize + 1];
        glGetShaderInfoLog(programId_, errorSize, &errorSize, error);
        error[errorSize] = '\0';
        std::cout << error << std::endl;

        delete[] error;
        glDeleteProgram(programId_);
    }
}

Shader::~Shader() {
    if(verterxId_ != 0)
        glDeleteShader(verterxId_);
    if(fragmentId_ != 0)
        glDeleteShader(fragmentId_);
    if(programId_ != 0)
        glDeleteShader(programId_);
}

GLuint Shader::getProgramId() {
    return programId_;
}

bool Shader::compile(GLuint &shaderId, GLenum type, GLchar* src) {
    shaderId = glCreateShader(type);
    if(shaderId == 0) {
        std::cout << "ERROR: shader type (" << type << ") does not exist" << std::endl;
        return false;
    }
    glShaderSource(shaderId, 1, (const char**) &src, 0);
    glCompileShader(shaderId);

    GLint errorCp(0);
    glGetShaderiv(shaderId, GL_COMPILE_STATUS, &errorCp);
    if(errorCp != GL_TRUE) {
        std::cout << "ERROR: while compiling Shader :" << std::endl;
        GLint errorSize(0);
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &errorSize);

        char *error = new char[errorSize + 1];
        glGetShaderInfoLog(shaderId, errorSize, &errorSize, error);
        error[errorSize] = '\0';
        std::cout << error << std::endl;

        delete[] error;
        glDeleteShader(shaderId);
        return false;
    }
    return true;
}

ImageHandler::ImageHandler() {}

ImageHandler::~ImageHandler() {
    close();
}

void ImageHandler::close() {
    glDeleteTextures(1, &imageTex);
}

bool ImageHandler::initialize(sl::Resolution res) {
    shader.it = Shader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER);
    texID = glGetUniformLocation(shader.it.getProgramId(), "texImage");
    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f, 1.0f, 0.0f };

    glGenBuffers(1, &quad_vb);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res.width, res.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_gl_ressource, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    return (err == cudaSuccess);
}

void ImageHandler::pushNewImage(sl::Mat& image) {
    cudaArray_t ArrIm;
    cudaGraphicsMapResources(1, &cuda_gl_ressource, 0);
    cudaGraphicsSubResourceGetMappedArray(&ArrIm, cuda_gl_ressource, 0, 0);
    cudaMemcpy2DToArray(ArrIm, 0, 0, image.getPtr<sl::uchar1>(sl::MEM::GPU), image.getStepBytes(sl::MEM::GPU), image.getPixelBytes() * image.getWidth(), image.getHeight(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &cuda_gl_ressource, 0);
}

void ImageHandler::draw() {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_TEXTURE_2D);
    glUseProgram(shader.it.getProgramId());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glUniform1i(texID, 0);
    //invert y axis and color for this image (since its reverted from cuda array)
    glUniform1i(glGetUniformLocation(shader.it.getProgramId(), "revert"), 1);
    glUniform1i(glGetUniformLocation(shader.it.getProgramId(), "rgbflip"), 1);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDisableVertexAttribArray(0);
    glUseProgram(0);
    glDisable(GL_TEXTURE_2D);
}
