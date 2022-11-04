#include "GLViewer.hpp"

#if defined(_DEBUG) && defined(_WIN32)
#error "This sample should not be built in Debug mode, use RelWithDebInfo if you want to do step by step."
#endif

#if _WIN32
#include <windows.h>
#include <iostream>
#include <shlobj.h>
#pragma comment(lib, "shell32.lib")
#endif 

#include <cuda_gl_interop.h>

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
"uniform mat4 u_mvpMatrix;\n"
"uniform vec3 u_color;\n"
"out vec3 b_color;\n"
"void main() {\n"
"   b_color = u_color;\n"
"   gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
"}";

GLchar* FPC_VERTEX_SHADER =
"#version 330 core\n"
"layout(location = 0) in vec4 in_Vertex;\n"
"uniform mat4 u_mvpMatrix;\n"
"uniform vec3 u_color;\n"
"out vec3 b_color;\n"
"void main() {\n"
"   b_color = u_color;\n"
"   gl_Position = u_mvpMatrix * vec4(in_Vertex.xyz, 1);\n"
"}";

GLchar* FRAGMENT_SHADER =
"#version 330 core\n"
"in vec3 b_color;\n"
"layout(location = 0) out vec4 color;\n"
"void main() {\n"
"   color = vec4(b_color,1);\n"
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

SubMapObj::SubMapObj() {
    current_fc = 0;
    vaoID_ = 0;
}

SubMapObj::~SubMapObj() {
    current_fc = 0;
    if(vaoID_) {
        glDeleteBuffers(2, vboID_);
        glDeleteVertexArrays(1, &vaoID_);
    }
}

template <>
void SubMapObj::update(sl::Chunk &chunk) {    
    if (vaoID_ == 0) {
        glGenVertexArrays(1, &vaoID_);
        glGenBuffers(2, vboID_);
    }

    glShadeModel(GL_SMOOTH);

    glBindVertexArray(vaoID_);

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
    glBufferData(GL_ARRAY_BUFFER, chunk.vertices.size() * sizeof(sl::float3), &chunk.vertices[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, chunk.triangles.size() * sizeof(sl::uint3), &chunk.triangles[0], GL_DYNAMIC_DRAW);
    current_fc = (int)chunk.triangles.size() * 3; 

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

template <>
void SubMapObj::update(sl::PointCloudChunk &chunk) {
    if (vaoID_ == 0) {
        glGenVertexArrays(1, &vaoID_);
        glGenBuffers(2, vboID_);
    }

    glShadeModel(GL_SMOOTH);

	const auto nb_v = chunk.vertices.size();
    index.resize(nb_v);
    for (int c = 0; c < nb_v; c++) index[c] = c;
    
    glBindVertexArray(vaoID_);

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
    glBufferData(GL_ARRAY_BUFFER, chunk.vertices.size() * sizeof(sl::float4), &chunk.vertices[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index.size() * sizeof(sl::uint1), &index[0], GL_DYNAMIC_DRAW);
    current_fc = (int)index.size();
    
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SubMapObj::draw() {
    if(current_fc && vaoID_) {
        glBindVertexArray(vaoID_);
        glDrawElements(index.size() ? GL_POINTS : GL_TRIANGLES, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}

GLViewer* currentInstance_ = nullptr;

GLViewer::GLViewer() :available(false) {
    currentInstance_ = this;
    pose.setIdentity();
    tracking_state = sl::POSITIONAL_TRACKING_STATE::OFF;
    mapping_state = sl::SPATIAL_MAPPING_STATE::NOT_ENABLED;
    change_state = false;
    new_chunks = chunks_pushed = false;
}

GLViewer::~GLViewer() {}

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


template<>
void GLViewer::initPtr(sl::Mesh* ptr) {
    p_mesh = ptr;
    draw_mesh = true;
}

template<>
void GLViewer::initPtr(sl::FusedPointCloud* ptr) {
    p_fpc = ptr;
    draw_mesh = false;
}

template<typename T>
bool GLViewer::init(int argc, char **argv, sl::CameraParameters camLeft, T *ptr) { 

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
    glutCreateWindow("ZED Spatial Mapping Viewer");
    glViewport(0, 0, width, height);

    GLenum err = glewInit();
    if (GLEW_OK != err)
        std::cout << "ERROR: glewInit failed: " << glewGetErrorString(err) << "\n";

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    bool status_ = image_handler.initialize(camLeft.image_size);
    if (!status_)
        std::cout << "ERROR: Failed to initialized Image Renderer" << std::endl;

    initPtr(ptr);

    // Create GLSL Shaders for Mesh and Image
    if(draw_mesh)
        shader_obj.it = Shader(MESH_VERTEX_SHADER, FRAGMENT_SHADER);
    else
        shader_obj.it = Shader(FPC_VERTEX_SHADER, FRAGMENT_SHADER);
    shader_obj.MVP_Mat = glGetUniformLocation(shader_obj.it.getProgramId(), "u_mvpMatrix");
    shader_obj.shColorLoc = glGetUniformLocation(shader_obj.it.getProgramId(), "u_color");

    // Create the rendering camera
    setRenderCameraProjection(camLeft, 0.5f, 20);

    glLineWidth(1.f);
    glPointSize(4.f);

    // Set glut callback before start
    glutDisplayFunc(GLViewer::drawCallback);
    glutReshapeFunc(GLViewer::reshapeCallback);
    glutKeyboardUpFunc(GLViewer::keyReleasedCallback);
    glutCloseFunc(CloseFunc);

    ask_clear = false;
    available = true;
    
    // Color of wireframe (soft blue)
    vertices_color.r = 0.35f;
    vertices_color.g = 0.65f;
    vertices_color.b = 0.95f;
    
    // ready to start
    chunks_pushed = true;

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

void printGL(float x, float y, const char *string) {
    glRasterPos2f(x, y);
    int len = (int) strlen(string);
    for(int i = 0; i < len; i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
    }
}

void GLViewer::reshapeCallback(int width, int height) {
    glViewport(0, 0, width, height);
}

void GLViewer::drawCallback() {
    currentInstance_->render();
}

void GLViewer::idle() {
    glutPostRedisplay();
}

void GLViewer::keyReleasedCallback(unsigned char c, int x, int y) {
    // space bar means change spatial mapping state
    currentInstance_->change_state = (c == 32);

    // Esc or Q hit means exit
    if((c == 'q') || (c == 'Q') || (c == 27))
        currentInstance_->exit();
}

bool GLViewer::updateImageAndState(sl::Mat &im, sl::Transform &pose_, sl::POSITIONAL_TRACKING_STATE track_state, sl::SPATIAL_MAPPING_STATE mapp_state) {
    if(mtx.try_lock()) {
        if(available) {
            image_handler.pushNewImage(im);
            pose = pose_;
            tracking_state = track_state;
            mapping_state = mapp_state;
        }
        mtx.unlock();
    }

    bool cpy_state = change_state;
    change_state = false;
    return cpy_state;
}

void GLViewer::clearCurrentMesh() {
    ask_clear = true;
    new_chunks = true;
}

void GLViewer::render() {
    if(available) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0, 0, 0, 1.f);
        mtx.lock();
        update();
        draw();
        printText();
        mtx.unlock();
        glutSwapBuffers();
        glutPostRedisplay();
    }
}

void GLViewer::update() {

    if (new_chunks) {

        if (ask_clear) {
            sub_maps.clear();
            ask_clear = false;
        }

        int nb_c = 0;
        if(draw_mesh)
            nb_c = p_mesh->chunks.size();
        else
            nb_c = p_fpc->chunks.size();

        if (nb_c > sub_maps.size()) {
            const float step = 500.f;
            size_t new_size = ((nb_c / step) + 1) * step;
            sub_maps.resize(new_size);
        }

        if (draw_mesh) {
            int c = 0;
            for (auto& it : sub_maps) {
                if ((c < nb_c) && p_mesh->chunks[c].has_been_updated)
                    it.update(p_mesh->chunks[c]);
                c++;
            }
        } else {
            int c = 0;
            for (auto& it : sub_maps) {
                if ((c < nb_c) && p_fpc->chunks[c].has_been_updated)
                    it.update(p_fpc->chunks[c]);
                c++;
            }
        }
        new_chunks = false;
        chunks_pushed = true;
    }
}

void GLViewer::printText() {
    if(available) {
        // Show actions
        if(mapping_state == sl::SPATIAL_MAPPING_STATE::NOT_ENABLED) {
            glColor3f(0.15f, 0.15f, 0.15f);
            printGL(-0.99f, 0.9f, "Hit Space Bar to activate Spatial Mapping.");
        } else {
            glColor3f(0.25f, 0.25f, 0.25f);
            printGL(-0.99f, 0.9f, "Hit Space Bar to stop spatial mapping.");
        }

        std::string positional_tracking_state_str("POSITIONAL TRACKING STATE : ");
        std::string spatial_mapping_state_str("SPATIAL MAPPING STATE : ");
        std::string state_str;
        // Show mapping state
        if ((tracking_state == sl::POSITIONAL_TRACKING_STATE::OK)) {
            if(mapping_state == sl::SPATIAL_MAPPING_STATE::OK || mapping_state == sl::SPATIAL_MAPPING_STATE::INITIALIZING)
                glColor3f(0.25f, 0.99f, 0.25f);
            else if(mapping_state == sl::SPATIAL_MAPPING_STATE::NOT_ENABLED)
                glColor3f(0.55f, 0.65f, 0.55f);
            else
                glColor3f(0.95f, 0.25f, 0.25f);
            state_str = spatial_mapping_state_str + sl::toString(mapping_state).c_str();
        } else {
            if(mapping_state != sl::SPATIAL_MAPPING_STATE::NOT_ENABLED) {
                glColor3f(0.95f, 0.25f, 0.25f);
                state_str = positional_tracking_state_str + sl::toString(tracking_state).c_str();
            } else {
                glColor3f(0.55f, 0.65f, 0.55f);
                state_str = spatial_mapping_state_str + sl::toString(sl::SPATIAL_MAPPING_STATE::NOT_ENABLED).c_str();
            }
        }
        printGL(-0.99f, 0.83f, state_str.c_str());
    }
}

void GLViewer::draw() {
    if(available) {
        image_handler.draw();

        // If the Positional tracking is good, we can draw the mesh over the current image
        if ((tracking_state == sl::POSITIONAL_TRACKING_STATE::OK) && sub_maps.size()) {
            // Draw the mesh in GL_TRIANGLES with a polygon mode in line (wire)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            // Send the projection and the Pose to the GLSL shader to make the projection of the 2D image.
            sl::Transform vpMatrix = camera_projection * sl::Transform::inverse(pose);
            glUseProgram(shader_obj.it.getProgramId());
            glUniformMatrix4fv(shader_obj.MVP_Mat, 1, GL_TRUE, vpMatrix.m);
            glUniform3fv(shader_obj.shColorLoc, 1, vertices_color.v);

            for (auto &it: sub_maps)
                it.draw();
            glUseProgram(0);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
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

std::string getDir() {
    std::string myDir;
#if _WIN32
    CHAR my_documents[MAX_PATH];
    HRESULT result = SHGetFolderPath(NULL, CSIDL_PERSONAL, NULL, SHGFP_TYPE_CURRENT, my_documents);
    if(result == S_OK)
        myDir = std::string(my_documents) + '/';
#else
    myDir = "./";
#endif
    return myDir;
}

template bool GLViewer::init<sl::FusedPointCloud>(int argc, char** argv, sl::CameraParameters camLeft, sl::FusedPointCloud* ptr);
template bool GLViewer::init<sl::Mesh>(int argc, char** argv, sl::CameraParameters camLeft, sl::Mesh* ptr);

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
