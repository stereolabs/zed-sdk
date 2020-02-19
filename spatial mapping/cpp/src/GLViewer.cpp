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
    new_images = new_chunks = chunks_pushed = false;
}

GLViewer::~GLViewer() {
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &imageTex);
    glDeleteTextures(1, &renderedTexture);
    glDeleteBuffers(1, &quad_vb);
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
        image.free();
        delete shader_mesh;
        delete shader_image;
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
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

    // Create GLUT window
    glutInitWindowSize(camLeft.image_size.width, camLeft.image_size.height);
    glutCreateWindow("ZED Spatial Mapping");

    // Init glew after window has been created
    glewInit();
    
    // Create and Register OpenGL Texture for Image (RGBA -- 4channels)
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, camLeft.image_size.width, camLeft.image_size.height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaSafeCall(cudaGraphicsGLRegisterImage(&cuda_gl_ressource, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

    initPtr(ptr);

    // Create GLSL Shaders for Mesh and Image
    if(draw_mesh)
        shader_mesh = new Shader(MESH_VERTEX_SHADER, FRAGMENT_SHADER);
    else
        shader_mesh = new Shader(FPC_VERTEX_SHADER, FRAGMENT_SHADER);

    shMVPMatrixLoc = glGetUniformLocation(shader_mesh->getProgramId(), "u_mvpMatrix");
    shColorLoc = glGetUniformLocation(shader_mesh->getProgramId(), "u_color");
    shader_image = new Shader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER);
    texID = glGetUniformLocation(shader_image->getProgramId(), "texImage");
    
    // Create Frame Buffer for offline rendering
    // Here we render the composition of the image and the projection of the mesh on top of it in a texture (using FBO - Frame Buffer Object)
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Generate a render texture (which will contain the image and mesh in wireframe overlay)
    glGenTextures(1, &renderedTexture);
    glBindTexture(GL_TEXTURE_2D, renderedTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, camLeft.image_size.width, camLeft.image_size.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    // Set "renderedTexture" as our color attachment #0
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Set the list of draw buffers.
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers);

    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        print("Error : invalid FrameBuffer");
        return true;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Generate a buffer to handle vertices for the GLSL shader
    // Generate the Quad for showing the image in a full viewport
    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f, 1.0f, 0.0f};

    glGenBuffers(1, &quad_vb);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glDisable(GL_TEXTURE_2D);

    // Set glut callback before start
    glutDisplayFunc(GLViewer::drawCallback);
    glutKeyboardUpFunc(GLViewer::keyReleasedCallback);
    glutCloseFunc(CloseFunc);

    ask_clear = false;
    available = true;
    
    image.alloc(camLeft.image_size, sl::MAT_TYPE::U8_C4, sl::MEM::GPU);
    cudaSafeCall(cudaGetLastError());

    // Create Projection Matrix for OpenGL. We will use this matrix in combination with the Pose (on REFERENCE_FRAME::WORLD) to project the mesh on the 2D Image.
    camera_projection(0, 0) = 1.0f / tanf(camLeft.h_fov * 3.1416f / 180.f * 0.5f);
    camera_projection(1, 1) = 1.0f / tanf(camLeft.v_fov * 3.1416f / 180.f * 0.5f);
    float znear = 0.001f;
    float zfar = 100.f;
    camera_projection(2, 2) = -(zfar + znear) / (zfar - znear);
    camera_projection(2, 3) = -(2.f * zfar * znear) / (zfar - znear);
    camera_projection(3, 2) = -1.f;
    camera_projection(0, 2) = (camLeft.image_size.width - 2.f * camLeft.cx) / camLeft.image_size.width;
    camera_projection(1, 2) = (-1.f * camLeft.image_size.height + 2.f * camLeft.cy) / camLeft.image_size.height;
    camera_projection(3, 3) = 0.f;
    
    // Color of wireframe (soft blue)
    vertices_color.r = 0.35f;
    vertices_color.g = 0.65f;
    vertices_color.b = 0.95f;
    
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    // ready to start
    chunks_pushed = true;

    return false;
}

void printGL(float x, float y, const char *string) {
    glRasterPos2f(x, y);
    int len = (int) strlen(string);
    for(int i = 0; i < len; i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
    }
}

void GLViewer::drawCallback() {
    currentInstance_->render();
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
            image.setFrom(im, sl::COPY_TYPE::GPU_GPU);
            cudaSafeCall(cudaGetLastError());
            pose = pose_;
            tracking_state = track_state;
            mapping_state = mapp_state;
        }
        new_images = true;
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

    // Update GPU data
    if(new_images) {
        cudaArray_t ArrIm;
        cudaSafeCall(cudaGraphicsMapResources(1, &cuda_gl_ressource, 0));
        cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&ArrIm, cuda_gl_ressource, 0, 0));
        cudaSafeCall(cudaMemcpy2DToArray(ArrIm, 0, 0, image.getPtr<sl::uchar1>(sl::MEM::GPU), image.getStepBytes(sl::MEM::GPU), image.getPixelBytes()*image.getWidth(), image.getHeight(), cudaMemcpyDeviceToDevice));
        cudaSafeCall(cudaGraphicsUnmapResources(1, &cuda_gl_ressource, 0));

        new_images = false;
    }

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
            glColor4f(0.15f, 0.15f, 0.15f, 1.f);
            printGL(-0.99f, 0.9f, "Hit Space Bar to activate Spatial Mapping.");
        } else {
            glColor4f(0.25f, 0.25f, 0.25f, 1.f);
            printGL(-0.99f, 0.9f, "Hit Space Bar to stop spatial mapping.");
        }

        std::string positional_tracking_state_str("POSITIONAL TRACKING STATE : ");
        std::string spatial_mapping_state_str("SPATIAL MAPPING STATE : ");
        std::string state_str;
        // Show mapping state
        if ((tracking_state == sl::POSITIONAL_TRACKING_STATE::OK)) {
            if(mapping_state == sl::SPATIAL_MAPPING_STATE::OK || mapping_state == sl::SPATIAL_MAPPING_STATE::INITIALIZING)
                glColor4f(0.25f, 0.99f, 0.25f, 1.f);
            else if(mapping_state == sl::SPATIAL_MAPPING_STATE::NOT_ENABLED)
                glColor4f(0.55f, 0.65f, 0.55f, 1.f);
            else
                glColor4f(0.95f, 0.25f, 0.25f, 1.f);
            state_str = spatial_mapping_state_str + sl::toString(mapping_state).c_str();
        } else {
            if(mapping_state != sl::SPATIAL_MAPPING_STATE::NOT_ENABLED) {
                glColor4f(0.95f, 0.25f, 0.25f, 1.f);
                state_str = positional_tracking_state_str + sl::toString(tracking_state).c_str();
            } else {
                glColor4f(0.55f, 0.65f, 0.55f, 1.f);
                state_str = spatial_mapping_state_str + sl::toString(sl::SPATIAL_MAPPING_STATE::NOT_ENABLED).c_str();
            }
        }
        printGL(-0.99f, 0.83f, state_str.c_str());
    }
}

void GLViewer::draw() {
    if(available) {        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE0);

        // Render image and wireframe mesh into a texture using frame buffer
        // Bind the frame buffer and specify the viewport (full screen)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        // Render the ZED view (Left) in the framebuffer
        glUseProgram(shader_image->getProgramId());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, imageTex);
        glUniform1i(texID, 0);
        //invert y axis and color for this image (since its reverted from cuda array)
        glUniform1i(glGetUniformLocation(shader_image->getProgramId(), "revert"), 1);
        glUniform1i(glGetUniformLocation(shader_image->getProgramId(), "rgbflip"), 1);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisableVertexAttribArray(0);
        glUseProgram(0);

        // If the Positional tracking is good, we can draw the mesh over the current image
        if ((tracking_state == sl::POSITIONAL_TRACKING_STATE::OK) && sub_maps.size()) {
            glLineWidth(1.f);
            glPointSize(4.f);
            glDisable(GL_TEXTURE_2D);
            // Send the projection and the Pose to the GLSL shader to make the projection of the 2D image.
            sl::Transform vpMatrix = camera_projection * sl::Transform::inverse(pose);
            glUseProgram(shader_mesh->getProgramId());
            glUniformMatrix4fv(shMVPMatrixLoc, 1, GL_TRUE, vpMatrix.m);

            glUniform3fv(shColorLoc, 1, vertices_color.v);
            // Draw the mesh in GL_TRIANGLES with a polygon mode in line (wire)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

            for (auto &it: sub_maps)
                it.draw();

            glUseProgram(0);
        }
        // Unbind the framebuffer since the texture is now updated
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Render the texture to the screen
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glUseProgram(shader_image->getProgramId());
        glBindTexture(GL_TEXTURE_2D, renderedTexture);
        glUniform1i(texID, 0);
        glUniform1i(glGetUniformLocation(shader_image->getProgramId(), "revert"), 0);
        glUniform1i(glGetUniformLocation(shader_image->getProgramId(), "rgbflip"), 0);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*) 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisableVertexAttribArray(0);
        glUseProgram(0);
        glDisable(GL_TEXTURE_2D);
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
