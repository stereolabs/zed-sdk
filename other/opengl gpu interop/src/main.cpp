
///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2016, STEREOLABS.
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


/***********************************************************************************************
 ** This sample demonstrates how to grab images and depth map with the ZED SDK                **
 ** The GPU buffer is ingested directly into OpenGL texture to avoid GPU->CPU readback time   **
 ** For the Left image, a GLSL shader is used for RGBA-->BGRA transformation, as an example   **
 ***********************************************************************************************/

#include <stdio.h>
#include <string.h>
#include <ctime>

#include <sl_zed/Camera.hpp>
#include <thread>

#include <GL/glew.h>
#include "GL/freeglut.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace sl;
using namespace std;

//Uncomment the following line to activate frame dropped counter
//#define DROPPED_FRAME_COUNT 

// Resource declarations (GL texture ID, GL shader ID...)
GLuint imageTex;
GLuint depthTex;
GLuint shaderF;
GLuint program;
cudaGraphicsResource* pcuImageRes;
cudaGraphicsResource* pcuDepthRes;

Camera zed;
Mat gpuLeftImage;
Mat gpuDepthImage;

bool quit;

#ifdef DROPPED_FRAME_COUNT
int save_dropped_frame = 0;
int grab_counter = 0;
#endif

// Simple fragment shader to flip Red and Blue
string strFragmentShad = ("uniform sampler2D texImage;\n"
                          " void main() {\n"
                          " vec4 color = texture2D(texImage, gl_TexCoord[0].st);\n"
                          " gl_FragColor = vec4(color.b, color.g, color.r, color.a);\n}");

// GLUT main loop: grab --> extract GPU Mat --> send to OpenGL and quad
void draw() {
    // Used for jetson only, the calling thread will be executed on the 2nd core.
    Camera::sticktoCPUCore(2);

    int res = zed.grab();

    if (res == 0) {
        // Count dropped frames
#ifdef DROPPED_FRAME_COUNT
        grab_counter++;
        if (zed.getFrameDroppedCount() > save_dropped_frame) {
            save_dropped_frame = zed.getFrameDroppedCount();
            cout << save_dropped_frame << " dropped frames detected (ratio = " << 100.f * save_dropped_frame / (float) (save_dropped_frame + grab_counter) << ")" << endl;

        }
#endif

        // Map GPU Ressource for Image
        // With Gl texture, we have to use the cudaGraphicsSubResourceGetMappedArray cuda functions. It will link the gl texture with a cuArray
        // Then, we just have to copy our GPU Buffer to the CudaArray (D2D copy)
        if (zed.retrieveImage(gpuLeftImage, VIEW_LEFT, MEM_GPU) == SUCCESS) {
            cudaArray_t ArrIm;
            cudaGraphicsMapResources(1, &pcuImageRes, 0);
            cudaGraphicsSubResourceGetMappedArray(&ArrIm, pcuImageRes, 0, 0);
            cudaMemcpy2DToArray(ArrIm, 0, 0, gpuLeftImage.getPtr<sl::uchar1>(MEM_GPU), gpuLeftImage.getStepBytes(MEM_GPU), gpuLeftImage.getWidth() * sizeof(sl::uchar4), gpuLeftImage.getHeight(), cudaMemcpyDeviceToDevice);
            cudaGraphicsUnmapResources(1, &pcuImageRes, 0);
        }

        // Map GPU Ressource for Depth. Depth image == 8U 4channels
        if (zed.retrieveImage(gpuDepthImage, VIEW_DEPTH, MEM_GPU) == SUCCESS) {
            cudaArray_t ArrDe;
            cudaGraphicsMapResources(1, &pcuDepthRes, 0);
            cudaGraphicsSubResourceGetMappedArray(&ArrDe, pcuDepthRes, 0, 0);
            cudaMemcpy2DToArray(ArrDe, 0, 0, gpuDepthImage.getPtr<sl::uchar1>(MEM_GPU), gpuDepthImage.getStepBytes(MEM_GPU), gpuLeftImage.getWidth() * sizeof(sl::uchar4), gpuLeftImage.getHeight(), cudaMemcpyDeviceToDevice);
            cudaGraphicsUnmapResources(1, &pcuDepthRes, 0);
        }

        // OpenGL
        glDrawBuffer(GL_BACK); // Write to both BACK_LEFT & BACK_RIGHT
        glLoadIdentity();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        // Draw image texture in left part of side by side
        glBindTexture(GL_TEXTURE_2D, imageTex);
        // Flip R and B with GLSL Shader
        glUseProgram(program);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 1.0);
        glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0, 1.0);
        glVertex2f(0.0, -1.0);
        glTexCoord2f(1.0, 0.0);
        glVertex2f(0.0, 1.0);
        glTexCoord2f(0.0, 0.0);
        glVertex2f(-1.0, 1.0);
        glEnd();

        glUseProgram(0);

        // Draw depth texture in right part of side by side
        glBindTexture(GL_TEXTURE_2D, depthTex);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 1.0);
        glVertex2f(0.0, -1.0);
        glTexCoord2f(1.0, 1.0);
        glVertex2f(1.0, -1.0);
        glTexCoord2f(1.0, 0.0);
        glVertex2f(1.0, 1.0);
        glTexCoord2f(0.0, 0.0);
        glVertex2f(0.0, 1.0);
        glEnd();

        // Swap
        glutSwapBuffers();
    }

    glutPostRedisplay();

    if (quit) {
        gpuLeftImage.free();
        gpuDepthImage.free();
        zed.close();
        glDeleteShader(shaderF);
        glDeleteProgram(program);
        glBindTexture(GL_TEXTURE_2D, 0);
        glutDestroyWindow(1);
    }
}

void close() {
    gpuLeftImage.free();
    gpuDepthImage.free();
    zed.close();
    glDeleteShader(shaderF);
    glDeleteProgram(program);
    glBindTexture(GL_TEXTURE_2D, 0);
}

int main(int argc, char **argv) {

    if (argc > 2) {
        cout << "Only the path of a SVO can be passed in arg" << endl;
        return -1;
    }
    // init glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

    // Configure Window Postion
    glutInitWindowPosition(50, 25);

    // Configure Window Size
    glutInitWindowSize(1280, 480);

    // Create Window
    glutCreateWindow("ZED OGL interop");

    // init GLEW Library
    glewInit();

    InitParameters init_parameters;
    // Setup our ZED Camera (construct and Init)
    if (argc == 1) { // Use in Live Mode
        init_parameters.camera_resolution = RESOLUTION_HD720;
        init_parameters.camera_fps = 30.f;
    } else // Use in SVO playback mode
        init_parameters.svo_input_filename = String(argv[1]);

    init_parameters.depth_mode = DEPTH_MODE_PERFORMANCE;
    ERROR_CODE err = zed.open(init_parameters);

    // ERRCODE display
    if (err != SUCCESS) {
        cout << "ZED Opening Error: " << errorCode2str(err) << endl;
        zed.close();
        return -1;
    }

    quit = false;

    // Get Image Size
    int width = zed.getResolution().width;
    int height = zed.getResolution().height;

    cudaError_t err1, err2;

    // Create and Register OpenGL Texture for Image (RGBA -- 4channels)
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    err1 = cudaGraphicsGLRegisterImage(&pcuImageRes, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // Create and Register a OpenGL texture for Depth (RGBA- 4 Channels)
    glGenTextures(1, &depthTex);
    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    err2 = cudaGraphicsGLRegisterImage(&pcuDepthRes, depthTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    if (err1 != 0 || err2 != 0) return -1;

    // Create GLSL fragment Shader for future processing (and here flip R/B)
    GLuint shaderF = glCreateShader(GL_FRAGMENT_SHADER); //fragment shader
    const char* pszConstString = strFragmentShad.c_str();
    glShaderSource(shaderF, 1, (const char**) &pszConstString, NULL);

    // Compile the shader source code and check
    glCompileShader(shaderF);
    GLint compile_status = GL_FALSE;
    glGetShaderiv(shaderF, GL_COMPILE_STATUS, &compile_status);
    if (compile_status != GL_TRUE) return -2;

    // Create the progam for both V and F Shader
    program = glCreateProgram();
    glAttachShader(program, shaderF);

    glLinkProgram(program);
    GLint link_status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if (link_status != GL_TRUE) return -2;

    glUniform1i(glGetUniformLocation(program, "texImage"), 0);

    // Set Draw Loop
    glutDisplayFunc(draw);
    glutCloseFunc(close);
    glutMainLoop();

    return 0;
}
