
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


/***********************************************************************************************
 ** This sample demonstrates how to grab images and depth map with the ZED SDK                **
 ** The GPU buffer is ingested directly into OpenGL texture to avoid GPU->CPU readback time   **
 ** For the Left image, a GLSL shader is used for RGBA-->BGRA transformation, as an example   **
 ***********************************************************************************************/

#include <stdio.h>
#include <string.h>
#include <ctime>

#include <sl/Camera.hpp>
#include <thread>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace sl;
using namespace std;

// Resource declarations (texture, GLSL fragment shader, GLSL program...)
GLuint imageTex;
GLuint depthTex;
GLuint shaderF;
GLuint program;
// Cuda resources for CUDA-OpenGL interoperability
cudaGraphicsResource* pcuImageRes;
cudaGraphicsResource* pcuDepthRes;

Camera zed;
Mat gpuLeftImage;
Mat gpuDepthImage;

// Simple fragment shader that switch red and blue channels (RGBA -->BGRA)
string strFragmentShad = ("uniform sampler2D texImage;\n"
        " void main() {\n"
        " vec4 color = texture2D(texImage, gl_TexCoord[0].st);\n"
        " gl_FragColor = vec4(color.b, color.g, color.r, color.a);\n}");

// Main loop for acquisition and rendering : 
// * grab from the ZED SDK
// * Map cuda and opengl resources and copy the GPU buffer into a CUDA array
// * Use the OpenGL texture to render on the screen

void draw() {
    if (zed.grab() == ERROR_CODE::SUCCESS) {
        // Map GPU Resource for left image
        // With OpenGL textures, we need to use the cudaGraphicsSubResourceGetMappedArray CUDA functions. It will link/sync the OpenGL texture with a CUDA cuArray
        // Then, we just have to copy our GPU Buffer to the CudaArray (DeviceToDevice copy) and the texture will contain the GPU buffer content.
        // That's the most efficient way since we don't have to go back on the CPU to render the texture. Make sure that retrieveXXX() functions of the ZED SDK
        // are used with sl::MEM::GPU parameters.
        if (zed.retrieveImage(gpuLeftImage, VIEW::LEFT, MEM::GPU) == ERROR_CODE::SUCCESS) {
            cudaArray_t ArrIm;
            cudaGraphicsMapResources(1, &pcuImageRes, 0);
            cudaGraphicsSubResourceGetMappedArray(&ArrIm, pcuImageRes, 0, 0);
            cudaMemcpy2DToArray(ArrIm, 0, 0, gpuLeftImage.getPtr<sl::uchar1>(MEM::GPU), gpuLeftImage.getStepBytes(MEM::GPU), gpuLeftImage.getWidth() * sizeof (sl::uchar4), gpuLeftImage.getHeight(), cudaMemcpyDeviceToDevice);
            cudaGraphicsUnmapResources(1, &pcuImageRes, 0);
        }

        // Map GPU Resource for depth image.
        // Note that we use the depth image here in a 8UC4 (RGBA) format.
        if (zed.retrieveImage(gpuDepthImage, VIEW::DEPTH, MEM::GPU) == ERROR_CODE::SUCCESS) {
            cudaArray_t ArrDe;
            cudaGraphicsMapResources(1, &pcuDepthRes, 0);
            cudaGraphicsSubResourceGetMappedArray(&ArrDe, pcuDepthRes, 0, 0);
            cudaMemcpy2DToArray(ArrDe, 0, 0, gpuDepthImage.getPtr<sl::uchar1>(MEM::GPU), gpuDepthImage.getStepBytes(MEM::GPU), gpuLeftImage.getWidth() * sizeof (sl::uchar4), gpuLeftImage.getHeight(), cudaMemcpyDeviceToDevice);
            cudaGraphicsUnmapResources(1, &pcuDepthRes, 0);
        }

        ////  OpenGL rendering part ////
        glDrawBuffer(GL_BACK); // Write to both BACK_LEFT & BACK_RIGHT
        glLoadIdentity();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        // Left Image on left side of the screen
        glBindTexture(GL_TEXTURE_2D, imageTex);

        // Use GLSL program to switch red and blue channels
        glUseProgram(program);

        // Render the final texture
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

        // Depth image on right side of the screen
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
}

void close() {
    gpuLeftImage.free();
    gpuDepthImage.free();
    zed.close();
    glDeleteShader(shaderF);
    glDeleteProgram(program);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void keyPressedCallback(unsigned char c, int x, int y) {
    if ((c == 27) || (c == 'q'))
        glutLeaveMainLoop();
}

int main(int argc, char **argv) {

    if (argc > 2) {
        cout << "Only the path of a SVO can be passed in arg" << endl;
        return EXIT_FAILURE;
    }
    // init glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    
    // Configure Window Size
    // define display size
    int wnd_w = 720, wnd_h = 404;
    glutInitWindowSize(wnd_w *2, wnd_h);

    // Configure Window Postion, centered on screen
    int screen_w = glutGet(GLUT_SCREEN_WIDTH);
    int screen_h = glutGet(GLUT_SCREEN_HEIGHT);
    glutInitWindowPosition(screen_w /2 - wnd_w, screen_h /2 - wnd_h/2);
    
    // Create Window
    glutCreateWindow("ZED OGL interop");

    // init GLEW Library
    glewInit();

    InitParameters init_parameters;
    // Setup our ZED Camera (construct and Init)
    if (argc == 1) { // Use in Live Mode
        init_parameters.camera_resolution = RESOLUTION::HD720;
        init_parameters.camera_fps = 30;
    } else // Use in SVO playback mode
        init_parameters.input.setFromSVOFile(String(argv[1]));

    init_parameters.depth_mode = DEPTH_MODE::PERFORMANCE;
    ERROR_CODE err = zed.open(init_parameters);

    // ERRCODE display
    if (err != ERROR_CODE::SUCCESS) {
        cout << "ZED Opening Error: " << err << endl;
        zed.close();
        return EXIT_FAILURE;
    }
    
    // Get Image Size
    auto res_ = zed.getCameraInformation().camera_configuration.resolution;

    cudaError_t err1, err2;

    // Create an OpenGL texture and register the CUDA resource on this texture for left image (8UC4 -- RGBA)
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res_.width, res_.height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    err1 = cudaGraphicsGLRegisterImage(&pcuImageRes, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // Create an OpenGL texture and register the CUDA resource on this texture for depth image (8UC4 -- RGBA)
    glGenTextures(1, &depthTex);
    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res_.width, res_.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    err2 = cudaGraphicsGLRegisterImage(&pcuDepthRes, depthTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // If any error are triggered, exit the program
    if (err1 != 0 || err2 != 0) return -1;

    // Create the GLSL program that will run the fragment shader (defined at the top)
    // * Create the fragment shader from the string source
    // * Compile the shader and check for errors
    // * Create the GLSL program and attach the shader to it
    // * Link the program and check for errors
    // * Specify the uniform variable of the shader
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

    // Link the program and check for errors
    glLinkProgram(program);
    GLint link_status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if (link_status != GL_TRUE) return -2;

    // Set the uniform variable for texImage (sampler2D) to the texture unit (GL_TEXTURE0 by default --> id = 0)
    glUniform1i(glGetUniformLocation(program, "texImage"), 0);

    // Start the draw loop and closing event function
    glutDisplayFunc(draw);
    glutCloseFunc(close);
    glutKeyboardFunc(keyPressedCallback);
    glutMainLoop();

    return EXIT_SUCCESS;
}
