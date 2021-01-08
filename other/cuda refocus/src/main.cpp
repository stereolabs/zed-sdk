///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, STEREOLABS.
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


/****************************************************************************************************
 ** This sample demonstrates how to grab and process images/depth on a CUDA kernel                 **
 ** This sample creates a simple layered depth-of-filed rendering based on CUDAconvolution sample  **
 ****************************************************************************************************/

 // ZED SDK include
#include <sl/Camera.hpp>

// OpenGL extensions
#include "GL/glew.h"
#include "GL/freeglut.h"

// CUDA specific for OpenGL interoperability
#include <cuda_gl_interop.h>

// CUDA functions 
#include "dof_gpu.h"

using namespace sl;
using namespace std;

// Declare some resources (GL texture ID, GL shader ID...)
GLuint imageTex;
cudaGraphicsResource* pcuImageRes;

// ZED Camera object
Camera zed;

// Mat ressources
Mat gpu_image_left;
Mat gpu_Image_render;
Mat gpu_depth;
Mat gpu_depth_normalized;
Mat gpu_image_convol;

// Focus point detected in pixels (X,Y) when mouse click event
float norm_depth_focus_point = 0.f;

void mouseButtonCallback(int button, int state, int x, int y) {
    if (button == 0 && state) {
        // Get the depth at the mouse click point
        float depth_focus_point = 0.f;
        float max_range = zed.getInitParameters().depth_maximum_distance;
        float min_range = zed.getInitParameters().depth_minimum_distance;
        gpu_depth.getValue<sl::float1>(x, y, &depth_focus_point, MEM::GPU);
        // Check that the value is valid
        if (isValidMeasure(depth_focus_point)) {
            cout << " Focus point set at : " << depth_focus_point << "mm {" << x << "," << y << "}" << endl;
            norm_depth_focus_point = (max_range - depth_focus_point) / (max_range - min_range);
            norm_depth_focus_point = norm_depth_focus_point > 1.f ? 1.f : (norm_depth_focus_point < 0.f ? 0.f : norm_depth_focus_point);
        }
    }
}

void draw() {
    RuntimeParameters params;
    params.sensing_mode = SENSING_MODE::FILL;

    if (zed.grab(params) == ERROR_CODE::SUCCESS) {
        // Retrieve Image and Depth
        zed.retrieveImage(gpu_image_left, VIEW::LEFT, MEM::GPU);
        zed.retrieveMeasure(gpu_depth, MEASURE::DEPTH, MEM::GPU);

        // Process Image with CUDA
        // Normalize the depth map and make separable convolution
        float max_range = zed.getInitParameters().depth_maximum_distance;
        float min_range = zed.getInitParameters().depth_minimum_distance;


        normalizeDepth(gpu_depth.getPtr<float>(MEM::GPU), gpu_depth_normalized.getPtr<float>(MEM::GPU), gpu_depth.getStep(MEM::GPU), min_range, max_range, gpu_depth.getWidth(), gpu_depth.getHeight());
        convolutionRows(gpu_image_convol.getPtr<sl::uchar4>(MEM::GPU), gpu_image_left.getPtr<sl::uchar4>(MEM::GPU), gpu_depth_normalized.getPtr<float>(MEM::GPU), gpu_image_left.getWidth(), gpu_image_left.getHeight(), gpu_depth_normalized.getStep(MEM::GPU), norm_depth_focus_point);
        convolutionColumns(gpu_Image_render.getPtr<sl::uchar4>(MEM::GPU), gpu_image_convol.getPtr<sl::uchar4>(MEM::GPU), gpu_depth_normalized.getPtr<float>(MEM::GPU), gpu_image_left.getWidth(), gpu_image_left.getHeight(), gpu_depth_normalized.getStep(MEM::GPU), norm_depth_focus_point);

        // Map to OpenGL and display
        cudaArray_t ArrIm;
        cudaGraphicsMapResources(1, &pcuImageRes, 0);
        cudaGraphicsSubResourceGetMappedArray(&ArrIm, pcuImageRes, 0, 0);
        cudaMemcpy2DToArray(ArrIm, 0, 0, gpu_Image_render.getPtr<sl::uchar4>(MEM::GPU), gpu_Image_render.getStepBytes(MEM::GPU), gpu_Image_render.getWidth() * sizeof(sl::uchar4), gpu_Image_render.getHeight(), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &pcuImageRes, 0);

        //OpenGL Part
        glDrawBuffer(GL_BACK);
        glLoadIdentity();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glBindTexture(GL_TEXTURE_2D, imageTex);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 1.0);
        glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0, 1.0);
        glVertex2f(1.0, -1.0);
        glTexCoord2f(1.0, 0.0);
        glVertex2f(1.0, 1.0);
        glTexCoord2f(0.0, 0.0);
        glVertex2f(-1.0, 1.0);
        glEnd();

        glutSwapBuffers();
    }

    glutPostRedisplay();
}

int main(int argc, char **argv) {

    if (argc > 2) {
        cout << "Only the path of a SVO can be passed in arg" << endl;
        return EXIT_FAILURE;
    }

    // Init glut
    glutInit(&argc, argv);

    //Create Window
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowPosition(50, 25);
    glutInitWindowSize(1280, 720);
    glutCreateWindow("ZED CUDA Refocus");

    //init GLEW
    glewInit();
    
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::ULTRA;
    init_parameters.camera_resolution = RESOLUTION::HD720;
    init_parameters.coordinate_units = UNIT::MILLIMETER;
    init_parameters.depth_minimum_distance = 400.0f;

    // Open the camera
    ERROR_CODE zed_open_state = zed.open(init_parameters);
    if (zed_open_state != ERROR_CODE::SUCCESS) {
        std::cout << "Error " << zed_open_state << ", exit program.\n";
        return EXIT_FAILURE;
    }

    // Get Image Size
    sl::Resolution camera_resolution_ = zed.getCameraInformation().camera_configuration.resolution;

    // Create and Register OpenGL Texture for Image (RGBA -- 4channels)
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, camera_resolution_.width, camera_resolution_.height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    cudaError_t state = cudaGraphicsGLRegisterImage(&pcuImageRes, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (state != cudaSuccess)
        return EXIT_FAILURE;

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    // Alloc Mat and tmp buffer
    gpu_image_left.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_Image_render.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_depth.alloc(camera_resolution_, MAT_TYPE::F32_C1, MEM::GPU);
    gpu_depth_normalized.alloc(camera_resolution_, MAT_TYPE::F32_C1, MEM::GPU);
    gpu_image_convol.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);

    vector<float> gauss_vec;
    // Create all the gaussien kernel for different radius and copy them to GPU
    for (int i = 0; i < KERNEL_RADIUS; ++i) {
        gauss_vec.resize((i + 1) * 2 + 1, 0);

        // Compute Gaussian coeff
        int rad = (gauss_vec.size() - 1) / 2;
        float sigma = 0.3f * ((gauss_vec.size() - 1.f)*0.5f - 1.f) + 0.8f;
        float sum = 0;
        for (int u = -rad; u <= rad; u++) {
            float gauss_value = expf(-1.f * (powf(u, 2.f) / (2.f * powf(sigma, 2.f))));
            gauss_vec[u + rad] = gauss_value;
            sum += gauss_value;
        }
        sum = 1.f / sum;
        for (int u = 0; u < gauss_vec.size(); u++)
            gauss_vec[u] *= sum;

        // Copy coeff to GPU
        copyKernel(gauss_vec.data(), i);
    }

    cout << "** Click on the image to set the focus distance **" << endl;

    glutDisplayFunc(draw);
    glutMouseFunc(mouseButtonCallback);
    glutMainLoop(); // Start main loop 

    //On close
    gpu_image_left.free();
    gpu_Image_render.free();
    gpu_depth.free();
    gpu_depth_normalized.free();
    gpu_image_convol.free();
    zed.close();
    return EXIT_SUCCESS;
}
