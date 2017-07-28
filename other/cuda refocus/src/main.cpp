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


/****************************************************************************************************
 ** This sample demonstrates how to grab and process images/depth on a CUDA kernel                 **
 ** This sample creates a simple layered depth-of-filed rendering based on CUDAconvolution sample  **
 ****************************************************************************************************/



 
// ZED SDK include
#include <sl/Camera.hpp>

// OpenGL extensions
#include "GL/glew.h"
#include "GL/glut.h"

// CUDA specific for OpenGL interoperability
#include <cuda_gl_interop.h>

// OpenCV include (to create gaussien kernel)
#include "opencv2/opencv.hpp"

// CUDA functions 
#include "dof_gpu.h"

using namespace sl;
using namespace std;

 
// Declare some resources (GL texture ID, GL shader ID...)
GLuint imageTex;
cudaGraphicsResource* pcuImageRes;

// ZED Camera object
sl::Camera* zed;

// sl::Mat ressources
sl::Mat gpuImageLeft;
sl::Mat gpuImageOutput;
sl::Mat gpuDepthMap;
sl::Mat gpuDepthMapNorm;

// tmp buffer for convolution
::uchar4* d_buffer_image;


// Focus point detected in pixels (X,Y) when mouse click event
int x_focus_point;
int y_focus_point;
float depth_focus_point = 0.f;
float norm_depth_focus_point = 0.f;


 
bool quit;
 
inline float clamp(float v, float v_min, float v_max) {
	return v < v_min ? v_min : (v > v_max ? v_max : v);
}


void mouseButtonCallback(int button, int state, int x, int y) {
 
	if (button==0) {
 		x_focus_point = x;
		y_focus_point = y;
		//--> get the depth at the mouse click point
		gpuDepthMap.getValue<sl::float1>(x_focus_point, y_focus_point, &depth_focus_point, sl::MEM_GPU);


		//--> check that the value is a number...
		if (std::isfinite(depth_focus_point))
		{
			std::cout << " Focus point set at : " << depth_focus_point << " mm at " << x_focus_point << "," << y_focus_point << std::endl;
			norm_depth_focus_point = (zed->getDepthMaxRangeValue() - depth_focus_point) / (zed->getDepthMaxRangeValue() - zed->getDepthMinRangeValue());
			clamp(norm_depth_focus_point, 0.f, 1.f);
		}
	}
}
 

void draw() {

	sl::RuntimeParameters params;
	params.sensing_mode = SENSING_MODE_FILL;

	int res = zed->grab(params);

    if (res == 0) {


		/// Retrieve Image and Depth
        zed->retrieveImage(gpuImageLeft,sl::VIEW_LEFT,sl::MEM_GPU);
	    int err = zed->retrieveMeasure(gpuDepthMap, sl::MEASURE_DEPTH, sl::MEM_GPU);

 		/// Process Image with CUDA
        ///--> normalize the depth map and make separable convolution
		normalizeDepth(gpuDepthMap.getPtr<float>(MEM_GPU), gpuDepthMapNorm.getPtr<float>(MEM_GPU), gpuDepthMap.getStep(MEM_GPU),  zed->getDepthMinRangeValue(), zed->getDepthMaxRangeValue(), gpuDepthMap.getWidth(), gpuDepthMap.getHeight());
		convolutionRowsGPU((::uchar4*)d_buffer_image,(::uchar4*)gpuImageLeft.getPtr<sl::uchar4>(MEM_GPU), gpuDepthMapNorm.getPtr<float>(MEM_GPU), gpuImageLeft.getWidth(), gpuImageLeft.getHeight(), gpuDepthMapNorm.getStep(MEM_GPU), norm_depth_focus_point);
		convolutionColumnsGPU((::uchar4*)gpuImageOutput.getPtr<sl::uchar4>(MEM_GPU), (::uchar4*)d_buffer_image, gpuDepthMapNorm.getPtr<float>(MEM_GPU), gpuImageLeft.getWidth(), gpuImageLeft.getHeight(), gpuDepthMapNorm.getStep(MEM_GPU), norm_depth_focus_point);
	
  
		/// Map to OpenGL and display
		cudaArray_t ArrIm;
        cudaGraphicsMapResources(1, &pcuImageRes, 0);
        cudaGraphicsSubResourceGetMappedArray(&ArrIm, pcuImageRes, 0, 0);
        cudaMemcpy2DToArray(ArrIm, 0, 0, gpuImageOutput.getPtr<sl::uchar4>(sl::MEM_GPU), gpuImageOutput.getStepBytes(sl::MEM_GPU), gpuImageOutput.getWidth() * sizeof(sl::uchar4), gpuImageOutput.getHeight(), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &pcuImageRes, 0);

       		

		//OpenGL Part
        glDrawBuffer(GL_BACK); //write to both BACK_LEFT & BACK_RIGHT
        glLoadIdentity();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        //Draw Image Texture in Left Part of Side by Side
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


	    //swap.
        glutSwapBuffers();

    }

    glutPostRedisplay();

    if (quit) {
        glutDestroyWindow(1);

        delete zed;
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

int main(int argc, char **argv) {

    if (argc > 2) {
        std::cout << "Only the path of a SVO can be passed in arg" << std::endl;
        return -1;
    }
    //init glut
    glutInit(&argc, argv);

    /*Setting up  The Display  */
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    //Configure Window Postion
    glutInitWindowPosition(50, 25);

    //Configure Window Size
    glutInitWindowSize(1280, 720);

    //Create Window
    glutCreateWindow("ZED DISPARITY Viewer");

    //init GLEW Library
    glewInit();


    zed = new sl::Camera();


    sl::InitParameters parameters;
	parameters.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
	parameters.camera_resolution = sl::RESOLUTION_HD720;
	parameters.coordinate_units = UNIT_MILLIMETER;
	parameters.depth_minimum_distance = 50.0;



	sl::ERROR_CODE err = zed->open(parameters);
    // ERRCODE display
    std::cout << "ZED Init Err : " << sl::errorCode2str(err) << std::endl;
    if (err != sl::SUCCESS) {
        delete zed;
        return -1;
    }

    quit = false;

    // Get Image Size
	int image_width_ = zed->getResolution().width;
	int image_height_ = zed->getResolution().height;

    cudaError_t err1;

    // Create and Register OpenGL Texture for Image (RGBA -- 4channels)
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width_, image_height_, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    err1 = cudaGraphicsGLRegisterImage(&pcuImageRes, imageTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
    if (err1 != 0 ) return -1;

	

	// Alloc sl::Mat and tmp buffer
	gpuImageLeft.alloc(zed->getResolution(), sl::MAT_TYPE_8U_C4, sl::MEM_GPU);
	gpuImageOutput.alloc(zed->getResolution(), sl::MAT_TYPE_8U_C4, sl::MEM_GPU);
	gpuDepthMap.alloc(zed->getResolution(), sl::MAT_TYPE_32F_C1, sl::MEM_GPU);
	gpuDepthMapNorm.alloc(zed->getResolution(), sl::MAT_TYPE_32F_C1, sl::MEM_GPU);
	cudaMalloc((void **)&d_buffer_image, image_width_*image_height_ * 4);


	// Create all the gaussien kernel for different radius and copy them to GPU
	vector<cv::Mat> gaussianKernel;
	vector<float*> h_kernel;
	vector<int> filter_sizes_;
	for (int radius = 1; radius <= KERNEL_RADIUS; ++radius)
		filter_sizes_.push_back(2 * radius + 1);

	for (int i = 0; i < filter_sizes_.size(); ++i) {
		gaussianKernel.push_back(cv::getGaussianKernel(filter_sizes_[i],-1, CV_32F));
		h_kernel.push_back(gaussianKernel[i].ptr<float>(0));
		copyKernel(h_kernel[i], i);

	}

	x_focus_point = image_width_ / 2;
	y_focus_point = image_height_ / 2;

	std::cout << "** Click on the image to set the focus distance **" << std::endl;

    //Set Draw Loop
    glutDisplayFunc(draw);
	glutMouseFunc(mouseButtonCallback);
    glutMainLoop();

    return 0;
}



