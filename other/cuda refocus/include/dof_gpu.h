#ifndef DOF_GPU_H
#define DOF_GPU_H

/* dof_gpu.h.
 *
 * This file contains the interface to the CUDA functions ,
 * for rendering depth of field, based on Gaussian blurring
 * using separable convolution, with depth-dependent kernel size.
 * Separable convolution is based on convolution CUDA Sample with kernel-size adaptation
 */

#include <stdio.h> 
#include "cuda.h"
#include "cuda_runtime.h"
#include <npp.h>
#include "device_launch_parameters.h" 
#include <math.h> 
#include <algorithm>
#include "npp.h" 
#include "device_functions.h"
#include <stdint.h>



#define KERNEL_RADIUS 32 //see assert in convolution kernel to see the limitations of size
#define KERNEL_LENGTH_X(x) (2 * x + 1)
#define MAX_KERNEL_LENGTH KERNEL_LENGTH(MAX_KERNEL_RADIUS)
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)


///////////// Copy gaussien kernel into GPU memory
extern "C" void copyKernel(float *kernel_coefficients, int kernel_index);

//////////// Normalize depth between 0.f and 1.f
extern "C" void normalizeDepth(float* depth, float* depth_out,unsigned int step, float min_distance, float max_distance, unsigned int width, unsigned height);

//////////// GPU convolution /////////////////////
extern "C" void convolutionRowsGPU(
	::uchar4 *d_Dst,
	::uchar4 *d_Src,
	float* i_depth,
	int imageW,
	int imageH,
	int depth_pitch, 
	float focus_point
);

extern "C" void convolutionColumnsGPU(
	::uchar4 *d_Dst,
	::uchar4 *d_Src,
	float* i_depth,
	int imageW,
	int imageH,
	int depth_pitch,
	float focus_point
);



#endif //DOF_GPU_H