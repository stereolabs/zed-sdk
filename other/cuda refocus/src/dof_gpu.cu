/*
 * This file contains the definition of the CUDA functions ,
 * for rendering depth of field, based on Gaussian blurring
 * using separable convolution, with depth-dependent kernel size.
 * Separable convolution is based on convolution CUDA Sample with kernel-size adaptation
*/

#include "dof_gpu.h"

__constant__ float c_kernel[KERNEL_RADIUS * (KERNEL_RADIUS + 2)];

void copyKernel(float *kernel_coefficients, int kernel_index) {
    int kernel_radius = kernel_index + 1;
    cudaMemcpyToSymbol(c_kernel, kernel_coefficients,
                       KERNEL_LENGTH_X(kernel_radius) * sizeof(float),
                       kernel_index * (kernel_index + 2) * sizeof(float));
}

__global__ void _k_normalizeDepth(float* depth, float* depth_norm, unsigned int step, float min_distance, float max_distance, unsigned int width, unsigned height) {
    uint32_t x_local = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t y_local = blockIdx.y*blockDim.y + threadIdx.y;

    if (x_local >= width || y_local >= height) return;

    float depth_world = depth[x_local + y_local *step];
    float depth_normalized = (max_distance - depth_world) / (max_distance - min_distance);

    if (depth_normalized < 0.f) depth_normalized = 0.f;
    if (depth_normalized > 1.f) depth_normalized = 1.f;

    if (isfinite(depth_normalized))
        depth_norm[x_local + y_local *step] = depth_normalized;
}

void normalizeDepth(float* depth, float* depth_out, unsigned int step, float min_distance, float max_distance, unsigned int width, unsigned height) {
    dim3 dimGrid, dimBlock;

    dimBlock.x = 32;
    dimBlock.y = 8;

    dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

    _k_normalizeDepth << <dimGrid, dimBlock, 0 >> > (depth, depth_out, step, min_distance, max_distance, width, height);
}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 32
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

__global__ void _k_convolutionRows(sl::uchar4 *d_Dst, sl::uchar4 *d_Src, float* depth, int imageW, int imageH, int pitch, int pitch_depth, float focus_depth) {
    __shared__ sl::uchar4 s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;
    depth += baseY * pitch_depth + baseX;

    sl::uchar4 reset(0, 0, 0, 0);
    //Load main data
#pragma unroll
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    //Load left halo
#pragma unroll
    for (int i = 0; i < ROWS_HALO_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : reset;
    }

    //Load right halo
#pragma unroll
    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : reset;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        sl::float3 sum(0, 0, 0);
        int kernel_radius = floorf((KERNEL_RADIUS) * fabs(depth[i * ROWS_BLOCKDIM_X] - focus_depth));
        int kernel_mid = kernel_radius * kernel_radius - 1 + kernel_radius;
        if (kernel_radius > 0) {
            for (int j = -kernel_radius; j <= kernel_radius; ++j) {
                sum.x += c_kernel[kernel_mid + j] * (float) s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j].x;
                sum.y += c_kernel[kernel_mid + j] * (float) s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j].y;
                sum.z += c_kernel[kernel_mid + j] * (float) s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j].z;
            }
        } else {
            sum.x = (float) s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X].x;
            sum.y = (float) s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X].y;
            sum.z = (float) s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X].z;
        }

        d_Dst[i * ROWS_BLOCKDIM_X].x = sum.x;
        d_Dst[i * ROWS_BLOCKDIM_X].y = sum.y;
        d_Dst[i * ROWS_BLOCKDIM_X].z = sum.z;
        d_Dst[i * ROWS_BLOCKDIM_X].w = 255;
    }
}

void convolutionRows(sl::uchar4 *d_Dst, sl::uchar4 *d_Src, float* i_depth, int imageW, int imageH, int depth_pitch, float focus_point) {
    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
    _k_convolutionRows << <blocks, threads >> > (d_Dst, d_Src, i_depth, imageW, imageH, imageW, depth_pitch, focus_point);
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 2
#define   COLUMNS_HALO_STEPS 4

__global__ void _k_convolutionColumns(sl::uchar4 *d_Dst, sl::uchar4 *d_Src, float* depth, int imageW, int imageH, int pitch, int pitch_depth, float focus_depth) {
    __shared__ sl::uchar4 s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];
    sl::uchar4 reset(0, 0, 0, 0);
    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;
    depth += baseY * pitch_depth + baseX;

    //Main data
#pragma unroll
    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    //Upper halo
#pragma unroll
    for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : reset;
    }

    //Lower halo
#pragma unroll
    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : reset;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll
    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        sl::float3 sum(0, 0, 0);
        int kernel_radius = floorf((KERNEL_RADIUS) * fabs(depth[i * COLUMNS_BLOCKDIM_Y * pitch] - focus_depth));
        int kernel_mid = kernel_radius * kernel_radius - 1 + kernel_radius;

        if (kernel_radius > 0) {
            for (int j = -kernel_radius; j <= kernel_radius; ++j) {
                sum.x += c_kernel[kernel_mid + j] * (float) s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j].z;
                sum.y += c_kernel[kernel_mid + j] * (float) s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j].y;
                sum.z += c_kernel[kernel_mid + j] * (float) s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j].x;
            }
        } else {
            sum.x = (float) s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y].z;
            sum.y = (float) s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y].y;
            sum.z = (float) s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y].x;
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch].x = sum.x;
        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch].y = sum.y;
        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch].z = sum.z;
        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch].w = 255;
    }
}

void convolutionColumns(sl::uchar4 *d_Dst, sl::uchar4 *d_Src, float* i_depth, int imageW, int imageH, int depth_pitch, float focus_point) {
    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    _k_convolutionColumns << <blocks, threads >> > (d_Dst, d_Src, i_depth, imageW, imageH, imageW, depth_pitch, focus_point);
}
