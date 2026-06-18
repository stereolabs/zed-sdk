#include "conf_from_disparity.h"

#include <math.h>

__global__ void k_confFromDisparity(const float* disp, float* conf, int width, int height, float inv_edge_sigma) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const float d = disp[y * width + x];

    // invalid disparity -> no confidence (the ingest will mark these pixels invalid anyway)
    if (isnan(d) || d <= 0.f) {
        conf[y * width + x] = 0.f;
        return;
    }

    // geometric occlusion: the matching pixel x - d falls outside the right image
    if (static_cast<float>(x) - d < 0.f) {
        conf[y * width + x] = 0.f;
        return;
    }

    // depth discontinuities: central-difference disparity gradient (clamped at borders)
    const int xm = max(x - 1, 0), xp = min(x + 1, width - 1);
    const int ym = max(y - 1, 0), yp = min(y + 1, height - 1);
    float gx = disp[y * width + xp] - disp[y * width + xm];
    float gy = disp[yp * width + x] - disp[ym * width + x];
    if (isnan(gx))
        gx = 0.f;
    if (isnan(gy))
        gy = 0.f;
    const float grad = sqrtf(gx * gx + gy * gy);

    // smooth surface -> 1, sharp depth edge -> 0
    conf[y * width + x] = expf(-grad * inv_edge_sigma);
}

void computeDisparityConfidence(const float* d_disp, float* d_conf, int width, int height, float edge_sigma, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    k_confFromDisparity<<<grid, block, 0, stream>>>(d_disp, d_conf, width, height, 1.f / fmaxf(edge_sigma, 1e-3f));
}
