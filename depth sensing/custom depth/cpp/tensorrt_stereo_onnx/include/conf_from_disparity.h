#ifndef CONF_FROM_DISPARITY_H
#define CONF_FROM_DISPARITY_H

#include <cuda_runtime.h>

// Fast confidence derived from the disparity map itself (no extra inference):
//  - depth discontinuities (high disparity gradient) -> low confidence: conf = exp(-|grad d| / edge_sigma)
//  - geometric occlusion (x - d < 0: the match falls outside the right image) -> confidence 0
//  - invalid disparity (NaN or <= 0) -> confidence 0
// Output: [0,1] float map, 1 = confident (sl::CUSTOM_CONFIDENCE_CONVENTION::PROBABILITY).
// Asynchronous on the given stream.
void computeDisparityConfidence(const float* d_disp, float* d_conf, int width, int height, float edge_sigma, cudaStream_t stream);

#endif // CONF_FROM_DISPARITY_H
