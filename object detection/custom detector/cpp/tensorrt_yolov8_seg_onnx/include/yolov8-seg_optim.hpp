#ifndef __YOLOV8SEG_OPTIM_HPP__
#define __YOLOV8SEG_OPTIM_HPP__

#include <string>

#include <NvInfer.h>

struct OptimDim {
    nvinfer1::Dims4 size;
    std::string tensor_name;

    int setFromString(std::string const& arg);
};

int build_engine(std::string const& onnx_path, std::string const& engine_path, OptimDim const& dyn_dim_profile);

#endif // __YOLOV8SEG_OPTIM_HPP__ 