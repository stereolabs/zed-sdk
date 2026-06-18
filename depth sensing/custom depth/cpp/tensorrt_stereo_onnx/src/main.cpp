///////////////////////////////////////////////////////////////////////////
//
// Custom depth ingest sample — TensorRT stereo network (ONNX)
//
// Demonstrates DEPTH_MODE::CUSTOM with a GPU source: a generic stereo
// disparity network (two image inputs, one disparity output) runs with
// TensorRT, and its raw GPU output buffer is ingested directly into the
// SDK — no device-to-host copy of the disparity.
//
// Usage: ./ZED_Custom_Depth_TensorRT <model.onnx|model.engine> [file.svo2]
//
///////////////////////////////////////////////////////////////////////////

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cctype>
#include <fstream>
#include "stereo_net.hpp"
#include "conf_from_disparity.h"
#include "GLViewer.hpp"

static bool endsWith(const std::string& str, const std::string& suffix) {
    if (str.size() < suffix.size())
        return false;
    std::string end = str.substr(str.size() - suffix.size());
    std::transform(end.begin(), end.end(), end.begin(), ::tolower);
    return end == suffix;
}

static void printUsage(const char* app) {
    std::cout << "Usage: " << app << " [model.onnx | model.engine] [file.svo | file.svo2] [--imagenet | --raw]" << std::endl;
    std::cout << "  - model: .onnx (engine built and cached on first run) or prebuilt .engine." << std::endl;
    std::cout << "    If omitted, looks for the default downloaded with: python3 get_model.py" << std::endl;
    std::cout << "  - an SVO file is optional (live camera otherwise); arguments are recognized by extension, in any order" << std::endl;
    std::cout << "  - input normalization: auto-selected from the model name (crestereo -> raw 0-255," << std::endl;
    std::cout << "    otherwise ImageNet), or forced with --raw / --imagenet" << std::endl;
    std::cout << "  - see the README for the expected ONNX layout (2 image inputs, 1 disparity output)" << std::endl;
}

int main(int argc, char** argv) {
    std::string model_path, svo_path;
    int norm_override = -1; // -1 auto, 0 raw 0-255, 1 imagenet
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (endsWith(arg, ".onnx") || endsWith(arg, ".engine"))
            model_path = arg;
        else if (endsWith(arg, ".svo") || endsWith(arg, ".svo2"))
            svo_path = arg;
        else if (arg == "--raw")
            norm_override = 0;
        else if (arg == "--imagenet")
            norm_override = 1;
        else {
            std::cout << "Unrecognized argument: " << arg << std::endl;
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (model_path.empty()) {
        // default model fetched by get_model.py
        const std::string default_model = "models/crestereo_init_iter5_480x640.onnx";
        for (const auto& candidate : {default_model, "../" + default_model}) {
            if (std::ifstream(candidate).good()) {
                model_path = candidate;
                break;
            }
        }
        if (model_path.empty()) {
            std::cout << "No stereo network model found. Download the default one with:" << std::endl;
            std::cout << "    python3 get_model.py" << std::endl << std::endl;
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
        std::cout << "Using default model: " << model_path << std::endl;
    }

    // Normalization preset: CREStereo exports take raw 0-255 RGB; most others (incl.
    // Fast-FoundationStereo) take ImageNet-normalized input.
    std::string model_lower = model_path;
    std::transform(model_lower.begin(), model_lower.end(), model_lower.begin(), ::tolower);
    const bool raw_norm = (norm_override == 0) || (norm_override == -1 && model_lower.find("crestereo") != std::string::npos);
    std::cout << "Input normalization: " << (raw_norm ? "raw 0-255" : "ImageNet (pixel/255 - mean)/std") << std::endl;

    StereoNet net;
    if (!net.init(model_path)) {
        std::cout << "Failed to load the stereo network from " << model_path << std::endl;
        return EXIT_FAILURE;
    }

    sl::Camera zed;
    sl::InitParameters init_parameters;
    init_parameters.depth_mode = sl::DEPTH_MODE::CUSTOM;                          // <- no internal depth computation
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // GLViewer convention (OpenGL), default units (mm)
    init_parameters.depth_stabilization = 0; // show the raw ingested disparity (stereo nets are often already temporally stable)
    if (!svo_path.empty())
        init_parameters.input.setFromSVOFile(svo_path.c_str());

    auto returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Camera open failed: " << returned_state << std::endl;
        return EXIT_FAILURE;
    }

    // GEN_3 positional tracking is depth-free and fully supported in CUSTOM mode;
    // when enabled, the depth stabilizer also uses its pose for motion compensation.
    sl::PositionalTrackingParameters tracking_parameters;
    tracking_parameters.mode = sl::POSITIONAL_TRACKING_MODE::GEN_3;
    zed.enablePositionalTracking(tracking_parameters);

    // 3D point cloud viewer, fed with the SDK point cloud computed from the network disparity
    const sl::Resolution pc_res = zed.getRetrieveMeasureResolution();
    auto cuda_stream = zed.getCUDAStream();
    GLViewer viewer;
    GLenum errgl
        = viewer.init(argc, argv, zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam, cuda_stream, pc_res);
    if (errgl != GLEW_OK) {
        std::cout << "Error OpenGL: " << glewGetErrorString(errgl) << std::endl;
        return EXIT_FAILURE;
    }

    // Native GPU pre-processing: both rectified views are resized, converted and normalized
    // into densely packed NCHW float tensors in a single fused SDK pass — no host round trip.
    // Defaults match Fast-FoundationStereo and most torch exports: pixel = (pixel/255 - mean) / std,
    // ImageNet mean/std. Adapt mean/std/scale here if your export embeds a different preprocessing.
    sl::TensorParameters tensor_params;
    tensor_params.target_size = sl::Resolution(net.netWidth(), net.netHeight());
    tensor_params.pixel_type = sl::TensorParameters::PIXEL_TYPE::FLOAT;
    tensor_params.color_format
        = (net.netChannels() == 1) ? sl::TensorParameters::COLOR_FORMAT::GRAY : sl::TensorParameters::COLOR_FORMAT::RGB;
    tensor_params.layout = sl::TensorParameters::LAYOUT::NCHW;
    tensor_params.stretch = true; // full stretch to the network size (no letterboxing)
    if (raw_norm) {               // raw 0-255: no scaling, no mean/std (CREStereo exports)
        tensor_params.scale = {1.f, 1.f, 1.f};
        tensor_params.mean = {0.f, 0.f, 0.f};
        tensor_params.std = {1.f, 1.f, 1.f};
    } // else: defaults = pixel/255 then ImageNet mean/std (Fast-FoundationStereo & most torch exports)

    sl::Tensor left_tensor, right_tensor;
    sl::Mat depth_view, point_cloud;
    sl::RuntimeParameters rt_parameters;
    sl::Pose pose;

    // Confidence derived from the disparity itself (edges + occlusions), computed on GPU
    float* d_confidence = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_confidence), net.netWidth() * net.netHeight() * sizeof(float));
    const float edge_sigma = 4.f; // disparity gradient (px) at which confidence drops to ~37%

    // Interactive confidence threshold: filters the measures against the ingested/derived confidence
    const std::string depth_win = "SDK depth from custom network disparity";
    int conf_slider = 95;
    cv::namedWindow(depth_win, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("confidence", depth_win, &conf_slider, 100);

    std::cout << "Press 'q' to exit" << std::endl;

    while (viewer.isAvailable()) {
        // 1. Acquisition only
        if (zed.read() != sl::ERROR_CODE::SUCCESS)
            continue;
        const auto image_ts = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);

        // 2. + 3. Native GPU pre-processing of both rectified views, then inference — the
        // images never leave the GPU: SDK tensors are bound directly as network inputs.
        // Everything runs on the SDK CUDA stream: pre-processing is event-joined onto it,
        // inference enqueues after it, and the ingest below is FIFO-ordered on the same
        // stream — no host synchronization is needed anywhere in the chain.
        if (zed.retrieveTensor(left_tensor, right_tensor, tensor_params, cuda_stream) != sl::ERROR_CODE::SUCCESS) {
            std::cout << "retrieveTensor failed" << std::endl;
            break;
        }
        if (!net.infer(left_tensor.getPtr<float>(sl::MEM::GPU), right_tensor.getPtr<float>(sl::MEM::GPU), cuda_stream)) {
            std::cout << "Inference failed" << std::endl;
            break;
        }

        // 3.5 Basic graded confidence from the disparity itself: depth edges (gradient) and
        // geometric occlusions (x - d < 0). Same stream, FIFO-ordered after the inference.
        computeDisparityConfidence(net.getOutputDevicePtr(), d_confidence, net.netWidth(), net.netHeight(), edge_sigma, cuda_stream);

        // 4. Ingest the network output directly from GPU memory (no extra copy):
        //    disparity in pixels at the network resolution, positive = closer.
        sl::CustomDepthData custom_depth;
        custom_depth.map = sl::Mat(
            net.netWidth(),
            net.netHeight(),
            sl::MAT_TYPE::F32_C1,
            reinterpret_cast<sl::uchar1*>(net.getOutputDevicePtr()),
            net.netWidth() * sizeof(float),
            sl::MEM::GPU
        );
        custom_depth.format = sl::CUSTOM_DEPTH_FORMAT::DISPARITY;
        custom_depth.scale = 1.f; // adjust if your export is normalized (= max disparity) or negative (= -1)
        custom_depth.confidence = sl::Mat(
            net.netWidth(),
            net.netHeight(),
            sl::MAT_TYPE::F32_C1,
            reinterpret_cast<sl::uchar1*>(d_confidence),
            net.netWidth() * sizeof(float),
            sl::MEM::GPU
        );
        custom_depth.confidence_convention = sl::CUSTOM_CONFIDENCE_CONVENTION::PROBABILITY;
        custom_depth.timestamp = image_ts;
        auto ingest_state = zed.ingestCustomDepth(custom_depth);
        if (ingest_state != sl::ERROR_CODE::SUCCESS)
            std::cout << "Ingest failed: " << ingest_state << std::endl;

        // 5. Full pipeline on the network disparity
        rt_parameters.confidence_threshold = conf_slider;
        returned_state = zed.grab(rt_parameters);
        if (returned_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
            break;
        if (returned_state != sl::ERROR_CODE::SUCCESS) {
            std::cout << "Grab failed: " << returned_state << std::endl;
            continue;
        }

        zed.getPosition(pose, sl::REFERENCE_FRAME::WORLD);

        // 3D point cloud computed by the SDK from the network disparity
        zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pc_res, cuda_stream);
        viewer.updatePointCloud(point_cloud);

        zed.retrieveImage(depth_view, sl::VIEW::DEPTH, sl::MEM::CPU, sl::Resolution(720, 404));
        cv::Mat depth_cv(
            depth_view.getHeight(),
            depth_view.getWidth(),
            CV_8UC4,
            depth_view.getPtr<sl::uchar1>(sl::MEM::CPU),
            depth_view.getStepBytes(sl::MEM::CPU)
        );
        cv::imshow(depth_win, depth_cv);

        // Confidence rendering: binary without an ingested confidence map, graded if the net provides one
        {
            sl::Mat conf_view;
            if (zed.retrieveImage(conf_view, sl::VIEW::CONFIDENCE, sl::MEM::CPU, sl::Resolution(720, 404)) == sl::ERROR_CODE::SUCCESS) {
                cv::Mat conf_cv(
                    conf_view.getHeight(),
                    conf_view.getWidth(),
                    CV_8UC4,
                    conf_view.getPtr<sl::uchar1>(sl::MEM::CPU),
                    conf_view.getStepBytes(sl::MEM::CPU)
                );
                cv::imshow("Confidence", conf_cv);
            }
        }

        if ((cv::waitKey(1) & 0xFF) == 'q')
            break;
    }

    point_cloud.free();
    cudaFree(d_confidence);
    zed.close();
    return EXIT_SUCCESS;
}
