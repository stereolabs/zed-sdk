///////////////////////////////////////////////////////////////////////////
//
// Custom depth ingest sample — OpenCV StereoSGBM
//
// Demonstrates DEPTH_MODE::CUSTOM: the SDK acquisition/rectification is used,
// the disparity is computed externally (here cv::StereoSGBM on CPU, at half
// resolution), then ingested back. The full SDK pipeline (measures, point
// cloud, modules) runs on the ingested disparity.
//
///////////////////////////////////////////////////////////////////////////

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include "GLViewer.hpp"

int main(int argc, char** argv) {
    sl::Camera zed;

    sl::InitParameters init_parameters;
    init_parameters.depth_mode = sl::DEPTH_MODE::CUSTOM;                          // <- no internal depth computation
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // GLViewer convention (OpenGL), default units (mm)
    init_parameters.camera_resolution = sl::RESOLUTION::AUTO;
    init_parameters.depth_stabilization
        = 0; // show the raw ingested disparity (no temporal fusion; without tracking it would assume a static camera)
    if (argc > 1 && std::string(argv[1]).find(".svo") != std::string::npos)
        init_parameters.input.setFromSVOFile(argv[1]);

    auto returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Camera open failed: " << returned_state << std::endl;
        return EXIT_FAILURE;
    }

    // SGBM runs at half resolution for speed: the SDK accepts the map at any resolution,
    // disparity values are expressed in pixels AT THE MAP RESOLUTION (rescaled internally).
    const auto cam_res = zed.getCameraInformation().camera_configuration.resolution;
    const cv::Size sgbm_size(cam_res.width / 2, cam_res.height / 2);

    const int num_disparities = 16 * 8; // multiple of 16
    const int block_size = 5;
    auto sgbm = cv::StereoSGBM::create(0 /*minDisparity*/, num_disparities, block_size);
    sgbm->setP1(8 * block_size * block_size);
    sgbm->setP2(32 * block_size * block_size);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(2);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

    // 3D point cloud viewer, fed with the SDK point cloud computed from the ingested disparity
    const sl::Resolution pc_res = zed.getRetrieveMeasureResolution();
    auto cuda_stream = zed.getCUDAStream();
    GLViewer viewer;
    GLenum errgl
        = viewer.init(argc, argv, zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam, cuda_stream, pc_res);
    if (errgl != GLEW_OK) {
        std::cout << "Error OpenGL: " << glewGetErrorString(errgl) << std::endl;
        return EXIT_FAILURE;
    }

    sl::Mat left_gray, right_gray, depth_view, point_cloud;
    cv::Mat left_half, right_half, disp_16s, disp_32f;
    sl::RuntimeParameters rt_parameters;

    // Interactive confidence threshold: filters the measures against the ingested/derived confidence
    const std::string depth_win = "SDK depth from custom SGBM disparity";
    int conf_slider = 95;
    cv::namedWindow(depth_win, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("confidence", depth_win, &conf_slider, 100);

    std::cout << "Press 'q' to exit" << std::endl;

    while (viewer.isAvailable()) {
        // 1. Acquisition only: image + IMU (+ recording/streaming if enabled), no depth
        if (zed.read() != sl::ERROR_CODE::SUCCESS)
            continue;

        // 2. Rectified images (rectification runs on demand)
        zed.retrieveImage(left_gray, sl::VIEW::LEFT_GRAY, sl::MEM::CPU);
        zed.retrieveImage(right_gray, sl::VIEW::RIGHT_GRAY, sl::MEM::CPU);
        const auto image_ts = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);

        cv::Mat left_cv(
            left_gray.getHeight(),
            left_gray.getWidth(),
            CV_8UC1,
            left_gray.getPtr<sl::uchar1>(sl::MEM::CPU),
            left_gray.getStepBytes(sl::MEM::CPU)
        );
        cv::Mat right_cv(
            right_gray.getHeight(),
            right_gray.getWidth(),
            CV_8UC1,
            right_gray.getPtr<sl::uchar1>(sl::MEM::CPU),
            right_gray.getStepBytes(sl::MEM::CPU)
        );

        // 3. External disparity computation
        cv::resize(left_cv, left_half, sgbm_size, 0, 0, cv::INTER_AREA);
        cv::resize(right_cv, right_half, sgbm_size, 0, 0, cv::INTER_AREA);
        sgbm->compute(left_half, right_half, disp_16s);
        // SGBM outputs 16-bit fixed point (disparity * 16); invalid pixels are negative.
        // No cleanup needed: scale=1/16 restores pixels, and values <= 0 are classified
        // as invalid by the SDK at ingest.
        disp_16s.convertTo(disp_32f, CV_32F);

        // Graded pseudo-confidence from the local image gradient (texture): SGBM is unreliable on
        // textureless areas. Demo of the PROBABILITY confidence path — with it, the confidence
        // threshold slider sweeps the density instead of acting as a binary valid/invalid switch.
        cv::Mat grad_x, grad_y, conf_32f;
        cv::Sobel(left_half, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(left_half, grad_y, CV_32F, 0, 1, 3);
        cv::magnitude(grad_x, grad_y, conf_32f);
        cv::boxFilter(conf_32f, conf_32f, CV_32F, cv::Size(2 * block_size + 1, 2 * block_size + 1));
        conf_32f = cv::min(conf_32f / 64.f, 1.f); // ~64 gray-levels of local gradient -> fully confident

        // 4. Ingest
        sl::CustomDepthData custom_depth;
        custom_depth.map = sl::Mat(
            sgbm_size.width,
            sgbm_size.height,
            sl::MAT_TYPE::F32_C1,
            reinterpret_cast<sl::uchar1*>(disp_32f.data),
            disp_32f.step,
            sl::MEM::CPU
        );
        custom_depth.format = sl::CUSTOM_DEPTH_FORMAT::DISPARITY;
        custom_depth.scale = 1.f / 16.f; // SGBM fixed-point
        custom_depth.confidence = sl::Mat(
            sgbm_size.width,
            sgbm_size.height,
            sl::MAT_TYPE::F32_C1,
            reinterpret_cast<sl::uchar1*>(conf_32f.data),
            conf_32f.step,
            sl::MEM::CPU
        );
        custom_depth.confidence_convention = sl::CUSTOM_CONFIDENCE_CONVENTION::PROBABILITY; // [0,1], 1 = confident
        custom_depth.timestamp = image_ts;
        auto ingest_state = zed.ingestCustomDepth(custom_depth);
        if (ingest_state != sl::ERROR_CODE::SUCCESS)
            std::cout << "Ingest failed: " << ingest_state << std::endl;

        // 5. Full pipeline on the ingested disparity
        rt_parameters.confidence_threshold = conf_slider;
        returned_state = zed.grab(rt_parameters);
        if (returned_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
            break;
        if (returned_state != sl::ERROR_CODE::SUCCESS) {
            std::cout << "Grab failed: " << returned_state << std::endl;
            continue;
        }

        // 3D point cloud computed by the SDK from the ingested disparity
        zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pc_res, cuda_stream);
        viewer.updatePointCloud(point_cloud);

        // Metric sanity check: depth at the image center, straight from the ingested disparity
        static int frame_count = 0;
        if ((frame_count++ % 30) == 0) {
            sl::Mat depth_measure;
            zed.retrieveMeasure(depth_measure, sl::MEASURE::DEPTH, sl::MEM::CPU);
            float center_depth = 0.f;
            depth_measure.getValue<sl::float1>(depth_measure.getWidth() / 2, depth_measure.getHeight() / 2, &center_depth);
            std::cout << "Center depth: " << center_depth << " " << sl::toString(init_parameters.coordinate_units) << "      \r"
                      << std::flush;
        }

        // Display the SDK depth view, generated from the ingested disparity
        zed.retrieveImage(depth_view, sl::VIEW::DEPTH, sl::MEM::CPU, sl::Resolution(720, 404));
        cv::Mat depth_cv(
            depth_view.getHeight(),
            depth_view.getWidth(),
            CV_8UC4,
            depth_view.getPtr<sl::uchar1>(sl::MEM::CPU),
            depth_view.getStepBytes(sl::MEM::CPU)
        );
        cv::imshow(depth_win, depth_cv);

        // Confidence rendering: graded with the ingested texture-based confidence (binary if confidence is not provided)
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
    zed.close();
    return EXIT_SUCCESS;
}
