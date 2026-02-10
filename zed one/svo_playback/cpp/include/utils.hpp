#pragma once

// ZED include
#include <sl/CameraOne.hpp>

// OpenCV include (for display)
#include <opencv2/opencv.hpp>
#include <iostream>

/**
 * @brief Converts a ZED SDK Mat to an OpenCV Mat
 * @param input The ZED SDK Mat to convert
 * @return OpenCV Mat wrapping the same data (no copy)
 */
inline cv::Mat slMat2cvMat(sl::Mat& input) {
    int cv_type = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1:
            cv_type = CV_32FC1;
            break;
        case sl::MAT_TYPE::F32_C2:
            cv_type = CV_32FC2;
            break;
        case sl::MAT_TYPE::F32_C3:
            cv_type = CV_32FC3;
            break;
        case sl::MAT_TYPE::F32_C4:
            cv_type = CV_32FC4;
            break;
        case sl::MAT_TYPE::U8_C1:
            cv_type = CV_8UC1;
            break;
        case sl::MAT_TYPE::U8_C2:
            cv_type = CV_8UC2;
            break;
        case sl::MAT_TYPE::U8_C3:
            cv_type = CV_8UC3;
            break;
        case sl::MAT_TYPE::U8_C4:
            cv_type = CV_8UC4;
            break;
        default:
            break;
    }
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

/**
 * @brief Prints a formatted message with error code information
 * @param msg_prefix The message prefix
 * @param err_code The error code (default: SUCCESS)
 * @param msg_suffix Optional message suffix
 */
inline void print(std::string msg_prefix, sl::ERROR_CODE err_code = sl::ERROR_CODE::SUCCESS, std::string msg_suffix = "") {
    std::cout << "[Sample]";
    if (err_code > sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    else if (err_code < sl::ERROR_CODE::SUCCESS)
        std::cout << "[Warning] ";
    else
        std::cout << " ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << sl::toString(err_code) << " : ";
        std::cout << sl::toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}
