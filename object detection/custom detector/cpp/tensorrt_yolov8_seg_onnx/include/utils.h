#ifndef TRTX_YOLOV8SEG_UTILS_H_
#define TRTX_YOLOV8SEG_UTILS_H_

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <math.h>
#include <sl/Camera.hpp>

std::vector<std::vector<int>> const CLASS_COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

inline cv::Scalar generateColorID_u(int const idx) {
    if (idx < 0) return cv::Scalar(236, 184, 36, 255);
    int color_idx = idx % CLASS_COLORS.size();
    return cv::Scalar(CLASS_COLORS[color_idx][0U], CLASS_COLORS[color_idx][1U], CLASS_COLORS[color_idx][2U], 255);
}

inline sl::float4 generateColorID_f(int const idx) {
    cv::Scalar const clr_u{generateColorID_u(idx)};
    sl::float4 clr_f{static_cast<float>(clr_u.val[0U]), static_cast<float>(clr_u.val[1U]), static_cast<float>(clr_u.val[2U]), 255.F};
    return clr_f / 255.F;
}

inline sl::float4 getColorClass(int const idx) {
    cv::Scalar const scalar{generateColorID_u(idx)};
    sl::float4 clr{static_cast<float>(scalar[0U]), static_cast<float>(scalar[1U]), static_cast<float>(scalar[2U]), static_cast<float>(scalar[3U])};
    return clr / 255.F;
}

inline bool renderObject(const sl::ObjectData& i, const bool isTrackingON) {
    if (isTrackingON)
        return (i.tracking_state == sl::OBJECT_TRACKING_STATE::OK);
    else
        return (i.tracking_state == sl::OBJECT_TRACKING_STATE::OK || i.tracking_state == sl::OBJECT_TRACKING_STATE::OFF);
}

template<typename T>
inline uchar _applyFading(T val, float current_alpha, double current_clr) {
    return static_cast<uchar> (current_alpha * current_clr + (1.0 - current_alpha) * val);
}

inline cv::Vec4b applyFading(cv::Scalar val, float current_alpha, cv::Scalar current_clr) {
    cv::Vec4b out;
    out[0] = _applyFading(val.val[0], current_alpha, current_clr.val[0]);
    out[1] = _applyFading(val.val[1], current_alpha, current_clr.val[1]);
    out[2] = _applyFading(val.val[2], current_alpha, current_clr.val[2]);
    out[3] = 255;
    return out;
}

inline void drawVerticalLine(
        cv::Mat &left_display,
        cv::Point2i start_pt,
        cv::Point2i end_pt,
        cv::Scalar clr,
        int thickness) {
    int n_steps = 7;
    cv::Point2i pt1, pt4;
    pt1.x = ((n_steps - 1) * start_pt.x + end_pt.x) / n_steps;
    pt1.y = ((n_steps - 1) * start_pt.y + end_pt.y) / n_steps;

    pt4.x = (start_pt.x + (n_steps - 1) * end_pt.x) / n_steps;
    pt4.y = (start_pt.y + (n_steps - 1) * end_pt.y) / n_steps;

    cv::line(left_display, start_pt, pt1, clr, thickness);
    cv::line(left_display, pt4, end_pt, clr, thickness);
}

inline cv::Mat slMat2cvMat(sl::Mat const& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1;
            break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2;
            break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3;
            break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4;
            break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1;
            break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2;
            break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3;
            break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4;
            break;
        default: break;
    }

    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

// Be careful about the memory owner, might want to use it like : 
//     cvMat2slMat(cv_mat).copyTo(sl_mat, sl::COPY_TYPE::CPU_CPU);
inline sl::Mat cvMat2slMat(cv::Mat const& input) {
    sl::MAT_TYPE sl_type;
    switch (input.type()) {
        case CV_32FC1: sl_type = sl::MAT_TYPE::F32_C1;
            break;
        case CV_32FC2: sl_type = sl::MAT_TYPE::F32_C2;
            break;
        case CV_32FC3: sl_type = sl::MAT_TYPE::F32_C3;
            break;
        case CV_32FC4: sl_type = sl::MAT_TYPE::F32_C4;
            break;
        case CV_8UC1: sl_type = sl::MAT_TYPE::U8_C1;
            break;
        case CV_8UC2: sl_type = sl::MAT_TYPE::U8_C2;
            break;
        case CV_8UC3: sl_type = sl::MAT_TYPE::U8_C3;
            break;
        case CV_8UC4: sl_type = sl::MAT_TYPE::U8_C4;
            break;
        default: break;
    }
    return sl::Mat(input.cols, input.rows, sl_type, input.data, input.step, sl::MEM::CPU);
}

inline bool readFile(std::string filename, std::vector<uint8_t> &file_content) {
    // open the file:
    std::ifstream instream(filename, std::ios::in | std::ios::binary);
    if (!instream.is_open()) return true;
    file_content = std::vector<uint8_t>((std::istreambuf_iterator<char>(instream)), std::istreambuf_iterator<char>());
    return false;
}

inline std::vector<std::string> split_str(std::string const& s, std::string const& delimiter) {
    size_t pos_start{0U}, pos_end, delim_len{delimiter.length()};
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

#endif  // TRTX_YOLOV8SEG_UTILS_H_

