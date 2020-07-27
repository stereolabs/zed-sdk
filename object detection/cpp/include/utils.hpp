#ifndef DRAWING_UTILS_HPP
#define DRAWING_UTILS_HPP

#include <math.h>

#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

float const id_colors[5][3] = {
    { 232.0f, 176.0f ,59.0f },
    { 175.0f, 208.0f ,25.0f },
    { 102.0f, 205.0f ,105.0f},
    { 185.0f, 0.0f   ,255.0f},
    { 99.0f, 107.0f  ,252.0f}
};

inline cv::Scalar generateColorID_u(int idx) {
    if (idx < 0) return cv::Scalar(236, 184, 36, 255);
    int color_idx = idx % 5;
    return cv::Scalar(id_colors[color_idx][0], id_colors[color_idx][1], id_colors[color_idx][2], 255);
}

inline sl::float4 generateColorID_f(int idx) {
    auto clr_u = generateColorID_u(idx);
    return sl::float4(static_cast<float>(clr_u.val[0]) / 255.f, static_cast<float>(clr_u.val[1]) / 255.f, static_cast<float>(clr_u.val[2]) / 255.f, 1.f);
}

inline bool renderObject(const sl::ObjectData& i) {
    return (i.tracking_state == sl::OBJECT_TRACKING_STATE::OK || i.tracking_state == sl::OBJECT_TRACKING_STATE::OFF);
}

float const class_colors[6][3] = {
    { 44.0f, 117.0f, 255.0f}, // PEOPLE
    { 255.0f, 0.0f, 255.0f}, // VEHICLE
    { 0.0f, 0.0f, 255.0f},
    { 0.0f, 255.0f, 255.0f},
    { 0.0f, 255.0f, 0.0f},
    { 255.0f, 255.0f, 255.0f}
};

inline sl::float4 getColorClass(int idx) {
    idx = std::min(5, idx);
    sl::float4 clr(class_colors[idx][0], class_colors[idx][1], class_colors[idx][2], 1.f);
    return clr / 255.f;
}

template<typename T>
inline uchar _applyFading(T val, float current_alpha, double current_clr){
     return  static_cast<uchar>(current_alpha * current_clr + (1.0 - current_alpha) * val);
}

inline cv::Vec3b applyFading(cv::Scalar val, float current_alpha, cv::Scalar current_clr){
    cv::Vec3b out;
     out[0] = _applyFading(val.val[0], current_alpha, current_clr.val[0]);
     out[1] = _applyFading(val.val[1], current_alpha, current_clr.val[1]);
     out[2] = _applyFading(val.val[2], current_alpha, current_clr.val[2]);
     return out;
}

template<int C>
inline cv::Vec<uchar, C> applyFading(cv::Vec<uchar, C> val, float current_alpha, cv::Scalar current_clr){
    cv::Vec<uchar, C>  out;
    out[0] = _applyFading(val[0], current_alpha, current_clr.val[0]);
    out[1] = _applyFading(val[1], current_alpha, current_clr.val[1]);
    out[2] = _applyFading(val[2], current_alpha, current_clr.val[2]);
    return out;
}

inline void drawFadedLine(
        cv::Mat &left_display,
        cv::Point2i start_pt,
        cv::Point2i end_pt,
        cv::Scalar start_clr,
        cv::Scalar end_clr,
        int thickness = 3) {
    cv::LineIterator left_line_it(left_display, start_pt, end_pt, 8);
    int n_line_pts = left_line_it.count;
    int offset = thickness - 1;
    for (int i = 0; i < n_line_pts; ++i, ++left_line_it) {
        float ratio = float(i) / float(n_line_pts);
        cv::Scalar current_clr;
#if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
        for (int c = 0; c < 4; c++)
            current_clr.val[c] = ratio * end_clr.val[c] + (1.0f - ratio) * start_clr.val[c];
#else
        current_clr = ratio * end_clr + (1.f - ratio) * start_clr;
#endif
        float current_alpha = static_cast<float>(current_clr.val[3]) / 255.f;
        cv::Point2i current_position = left_line_it.pos();

        cv::Vec4b mid_px = left_display.at<cv::Vec4b>(current_position.y, current_position.x);
        left_display.at<cv::Vec4b>(current_position.y, current_position.x) = applyFading(mid_px, current_alpha,  current_clr);
        if (current_position.x - offset >= 0) {
            for (int off = 1; off <= offset; ++off) {
                cv::Vec4b px = left_display.at<cv::Vec4b>(current_position.y, current_position.x - off);               
                left_display.at<cv::Vec4b>(current_position.y, current_position.x - off) =  applyFading(px,current_alpha,  current_clr);
            }
        }
        if (current_position.x + offset < left_display.cols) {
            for (int off = 1; off <= offset; ++off) {
                cv::Vec4b px = left_display.at<cv::Vec4b>(current_position.y, current_position.x + off);
                left_display.at<cv::Vec4b>(current_position.y, current_position.x + off) = applyFading(px,current_alpha,  current_clr);
            }
        }
    }
}

inline void drawVerticalLine(
        cv::Mat &left_display,
        cv::Point2i start_pt,
        cv::Point2i end_pt,
        cv::Scalar clr,
        cv::Scalar faded_clr,
        int thickness = 3) {
    int n_steps = 24;
    cv::Point2i pt1, pt2, pt3, pt4;
#if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
    pt1.x = ((n_steps - 1) * start_pt.x + end_pt.x) / n_steps;
    pt2.x = ((n_steps - 2) * start_pt.x + 2 * end_pt.x) / n_steps;
    pt3.x = (2 * start_pt.x + (n_steps - 2) * end_pt.x) / n_steps;
    pt4.x = (start_pt.x + (n_steps - 1) * end_pt.x) / n_steps;
    pt1.y = ((n_steps - 1) * start_pt.y + end_pt.y) / n_steps;
    pt2.y = ((n_steps - 2) * start_pt.y + 2 * end_pt.y) / n_steps;
    pt3.y = (2 * start_pt.y + (n_steps - 2) * end_pt.y) / n_steps;
    pt4.y = (start_pt.y + (n_steps - 1) * end_pt.y) / n_steps;
#else
    pt1 = ((n_steps - 1) * start_pt + end_pt) / n_steps;
    pt2 = ((n_steps - 2) * start_pt + 2 * end_pt) / n_steps;
    pt3 = (2 * start_pt + (n_steps - 2) * end_pt) / n_steps;
    pt4 = (start_pt + (n_steps - 1) * end_pt) / n_steps;
#endif

    cv::line(left_display, start_pt, pt1, clr, thickness);
    drawFadedLine(
            left_display,
            pt1,
            pt2,
            clr,
            faded_clr,
            thickness
            );
    drawFadedLine(
            left_display,
            pt4,
            pt3,
            clr,
            faded_clr,
            thickness
            );
    cv::line(left_display, pt4, end_pt, clr, thickness);
}

inline cv::Mat slMat2cvMat(sl::Mat& input) {
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


#endif
