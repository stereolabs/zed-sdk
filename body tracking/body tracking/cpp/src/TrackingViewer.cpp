#include "TrackingViewer.hpp"

// -------------------------------------------------
//            2D LEFT VIEW
// -------------------------------------------------

#define VERBOSE_DISPLAY 0

template<typename T>
inline cv::Point2f cvt(T pt, sl::float2 scale) {
    return cv::Point2f(pt.x * scale.x, pt.y * scale.y);
}

template<typename T>
void createSKPrimitive(sl::BodyData& body, const std::vector<std::pair<T, T>>&map, sl::float2 img_scale, cv::Mat& left_display, cv::Rect &roi_render, cv::Scalar color) {

    for (const auto& parts : map) {
        auto kp_a = cvt(body.keypoint_2d[getIdx(parts.first)], img_scale);
        auto kp_b = cvt(body.keypoint_2d[getIdx(parts.second)], img_scale);
        if (roi_render.contains(kp_a) && roi_render.contains(kp_b)) {

#if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
            cv::line(left_display, kp_a, kp_b, color, 1);
#else
            cv::line(left_display, kp_a, kp_b, color, 1, cv::LINE_AA);
#endif
        }
    }


    int i = 0;
    // skeleton joints
    for (auto& kp : body.keypoint_2d) {
        cv::Point2f cv_kp = cvt(kp, img_scale);
        if (roi_render.contains(cv_kp)) {
            cv::circle(left_display, cv_kp, 3, color, -1);
#if VERBOSE_DISPLAY
            //auto str = std::to_string(i++);
            auto str = std::string(sl::toString((sl::BODY_38_PARTS)i++));
            cv::putText(left_display, str, cv_kp, cv::FONT_HERSHEY_COMPLEX, 0.4/*font_size*/, cv::Scalar(255, 0, 0)/*font_Color*/, 1/*font_weight*/);
#endif
        }
    }
}


void render_2D(cv::Mat &left_display, sl::float2 img_scale, std::vector<sl::BodyData> &bodies, bool isTrackingON, bool fastRender) {
    cv::Mat overlay = left_display.clone();
    cv::Rect roi_render(0, 0, left_display.size().width, left_display.size().height);

    // render skeleton joints and bones
    for (auto &it : bodies)
        if (renderObject(it, isTrackingON)) {
            if (it.keypoint_2d.size()) {
                cv::Scalar color = generateColorID_u(it.id);

                if (fastRender) {
                    if (it.keypoint_2d.size() == 18)
                        createSKPrimitive(it, BODY_18_BONES, img_scale, left_display, roi_render, color);
                    else if (it.keypoint_2d.size() == 34)
                        createSKPrimitive(it, BODY_34_BONES, img_scale, left_display, roi_render, color);
                    else if (it.keypoint_2d.size() > 34)
                        createSKPrimitive(it, BODY_BONES_FAST_RENDER, img_scale, left_display, roi_render, color);
                } else {
                    if (it.keypoint_2d.size() == 18)
                        createSKPrimitive(it, BODY_18_BONES, img_scale, left_display, roi_render, color);
                    else if (it.keypoint_2d.size() == 34)
                        createSKPrimitive(it, BODY_34_BONES, img_scale, left_display, roi_render, color);
                    else if (it.keypoint_2d.size() == 38)
                        createSKPrimitive(it, BODY_38_BONES, img_scale, left_display, roi_render, color);
                }
            }
        }
    // Here, overlay is as the left image, but with opaque masks on each detected objects
    cv::addWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display);
}
