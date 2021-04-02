#include "TrackingViewer.hpp"

// -------------------------------------------------
//            2D LEFT VIEW
// -------------------------------------------------

template<typename T>
inline cv::Point2f cvt(T pt, sl::float2 scale) {
    return cv::Point2f(pt.x * scale.x, pt.y * scale.y);
}

void render_2D(cv::Mat &left_display, sl::float2 img_scale, std::vector<sl::ObjectData> &objects) {
    cv::Mat overlay = left_display.clone();
    cv::Rect roi_render(0, 0, left_display.size().width, left_display.size().height);

    // render skeleton joints and bones
    for (auto i = objects.rbegin(); i != objects.rend(); ++i) {
        sl::ObjectData& obj = (*i);
        if (renderObject(obj)) {
            if (obj.keypoint_2d.size()) {
                cv::Scalar color = generateColorID_u(obj.id);
                // skeleton bones
                for (const auto& parts : SKELETON_BONES) {
                    auto kp_a = cvt(obj.keypoint_2d[getIdx(parts.first)], img_scale);
                    auto kp_b = cvt(obj.keypoint_2d[getIdx(parts.second)], img_scale);
					if (roi_render.contains(kp_a) && roi_render.contains(kp_b))
						cv::line(left_display, kp_a, kp_b, color, 1, cv::LINE_AA);
                }
				auto spine = (obj.keypoint_2d[getIdx(sl::BODY_PARTS::LEFT_HIP)] + obj.keypoint_2d[getIdx(sl::BODY_PARTS::RIGHT_HIP)]) / 2;
				auto kp_a = cvt(spine, img_scale);
				auto kp_b = cvt(obj.keypoint_2d[getIdx(sl::BODY_PARTS::NECK)], img_scale);
				if (roi_render.contains(kp_a) && roi_render.contains(kp_b))
					cv::line(left_display, kp_a, kp_b, color, 1, cv::LINE_AA);

				// skeleton joints
				for (auto& kp : obj.keypoint_2d) {
					cv::Point2f cv_kp = cvt(kp, img_scale);
					if (roi_render.contains(cv_kp))
						cv::circle(left_display, cv_kp, 3, color, -1);
				}
				cv::Point2f cv_kp = cvt(spine, img_scale);
				if (roi_render.contains(cv_kp))
					cv::circle(left_display, cv_kp, 3, color, -1);
            }
        }
    }
    // Here, overlay is as the left image, but with opaque masks on each detected objects
    cv::addWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display);
}
