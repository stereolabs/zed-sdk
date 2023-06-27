#ifndef GENERIC_DISPLAY_H
#define GENERIC_DISPLAY_H

#include <sl/Fusion.hpp>
#include "GLViewer.hpp"
#include <opencv2/opencv.hpp>

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

class GenericDisplay
{
public:
/**
 * @brief Construct a new Generic Display object
 * 
 */
    GenericDisplay();
    /**
     * @brief Destroy the Generic Display object
     * 
     */
    ~GenericDisplay();
    /**
     * @brief Init OpenGL display with the requested camera_model (used as moving element in OpenGL view)
     * 
     * @param argc default main argc
     * @param argv default main argv
     * @param camera_model zed camera model to use
     */
    void init(int argc, char **argv);
    /**
     * @brief Return if the OpenGL viewer is still open
     * 
     * @return true the OpenGL viewer is still open
     * @return false the OpenGL viewer was closed
     */
    bool isAvailable();
    /**
     * @brief Update the OpenGL view with last pose data
     * 
     * @param zed_rt last pose data
     * @param str_t std::string that represents current translations
     * @param str_r std::string that represents current rotations
     * @param state current tracking state
     */
    void updatePoseData(sl::Transform zed_rt, std::string str_t, std::string str_r, sl::POSITIONAL_TRACKING_STATE state);
    /**
     * @brief Display current fused pose either in KML file or in ZEDHub depending compilation options
     * 
     * @param geo_pose geopose to display
     * @param current_timestamp timestamp of the geopose to display
     */
    void updateGeoPoseData(sl::GeoPose geo_pose, sl::Timestamp current_timestamp);

protected:
    GLViewer opengl_viewer;
};

#endif