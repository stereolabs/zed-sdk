#ifndef GENERIC_DISPLAY_H
#define GENERIC_DISPLAY_H

#include <sl/Fusion.hpp>
#include "GLViewer.hpp"

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
     * @param state current tracking state
     */
    void updatePoseData(sl::Transform zed_rt, sl::FusedPositionalTrackingStatus state);
    /**
     * @brief Display current pose on the Live Server
     * 
     * @param geo_pose geopose to display
     */
    void updateRawGeoPoseData(sl::GNSSData geo_data);
    /**
     * @brief Display current fused pose on the Live Server & in a KML file
     * 
     * @param geo_pose geopose to display
     * @param current_timestamp timestamp of the geopose to display
     */
    void updateGeoPoseData(sl::GeoPose geo_pose);

protected:
    GLViewer opengl_viewer;
};

#endif