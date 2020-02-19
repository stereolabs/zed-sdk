#ifndef TRACKING_VIEWER_HPP
#define TRACKING_VIEWER_HPP

#include <iostream>
#include <deque>
#include <math.h>

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

// -------------------------------------------------
//            2D LEFT VIEW
// -------------------------------------------------

inline sl::float2 getImagePosition(std::vector<sl::uint2> &bounding_box_image, sl::float2 img_scale) {
    sl::float2 position;
    position.x = (bounding_box_image[0].x + (bounding_box_image[2].x - bounding_box_image[0].x)*0.5f) * img_scale.x;
    position.y = (bounding_box_image[0].y + (bounding_box_image[2].y - bounding_box_image[0].y)*0.5f) * img_scale.y;
    return position;
}

void render_2D(sl::Mat &left, sl::float2 img_scale, std::vector<sl::ObjectData> &objects, bool render_mask = false);


// -------------------------------------------------
//            2D TRACKING VIEW
// -------------------------------------------------

/*
    WARNING: Timestamp values are in nanoseconds
 */
enum class TrackPointState {
    OK = 0,
    PREDICTED,
    OFF
};

struct TrackPoint {

    TrackPoint() {
    };

    TrackPoint(sl::float3 pos, sl::OBJECT_TRACKING_STATE state, uint64_t timestamp_) {
        x = pos.x;
        y = pos.y;
        z = pos.z;

        if (state == sl::OBJECT_TRACKING_STATE::OK) {
            tracking_state = TrackPointState::OK;
        } else {
            tracking_state = TrackPointState::OFF;
        }

        timestamp = timestamp_;
    };

    TrackPoint(sl::float3 pos, TrackPointState state, uint64_t timestamp_) {
        x = pos.x;
        y = pos.y;
        z = pos.z;

        tracking_state = state;

        timestamp = timestamp_;
    };

    sl::float3 toSLFloat() {
        return sl::float3(x, y, z);
    }

    float x, y, z;
    uint64_t timestamp;
    TrackPointState tracking_state;
};

class Tracklet {
public:

    Tracklet(const sl::ObjectData obj, sl::OBJECT_CLASS type, uint64_t timestamp = 0) {
        id = obj.id;
        positions.push_back(TrackPoint(obj.position, obj.tracking_state, timestamp));
        positions_to_draw.push_back(TrackPoint(obj.position, obj.tracking_state, timestamp));
        tracking_state = obj.tracking_state;
        last_detected_timestamp = timestamp;
        recovery_cpt = recovery_length;
        is_alive = true;
        object_type = type;
    };

    void addDetectedPoint(const sl::ObjectData obj, uint64_t timestamp, int smoothing_window_size = 0);

    unsigned int id;
    std::deque<TrackPoint> positions; // Will store detected positions and the predicted ones
    std::deque<TrackPoint> positions_to_draw; // Will store the visualization output => when smoothing track, point won't be the same as the real points
    sl::OBJECT_TRACKING_STATE tracking_state;
    sl::OBJECT_CLASS object_type;
    uint64_t last_detected_timestamp;
    int recovery_cpt;

    // Track state
    bool is_alive;

private:
    static int const recovery_length = 10;
};

class TrackingViewer {
public:
    TrackingViewer();

    ~TrackingViewer() {
    };

    void generate_view(sl::Objects &objects, sl::Pose current_camera_pose, cv::Mat &tracking_view, bool tracking_enabled = true);

    // Window dimension getter

    int getWindowWidth() {
        return window_width;
    };

    int getWindowHeight() {
        return window_height;
    };

    // Tracking viewer configuration
    void setFPS(const int fps_, bool configure_all = true);

    void setHistorySize(const size_t history_size_) {
        history_size = history_size_;
    };

    void setMaxMissingPoints(const int m) {
        max_missing_points = m;
    };

    void setCameraCalibration(const sl::CalibrationParameters calib) {
        camera_calibration = calib;
    };

    void setMinLengthToDraw(const int l) {
        min_length_to_draw = l;
    };

    void setZMin(const float z_) {
        z_min = z_;
        x_min = z_ / 2.0f;
        x_max = -x_min;

        x_step = (x_max - x_min) / window_width;
        z_step = abs(z_min) / (window_height - camera_offset);
    };

    // Zoom functions
    void zoomIn();
    void zoomOut();

    void configureFromFPS();

    void toggleSmoothTracks() {
        do_smooth = !do_smooth;
    }

private:
    float x_min, x_max; // show objects between [x_min; x_max] (in millimeters)
    float z_min; // show objects between [z_min; 0] (z_min < 0) (in millimeters)

    // Conversion from world position to pixel coordinates
    float x_step, z_step;

    // window size
    int window_width, window_height;

    // Keep tracks of alive tracks
    std::vector<Tracklet> tracklets;

    // history management
    size_t history_size;
    int min_length_to_draw;

    // Visualization configuration
    cv::Mat background;
    bool has_background_ready;
    cv::Scalar end_of_track_color, background_color, fov_color;
    int camera_offset;

    // To keep track of frames
    int max_missing_points;
    int fps;
    uint64_t frame_time_step;

    // Camera settings
    sl::CalibrationParameters camera_calibration;
    float fov;

    // SMOOTH
    bool do_smooth;
    int smoothing_window_size;

    // ----------- Private methods ----------------------
    void addToTracklets(sl::Objects &objects);
    void detectUnchangedTrack(uint64_t current_timestamp);
    void pruneOldPoints();
    void computeFOV();
    void zoom(const float factor);

    // Utils
    cv::Point2i toCVPoint(double x, double z);
    // Utils with pose information
    cv::Point2i toCVPoint(sl::float3 position, sl::Pose pose);
    cv::Point2i toCVPoint(TrackPoint position, sl::Pose pose);

    // vizualization methods
    void drawTracklets(cv::Mat &tracking_view, sl::Pose current_camera_pose);
    void drawPosition(sl::Objects &objects, cv::Mat &tracking_view, sl::Pose current_camera_pose);
    void drawScale(cv::Mat &tracking_view);

    // background generation
    void generateBackground();
    void drawCamera();
    void drawHotkeys();
};

#endif