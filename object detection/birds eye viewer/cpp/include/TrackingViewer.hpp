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

void render_2D(cv::Mat &left, sl::float2 img_scale, std::vector<sl::ObjectData> &objects, bool render_mask = false, bool isTrackingON = false);


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

    // duration: duration of the trajectory in seconds
    TrackingViewer(sl::Resolution res, const int fps_, const float D_max, const int duration);

    ~TrackingViewer() {
    };

    void generate_view(sl::Objects &objects, sl::Pose current_camera_pose, cv::Mat &tracking_view, bool tracking_enabled);

    void setCameraCalibration(const sl::CalibrationParameters calib) {
        camera_calibration = calib;
        has_background_ready = false;
    };

    // Zoom functions
    void zoomIn();
    void zoomOut();
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
    uint64_t history_duration; //in ns
    int min_length_to_draw;

    // Visualization configuration
    cv::Mat background;
    bool has_background_ready;
    cv::Scalar background_color, fov_color;
    int camera_offset;

    // Camera settings
    sl::CalibrationParameters camera_calibration;
    float fov;

    // SMOOTH
    bool do_smooth;
    int smoothing_window_size;

    // ----------- Private methods ----------------------
    void addToTracklets(sl::Objects &objects);
    void detectUnchangedTrack(uint64_t current_timestamp);
    void pruneOldPoints(uint64_t current_timestamp);
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
