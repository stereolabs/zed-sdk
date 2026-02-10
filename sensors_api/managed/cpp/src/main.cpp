/**
 * @file main.cpp
 * @brief Sensors API Sample - Demonstrates multi-sensor streaming with ZED cameras and LiDAR
 *
 * This sample shows how to use the sl::Sensors API to manage multiple sensors
 * (ZED cameras and LiDAR) with a unified interface. It supports:
 * - Live streaming from connected devices
 * - Reading from recorded files (SVO for cameras, OSF for LiDAR)
 * - Recording to files while streaming
 * - 3D visualization with OpenGL or Rerun
 *
 * Controls:
 * - R: Toggle recording (SVO for cameras, OSF for LiDAR)
 * - WASD: Move camera in 3D view
 * - Mouse drag: Rotate view
 * - Q/ESC: Quit
 *
 * Usage:
 *   ./SensorsAPISample                              # Auto-detect all connected sensors
 *   ./SensorsAPISample sensors_config.json          # Use JSON configuration
 *   ./SensorsAPISample recording.svo                # Play back SVO file (fast mode)
 *   ./SensorsAPISample recording.osf                # Play back OSF file (fast mode)
 *   ./SensorsAPISample *.svo *.osf --realtime       # Play back at recorded frame rate
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <set>
#include <ctime>
#include <cmath>
#include <unordered_map>

#include <sl/Camera.hpp>
#include <sl/CameraOne.hpp>
#include <sl/Lidar.hpp>
#include <sl/Sensors.hpp>

#include "GLViewer.hpp"
#include "RerunViewer.hpp"
#include "json.hpp"

using json = nlohmann::json;

// =============================================================================
// Helper Types and Globals
// =============================================================================

enum class ViewerType {
    OpenGL,
    Rerun,
    Headless
};
enum class InputMode {
    Live,
    Playback
};
enum class DeviceType {
    LIDAR,
    ZED,
    ZED_ONE
};
enum class ReferenceMode {
    Sensor,   // Raw sensor data, no transform (default)
    Baselink, // Apply sensor poses for correct relative positioning
    World     // Enable positional tracking, transform to world frame
};

struct DeviceConfig {
    DeviceType type = DeviceType::ZED;
    sl::InputType input;
    sl::Matrix4f pose;
    bool has_pose = false;
    bool enable_object_detection = false;
    bool enable_body_tracking = false;
};

// Detection configuration
struct DetectionConfig {
    bool object_detection = false;
    bool body_tracking = false;
    int object_detection_confidence = 50;
    int body_tracking_confidence = 50;
    sl::OBJECT_DETECTION_MODEL object_model = sl::OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_ACCURATE;
    sl::BODY_TRACKING_MODEL body_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;
    sl::BODY_FORMAT body_format = sl::BODY_FORMAT::BODY_38;
};

struct RecordingState {
    std::atomic<bool> isRecording {false};
    std::atomic<bool> toggleRequested {false};
    std::string lastStatus;
    std::mutex statusMutex;
    std::string outputDir;
};

static RecordingState g_recordingState;
static ReferenceMode g_referenceMode = ReferenceMode::Baselink;
static DetectionConfig g_detectionConfig;

// Helper function to generate timestamped recording filename
std::string generateRecordingFilename(const std::string& prefix, const std::string& extension) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm);
    return prefix + "_" + buffer + extension;
}
static RerunOptions g_rerunOptions;

// =============================================================================
// Configuration Parsing
// =============================================================================

// Helper to build a pose matrix from rotation (Rodrigues vector) and translation
sl::Matrix4f buildPoseFromRotationTranslation(const std::vector<float>& rotation, const std::vector<float>& translation) {
    sl::Transform pose;
    pose.setIdentity();

    if (rotation.size() >= 3) {
        sl::float3 rotVec(rotation[0], rotation[1], rotation[2]);
        pose.setRotationVector(rotVec);
    }

    if (translation.size() >= 3) {
        pose.setTranslation(sl::float3(translation[0], translation[1], translation[2]));
    }

    return pose;
}

bool parseJsonConfig(const std::string& path, std::vector<DeviceConfig>& devices) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << path << std::endl;
        return false;
    }

    json config;
    try {
        file >> config;
    } catch (const json::exception& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return false;
    }

    // Parse ZED cameras
    if (config.contains("zeds") && config["zeds"].is_array()) {
        for (const auto& zed : config["zeds"]) {
            DeviceConfig cfg;
            cfg.type = DeviceType::ZED;

            if (zed.contains("serial")) {
                cfg.input.setFromSerialNumber(zed["serial"].get<int>());
            }
            if (zed.contains("svo")) {
                cfg.input.setFromSVOFile(zed["svo"].get<std::string>().c_str());
            }
            // Support pose as 16-element matrix
            if (zed.contains("pose") && zed["pose"].is_array() && zed["pose"].size() == 16) {
                auto& p = zed["pose"];
                for (int i = 0; i < 16; i++) {
                    cfg.pose.m[i] = p[i].get<float>();
                }
                cfg.has_pose = true;
            }
            // Support rotation (Euler angles) + translation
            else if (zed.contains("rotation") || zed.contains("translation")) {
                std::vector<float> rotation = {0, 0, 0};
                std::vector<float> translation = {0, 0, 0};
                if (zed.contains("rotation") && zed["rotation"].is_array()) {
                    for (size_t i = 0; i < std::min(zed["rotation"].size(), (size_t)3); i++) {
                        rotation[i] = zed["rotation"][i].get<float>();
                    }
                }
                if (zed.contains("translation") && zed["translation"].is_array()) {
                    for (size_t i = 0; i < std::min(zed["translation"].size(), (size_t)3); i++) {
                        translation[i] = zed["translation"][i].get<float>();
                    }
                }
                cfg.pose = buildPoseFromRotationTranslation(rotation, translation);
                cfg.has_pose = true;
                std::cout << "  Loaded pose for ZED: rot=[" << rotation[0] << "," << rotation[1] << "," << rotation[2] << "] trans=["
                          << translation[0] << "," << translation[1] << "," << translation[2] << "]" << std::endl;
            }
            // Per-camera detection settings
            if (zed.contains("object_detection")) {
                cfg.enable_object_detection = zed["object_detection"].get<bool>();
            }
            if (zed.contains("body_tracking")) {
                cfg.enable_body_tracking = zed["body_tracking"].get<bool>();
            }
            devices.push_back(cfg);
        }
    }

    // Parse LiDARs
    if (config.contains("lidars") && config["lidars"].is_array()) {
        for (const auto& lidar : config["lidars"]) {
            DeviceConfig cfg;
            cfg.type = DeviceType::LIDAR;

            if (lidar.contains("ip")) {
                cfg.input.setFromStream(sl::String(lidar["ip"].get<std::string>().c_str()));
            }
            if (lidar.contains("osf")) {
                cfg.input.setFromSVOFile(sl::String(lidar["osf"].get<std::string>().c_str()));
            }
            if (lidar.contains("svo")) {
                cfg.input.setFromSVOFile(sl::String(lidar["svo"].get<std::string>().c_str()));
            }
            // Support pose as 16-element matrix
            if (lidar.contains("pose") && lidar["pose"].is_array() && lidar["pose"].size() == 16) {
                auto& p = lidar["pose"];
                for (int i = 0; i < 16; i++) {
                    cfg.pose.m[i] = p[i].get<float>();
                }
                cfg.has_pose = true;
            }
            // Support rotation (Euler angles) + translation
            else if (lidar.contains("rotation") || lidar.contains("translation")) {
                std::vector<float> rotation = {0, 0, 0};
                std::vector<float> translation = {0, 0, 0};
                if (lidar.contains("rotation") && lidar["rotation"].is_array()) {
                    for (size_t i = 0; i < std::min(lidar["rotation"].size(), (size_t)3); i++) {
                        rotation[i] = lidar["rotation"][i].get<float>();
                    }
                }
                if (lidar.contains("translation") && lidar["translation"].is_array()) {
                    for (size_t i = 0; i < std::min(lidar["translation"].size(), (size_t)3); i++) {
                        translation[i] = lidar["translation"][i].get<float>();
                    }
                }
                cfg.pose = buildPoseFromRotationTranslation(rotation, translation);
                cfg.has_pose = true;
                std::cout << "  Loaded pose for LiDAR: rot=[" << rotation[0] << "," << rotation[1] << "," << rotation[2] << "] trans=["
                          << translation[0] << "," << translation[1] << "," << translation[2] << "]" << std::endl;
            }
            devices.push_back(cfg);
        }
    }

    // Parse global detection settings
    if (config.contains("object_detection")) {
        auto& od = config["object_detection"];
        g_detectionConfig.object_detection = od.value("enabled", false);
        g_detectionConfig.object_detection_confidence = od.value("confidence", 50);
        if (od.contains("model")) {
            std::string model = od["model"].get<std::string>();
            if (model == "accurate" || model == "MULTI_CLASS_BOX_ACCURATE") {
                g_detectionConfig.object_model = sl::OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_ACCURATE;
            } else if (model == "medium" || model == "MULTI_CLASS_BOX_MEDIUM") {
                g_detectionConfig.object_model = sl::OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_MEDIUM;
            } else if (model == "fast" || model == "MULTI_CLASS_BOX_FAST") {
                g_detectionConfig.object_model = sl::OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_FAST;
            }
        }
        std::cout << "  Object detection: " << (g_detectionConfig.object_detection ? "enabled" : "disabled")
                  << ", confidence=" << g_detectionConfig.object_detection_confidence << std::endl;
    }
    if (config.contains("body_tracking")) {
        auto& bt = config["body_tracking"];
        g_detectionConfig.body_tracking = bt.value("enabled", false);
        g_detectionConfig.body_tracking_confidence = bt.value("confidence", 50);
        if (bt.contains("model")) {
            std::string model = bt["model"].get<std::string>();
            if (model == "accurate" || model == "HUMAN_BODY_ACCURATE") {
                g_detectionConfig.body_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;
            } else if (model == "medium" || model == "HUMAN_BODY_MEDIUM") {
                g_detectionConfig.body_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
            } else if (model == "fast" || model == "HUMAN_BODY_FAST") {
                g_detectionConfig.body_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_FAST;
            }
        }
        if (bt.contains("format")) {
            std::string format = bt["format"].get<std::string>();
            if (format == "18" || format == "BODY_18") {
                g_detectionConfig.body_format = sl::BODY_FORMAT::BODY_18;
            } else if (format == "34" || format == "BODY_34") {
                g_detectionConfig.body_format = sl::BODY_FORMAT::BODY_34;
            } else if (format == "38" || format == "BODY_38") {
                g_detectionConfig.body_format = sl::BODY_FORMAT::BODY_38;
            }
        }
        std::cout << "  Body tracking: " << (g_detectionConfig.body_tracking ? "enabled" : "disabled")
                  << ", confidence=" << g_detectionConfig.body_tracking_confidence << std::endl;
    }

    return true;
}

// =============================================================================
// Recording Status Callback
// =============================================================================

std::string getRecordingStatus() {
    std::lock_guard<std::mutex> lock(g_recordingState.statusMutex);
    return g_recordingState.lastStatus;
}

void updateRecordingStatus(const std::string& status) {
    std::lock_guard<std::mutex> lock(g_recordingState.statusMutex);
    g_recordingState.lastStatus = status;
}

// =============================================================================
// Helper functions for sensor parameters
// =============================================================================

sl::InitLidarParameters getLidarParams(const sl::InputType& input, bool realTimeMode = false) {
    sl::InitLidarParameters params;
    params.input = input;
    params.mode = sl::LIDAR_MODE::AUTO;
    params.depth_minimum_distance = 1.0f;
    params.depth_maximum_distance = 80.0f;
    params.coordinate_units = sl::UNIT::METER;
    params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    params.svo_real_time_mode = realTimeMode;
    return params;
}

sl::InitParameters getZedParams(const sl::InputType& input, bool isPlayback = false) {
    sl::InitParameters params;
    params.input = input;
    params.depth_mode = sl::DEPTH_MODE::NEURAL;
    params.coordinate_units = sl::UNIT::METER;
    params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    // For playback, disable real-time mode to read all frames without timing-based drops
    // For live capture, enable real-time mode for natural pacing
    params.svo_real_time_mode = !isPlayback;
    params.async_grab_camera_recovery = true;
    return params;
}

sl::InitParametersOne getZedOneParams(const sl::InputType& input) {
    sl::InitParametersOne params;
    params.input = input;
    params.coordinate_units = sl::UNIT::METER;
    params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    return params;
}

// =============================================================================
// Main Sensors API Implementation
// =============================================================================

void runSensorsImpl(int argc, char** argv, ViewerType viewerType) {
    std::cout << "\n=== Sensors API Sample ===" << std::endl;

    std::vector<DeviceConfig> devices;
    InputMode inputMode = InputMode::Live;
    bool realTimePlayback = false; // If true, playback uses real-time pacing based on timestamps

    // Parse command line arguments - support multiple input files
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        // Check for --realtime flag
        if (arg == "--realtime" || arg == "-rt") {
            realTimePlayback = true;
            std::cout << "Real-time playback mode enabled" << std::endl;
            continue;
        }

        // Check for reference frame mode flags
        if (arg == "--baselink" || arg == "-b") {
            g_referenceMode = ReferenceMode::Baselink;
            std::cout << "Reference mode: BASELINK (sensor poses applied)" << std::endl;
            continue;
        }
        if (arg == "--world" || arg == "-w") {
            g_referenceMode = ReferenceMode::World;
            std::cout << "Reference mode: WORLD (positional tracking enabled)" << std::endl;
            continue;
        }
        if (arg == "--sensor" || arg == "-s") {
            g_referenceMode = ReferenceMode::Sensor;
            std::cout << "Reference mode: SENSOR (raw data, no transform)" << std::endl;
            continue;
        }

        // Check for object detection flag
        if (arg == "--object-detection" || arg == "-od") {
            g_detectionConfig.object_detection = true;
            std::cout << "Object detection enabled" << std::endl;
            continue;
        }

        // Check for body tracking flag
        if (arg == "--body-tracking" || arg == "-bt") {
            g_detectionConfig.body_tracking = true;
            std::cout << "Body tracking enabled" << std::endl;
            continue;
        }

        // Check if it's a JSON config file
        if (arg.size() > 5 && arg.substr(arg.size() - 5) == ".json") {
            if (!parseJsonConfig(arg, devices)) {
                return;
            }
            inputMode = InputMode::Playback;
            std::cout << "Loaded JSON config with " << devices.size() << " devices" << std::endl;
        }
        // Check if it's an SVO file (Camera playback)
        else if (arg.size() > 4 && (arg.substr(arg.size() - 4) == ".svo" || arg.substr(arg.size() - 5) == ".svo2")) {
            DeviceConfig cfg;
            cfg.type = DeviceType::ZED;
            cfg.input.setFromSVOFile(arg.c_str());
            devices.push_back(cfg);
            inputMode = InputMode::Playback;
            std::cout << "Playing back SVO file: " << arg << std::endl;
        }
        // Check if it's an OSF file (LiDAR playback)
        else if (arg.size() > 4 && arg.substr(arg.size() - 4) == ".osf") {
            DeviceConfig cfg;
            cfg.type = DeviceType::LIDAR;
            cfg.input.setFromSVOFile(sl::String(arg.c_str()));
            devices.push_back(cfg);
            inputMode = InputMode::Playback;
            std::cout << "Playing back OSF file: " << arg << std::endl;
        } else {
            std::cerr << "Unknown file type: " << arg << std::endl;
            std::cerr << "Supported: .json (config), .svo/.svo2 (camera), .osf (lidar)" << std::endl;
            return;
        }
    }

    // If no config provided, auto-detect devices
    if (devices.empty()) {
        std::cout << "No config provided, auto-detecting devices..." << std::endl;

        // Detect ZED cameras
        auto zedDevices = sl::Camera::getDeviceList();
        for (const auto& dev : zedDevices) {
            DeviceConfig cfg;
            cfg.type = DeviceType::ZED;
            cfg.input.setFromSerialNumber(dev.serial_number);
            devices.push_back(cfg);
            std::cout << "  Found ZED camera: " << dev.serial_number << std::endl;
        }

        // Detect ZED One cameras
        auto zedOneDevices = sl::CameraOne::getDeviceList();
        for (const auto& dev : zedOneDevices) {
            DeviceConfig cfg;
            cfg.type = DeviceType::ZED_ONE;
            cfg.input.setFromCameraID(dev.id);
            devices.push_back(cfg);
            std::cout << "  Found ZED One camera: " << dev.serial_number << std::endl;
        }

        // Detect LiDARs
        auto lidarDevices = sl::Lidar::getDeviceList();
        for (const auto& dev : lidarDevices) {
            DeviceConfig cfg;
            cfg.type = DeviceType::LIDAR;
            cfg.input.setFromStream(sl::String(dev.ip_address.c_str()));
            devices.push_back(cfg);
            std::cout << "  Found LiDAR: " << dev.name << " (" << dev.ip_address << ")" << std::endl;
        }

        if (devices.empty()) {
            std::cerr << "No devices found!" << std::endl;
            return;
        }
    }

    // Create and initialize Sensors manager
    sl::Sensors sensors;
    sl::InitSensorsParameters init_sensors_params;
    init_sensors_params.coordinate_units = sl::UNIT::METER;
    init_sensors_params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;

    std::cout << "Initializing Sensors API..." << std::endl;
    auto init_err = sensors.init(init_sensors_params);
    if (init_err != sl::SENSORS_ERROR_CODE::SUCCESS) {
        std::cerr << "Failed to init Sensors API: " << init_err << std::endl;
        return;
    }

    // Add devices
    std::vector<sl::SensorDeviceIdentifier> lidar_ids;
    std::vector<sl::SensorDeviceIdentifier> zed_ids;
    std::vector<sl::SensorDeviceIdentifier> zed_one_ids;

    for (const auto& dev : devices) {
        sl::SensorDeviceIdentifier id;
        sl::SENSORS_ERROR_CODE err;

        if (dev.type == DeviceType::LIDAR) {
            // For playback: use realTimePlayback flag to control pacing
            // For live: always use real-time (natural sensor rate)
            bool useRealTime = (inputMode == InputMode::Live) || realTimePlayback;
            auto params = getLidarParams(dev.input, useRealTime);
            std::cout << "Opening LiDAR at " << params.input.getConfiguration() << "..." << std::endl;
            err = sensors.add(params, id);
            if (err == sl::SENSORS_ERROR_CODE::SUCCESS) {
                if (dev.has_pose)
                    sensors.setSensorPose(sl::Transform(dev.pose), id);
                lidar_ids.push_back(id);
                std::cout << "  LiDAR Added. ID: " << id.getID() << std::endl;
            }
        } else if (dev.type == DeviceType::ZED) {
            // For playback: use realTimePlayback to pace based on timestamps
            // Default playback (no --realtime): read as fast as possible
            bool useRealTime = (inputMode == InputMode::Live) || realTimePlayback;
            auto params = getZedParams(dev.input, !useRealTime); // isPlayback inverts the logic
            std::cout << "Opening ZED at " << params.input.getConfiguration() << "..." << std::endl;
            err = sensors.add(params, id);
            if (err == sl::SENSORS_ERROR_CODE::SUCCESS) {
                if (dev.has_pose)
                    sensors.setSensorPose(sl::Transform(dev.pose), id);
                zed_ids.push_back(id);
                std::cout << "  ZED Added. ID: " << id.getID() << std::endl;
            }
        } else if (dev.type == DeviceType::ZED_ONE) {
            auto params = getZedOneParams(dev.input);
            std::cout << "Opening ZED One at " << params.input.getConfiguration() << "..." << std::endl;
            err = sensors.add(params, id);
            if (err == sl::SENSORS_ERROR_CODE::SUCCESS) {
                if (dev.has_pose)
                    sensors.setSensorPose(sl::Transform(dev.pose), id);
                zed_one_ids.push_back(id);
                std::cout << "  ZED One Added. ID: " << id.getID() << std::endl;
            }
        }

        if (err != sl::SENSORS_ERROR_CODE::SUCCESS) {
            std::cerr << "  Failed to add device: " << err << std::endl;
        }
    }

    // Check if we have any active sensors
    if (lidar_ids.empty() && zed_ids.empty() && zed_one_ids.empty()) {
        std::cerr << "No sensors were successfully added!" << std::endl;
        return;
    }

    // In playback mode, synchronize all SVO/OSF files to a common start timestamp
    if (inputMode == InputMode::Playback) {
        std::cout << "Synchronizing playback files to common start timestamp..." << std::endl;
        auto sync_err = sensors.syncSVO();
        if (sync_err != sl::SENSORS_ERROR_CODE::SUCCESS) {
            std::cerr << "Warning: Failed to sync some files to common timestamp: " << sync_err << std::endl;
        } else {
            std::cout << "Playback synchronized successfully" << std::endl;
        }
    }

    // Configure reference frame mode
    sl::RuntimeSensorsParameters runtime_params;
    switch (g_referenceMode) {
        case ReferenceMode::Sensor:
            runtime_params.reference_frame = sl::SENSORS_REFERENCE_FRAME::SENSOR;
            std::cout << "Using SENSOR reference frame (raw data)" << std::endl;
            break;
        case ReferenceMode::Baselink:
            runtime_params.reference_frame = sl::SENSORS_REFERENCE_FRAME::BASELINK;
            std::cout << "Using BASELINK reference frame (sensor poses applied)" << std::endl;
            break;
        case ReferenceMode::World:
            runtime_params.reference_frame = sl::SENSORS_REFERENCE_FRAME::WORLD;
            std::cout << "Using WORLD reference frame (positional tracking)" << std::endl;
            // Enable positional tracking for WORLD mode
            sl::PositionalTrackingSensorsParameters tracking_params;
            auto tracking_err = sensors.enablePositionalTracking(tracking_params);
            if (tracking_err != sl::SENSORS_ERROR_CODE::SUCCESS) {
                std::cerr << "Warning: Failed to enable positional tracking: " << tracking_err << std::endl;
                std::cerr << "Falling back to BASELINK mode" << std::endl;
                runtime_params.reference_frame = sl::SENSORS_REFERENCE_FRAME::BASELINK;
            } else {
                std::cout << "Positional tracking enabled" << std::endl;
            }
            break;
    }

    // Apply runtime parameters
    sensors.setRuntimeSensorsParameters(runtime_params);

    // Enable Object Detection if requested
    if (g_detectionConfig.object_detection && !zed_ids.empty()) {
        std::cout << "Enabling Object Detection..." << std::endl;

        // First enable positional tracking (required for object detection)
        if (g_referenceMode != ReferenceMode::World) {
            sl::PositionalTrackingSensorsParameters tracking_params;
            auto tracking_err = sensors.enablePositionalTracking(tracking_params);
            if (tracking_err != sl::SENSORS_ERROR_CODE::SUCCESS) {
                std::cerr << "Warning: Failed to enable positional tracking for object detection: " << tracking_err << std::endl;
            }
        }

        sl::ObjectDetectionSensorsParameters od_params;
        od_params.detection_model = g_detectionConfig.object_model;
        od_params.enable_tracking = true;
        od_params.enable_segmentation = false;

        auto od_err = sensors.enableObjectDetection(od_params);
        if (od_err != sl::SENSORS_ERROR_CODE::SUCCESS) {
            std::cerr << "Failed to enable Object Detection: " << od_err << std::endl;
            g_detectionConfig.object_detection = false;
        } else {
            std::cout << "Object Detection enabled successfully" << std::endl;
        }
    }

    // Enable Body Tracking if requested
    if (g_detectionConfig.body_tracking && !zed_ids.empty()) {
        std::cout << "Enabling Body Tracking..." << std::endl;

        // First enable positional tracking (required for body tracking)
        if (g_referenceMode != ReferenceMode::World && !g_detectionConfig.object_detection) {
            sl::PositionalTrackingSensorsParameters tracking_params;
            auto tracking_err = sensors.enablePositionalTracking(tracking_params);
            if (tracking_err != sl::SENSORS_ERROR_CODE::SUCCESS) {
                std::cerr << "Warning: Failed to enable positional tracking for body tracking: " << tracking_err << std::endl;
            }
        }

        sl::BodyTrackingSensorsParameters bt_params;
        bt_params.detection_model = g_detectionConfig.body_model;
        bt_params.body_format = g_detectionConfig.body_format;
        bt_params.enable_tracking = true;
        bt_params.enable_body_fitting = true;
        bt_params.enable_segmentation = false;

        auto bt_err = sensors.enableBodyTracking(bt_params);
        if (bt_err != sl::SENSORS_ERROR_CODE::SUCCESS) {
            std::cerr << "Failed to enable Body Tracking: " << bt_err << std::endl;
            g_detectionConfig.body_tracking = false;
        } else {
            std::cout << "Body Tracking enabled successfully" << std::endl;
        }
    }

    // Initialize viewer (optional in headless mode)
    std::unique_ptr<Viewer> viewer;
    GLViewer* glViewer = nullptr;
    bool headless = (viewerType == ViewerType::Headless);

    if (!headless) {
        if (viewerType == ViewerType::OpenGL) {
            glViewer = new GLViewer();
            viewer.reset(glViewer);
            glViewer->setRecordingCallback(getRecordingStatus);
            glViewer->setRecordingToggleCallback([]() {
                g_recordingState.toggleRequested = true;
            });
        } else {
            viewer.reset(new RerunViewer(g_rerunOptions));
        }
        viewer->init(argc, argv);

        if (!viewer->isAvailable()) {
            std::cerr << "Failed to initialize viewer!" << std::endl;
            return;
        }
    } else {
        std::cout << "Running in headless mode (no display)" << std::endl;
    }

    // Pre-allocate batched data structures
    sl::BatchedData<sl::Mat> batched_measures;
    sl::BatchedData<sl::Mat> batched_images;

    // Per-camera detection results (Sensors API retrieves per-device)
    std::unordered_map<std::string, sl::Objects> camera_objects;
    std::unordered_map<std::string, sl::Bodies> camera_bodies;

    // Runtime parameters for detection retrieval
    sl::ObjectDetectionRuntimeParameters od_runtime_params;
    od_runtime_params.detection_confidence_threshold = g_detectionConfig.object_detection_confidence;

    sl::BodyTrackingRuntimeParameters bt_runtime_params;
    bt_runtime_params.detection_confidence_threshold = g_detectionConfig.body_tracking_confidence;
    bt_runtime_params.skeleton_smoothing = 0.7f;

    std::cout << "\nStarting sensor streaming..." << std::endl;
    if (!headless) {
        std::cout << "Controls: R=toggle recording, WASD=move, Mouse=look, Q=quit" << std::endl;
    } else {
        std::cout << "Press Ctrl+C to stop" << std::endl;
    }

    int frameCount = 0;
    auto startTime = std::chrono::steady_clock::now();
    std::atomic<bool> running {true};

    // Main loop
    while (running && (headless || (viewer->isAvailable() && !viewer->isDone()))) {
        // Handle recording toggle
        if (g_recordingState.toggleRequested.exchange(false)) {
            if (g_recordingState.isRecording) {
                // Stop recording
                sensors.disableRecording();
                g_recordingState.isRecording = false;
                updateRecordingStatus("Recording stopped");
                std::cout << "Recording stopped" << std::endl;
            } else {
                // Start recording - create filenames for all devices
                sl::RecordingSensorsParameters rec_params;

                // Set output filenames for each device
                for (const auto& id : zed_ids) {
                    std::string filename = generateRecordingFilename("zed_" + std::to_string(id.getSerialNumber()), ".svo2");
                    rec_params.video_filenames[id] = sl::String(filename.c_str());
                    std::cout << "  Camera " << id.getSerialNumber() << " -> " << filename << std::endl;
                }
                for (const auto& id : zed_one_ids) {
                    std::string filename = generateRecordingFilename("zedone_" + std::to_string(id.getSerialNumber()), ".svo2");
                    rec_params.video_filenames[id] = sl::String(filename.c_str());
                    std::cout << "  CameraOne " << id.getSerialNumber() << " -> " << filename << std::endl;
                }
                for (const auto& id : lidar_ids) {
                    std::string filename = generateRecordingFilename(std::string("lidar_") + id.sensor_name.c_str(), ".osf");
                    rec_params.video_filenames[id] = sl::String(filename.c_str());
                    std::cout << "  LiDAR " << id.sensor_name.c_str() << " -> " << filename << std::endl;
                }

                // Set compression mode for cameras
                rec_params.compression_mode = sl::SVO_COMPRESSION_MODE::H265;

                auto rec_err = sensors.enableRecording(rec_params);
                if (rec_err == sl::SENSORS_ERROR_CODE::SUCCESS) {
                    g_recordingState.isRecording = true;
                    updateRecordingStatus("RECORDING");
                    std::cout << "Recording started" << std::endl;
                } else {
                    std::cerr << "Failed to start recording: " << rec_err << std::endl;
                    updateRecordingStatus("Recording failed: " + std::to_string(static_cast<int>(rec_err)));
                }
            }
        }

        // Process data from all sensors
        auto process_status = sensors.process();
        if (process_status != sl::SENSORS_ERROR_CODE::SUCCESS) {
            if (inputMode == InputMode::Playback && (process_status == sl::SENSORS_ERROR_CODE::END_OF_SVOFILE_REACHED)) {
                std::cout << "End of all playback files reached" << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Retrieve data
        // Note: LiDAR currently only supports CPU retrieval through Sensors API
        // ZED cameras support both, but we use CPU for simplicity/compatibility
        // Resolution(-1,-1) = optimal size (cameras use internal compute size, LiDAR uses native resolution)
        sensors.retrieveMeasure(batched_measures, sl::MEASURE::XYZRGBA, sl::MEM::CPU, sl::Resolution(-1, -1));
        sensors.retrieveImage(batched_images, sl::VIEW::LEFT, sl::MEM::CPU);

        // Retrieve object detection and body tracking results per camera
        if (g_detectionConfig.object_detection || g_detectionConfig.body_tracking) {
            for (const auto& id : zed_ids) {
                std::string id_str = std::to_string(id.getSerialNumber());

                // Retrieve object detection results
                if (g_detectionConfig.object_detection) {
                    sensors.retrieveObjects(camera_objects[id_str], id);
                }

                // Retrieve body tracking results
                if (g_detectionConfig.body_tracking) {
                    sensors.retrieveBodies(camera_bodies[id_str], id);
                }
            }
        }

        // Update viewer with LiDAR data (if not headless)
        if (!headless) {
            for (const auto& id : lidar_ids) {
                if (batched_measures.find(id) != batched_measures.end() && batched_images.find(id) != batched_images.end()) {
                    sl::Mat& point_cloud = batched_measures[id];
                    sl::Mat& intensity_img = batched_images[id];
                    viewer->updateLidar(std::to_string(id.getSerialNumber()), point_cloud, intensity_img);
                }
            }

            // Update viewer with ZED data
            for (const auto& id : zed_ids) {
                std::string id_str = std::to_string(id.getSerialNumber());

                if (batched_images.find(id) != batched_images.end()) {
                    sl::Mat& image = batched_images[id];
                    sl::Mat* point_cloud = nullptr;
                    if (batched_measures.find(id) != batched_measures.end()) {
                        point_cloud = &batched_measures[id];
                    }
                    viewer->updateZed(id_str, image, point_cloud);
                }

                // Update viewer with object detection results
                if (g_detectionConfig.object_detection && camera_objects.find(id_str) != camera_objects.end()) {
                    viewer->updateZedObjects(id_str, camera_objects[id_str]);
                }

                // Update viewer with body tracking results
                if (g_detectionConfig.body_tracking && camera_bodies.find(id_str) != camera_bodies.end()) {
                    viewer->updateZedBodies(id_str, camera_bodies[id_str]);
                }
            }

            // Update viewer with ZED One data (2D image only, no point cloud)
            for (const auto& id : zed_one_ids) {
                if (batched_images.find(id) != batched_images.end()) {
                    sl::Mat& image = batched_images[id];
                    viewer->updateZedOne(std::to_string(id.getSerialNumber()), image);
                }
            }
        }

        frameCount++;

        // Print FPS every 100 frames
        if (frameCount % 100 == 0) {
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            float fps = frameCount / elapsed;
            std::cout << "FPS: " << fps << " | Cameras: " << zed_ids.size() << " | CameraOnes: " << zed_one_ids.size()
                      << " | LiDARs: " << lidar_ids.size();
            if (g_recordingState.isRecording) {
                std::cout << " | RECORDING";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "\nSample finished. Total frames: " << frameCount << std::endl;
}

// =============================================================================
// Main Entry Point
// =============================================================================

void printUsage(const char* progName) {
    std::cout << "\nUsage: " << progName << " [options] [input...]" << std::endl;
    std::cout << "\nInput (optional, multiple files supported):" << std::endl;
    std::cout << "  config.json     JSON configuration file for multiple sensors" << std::endl;
    std::cout << "  recording.svo   SVO file for camera playback" << std::endl;
    std::cout << "  recording.osf   OSF file for LiDAR playback" << std::endl;
    std::cout << "  (none)          Auto-detect all connected sensors" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << progName << "                                    # Auto-detect devices" << std::endl;
    std::cout << "  " << progName << " camera.svo2                        # Play single camera" << std::endl;
    std::cout << "  " << progName << " lidar.osf                          # Play single LiDAR" << std::endl;
    std::cout << "  " << progName << " camera.svo2 lidar.osf              # Play both together" << std::endl;
    std::cout << "  " << progName << " cam1.svo2 cam2.svo2 lidar.osf      # Play multiple files" << std::endl;
    std::cout << "  " << progName << " config.json                        # Use JSON config" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --baselink, -b          Use BASELINK reference frame (default, sensor poses applied)" << std::endl;
    std::cout << "  --world, -w             Use WORLD reference frame (enable positional tracking)" << std::endl;
    std::cout << "  --sensor, -s            Use SENSOR reference frame (raw data, no transform)" << std::endl;
    std::cout << "  --realtime, -rt         Play back at recorded frame rate" << std::endl;
    std::cout << "  --object-detection, -od Enable object detection on ZED cameras" << std::endl;
    std::cout << "  --body-tracking, -bt    Enable body tracking on ZED cameras" << std::endl;
    std::cout << "  --rerun                 Use Rerun viewer instead of OpenGL" << std::endl;
    std::cout << "  --connect <host:port>   Connect Rerun to remote viewer (e.g., 192.168.1.100:9876)" << std::endl;
    std::cout << "  --save <file.rrd>       Save Rerun data to file for offline viewing" << std::endl;
    std::cout << "  --headless              Run without display (data processing only)" << std::endl;
    std::cout << "  --help, -h              Show this help message" << std::endl;
    std::cout << "\nControls during runtime:" << std::endl;
    std::cout << "  R               Toggle recording (SVO for cameras, OSF for LiDAR)" << std::endl;
    std::cout << "  W/A/S/D         Move camera in 3D view" << std::endl;
    std::cout << "  Mouse drag      Rotate view" << std::endl;
    std::cout << "  Q/ESC           Quit" << std::endl;
    std::cout << "\nJSON config example:" << std::endl;
    std::cout << R"({
  "zeds": [
    {"serial": 12345678, "object_detection": true, "body_tracking": false},
    {"svo": "recording.svo"}
  ],
  "lidars": [
    {"ip": "192.168.1.100"},
    {"osf": "recording.osf", "pose": [1,0,0,1000, 0,1,0,0, 0,0,1,500, 0,0,0,1]}
  ],
  "object_detection": {
    "enabled": true,
    "confidence": 50,
    "model": "accurate"
  },
  "body_tracking": {
    "enabled": true,
    "confidence": 50,
    "model": "accurate",
    "format": "38"
  }
})" << std::endl;
}

int main(int argc, char** argv) {
    ViewerType viewerType = ViewerType::OpenGL;

    // Process command-line options
    std::vector<char*> filteredArgs;
    filteredArgs.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--rerun") {
            viewerType = ViewerType::Rerun;
        } else if (arg == "--connect" && i + 1 < argc) {
            viewerType = ViewerType::Rerun;
            g_rerunOptions.mode = RerunMode::Connect;
            g_rerunOptions.connect_addr = argv[++i];
        } else if (arg == "--save" && i + 1 < argc) {
            viewerType = ViewerType::Rerun;
            g_rerunOptions.mode = RerunMode::Save;
            g_rerunOptions.save_path = argv[++i];
        } else if (arg == "--headless") {
            viewerType = ViewerType::Headless;
        } else if (arg == "--baselink" || arg == "-b") {
            g_referenceMode = ReferenceMode::Baselink;
        } else if (arg == "--world" || arg == "-w") {
            g_referenceMode = ReferenceMode::World;
        } else if (arg == "--sensor" || arg == "-s") {
            g_referenceMode = ReferenceMode::Sensor;
        } else if (arg == "--object-detection" || arg == "-od") {
            g_detectionConfig.object_detection = true;
        } else if (arg == "--body-tracking" || arg == "-bt") {
            g_detectionConfig.body_tracking = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            filteredArgs.push_back(argv[i]);
        }
    }

    // Update argc/argv with filtered arguments
    int newArgc = filteredArgs.size();
    char** newArgv = filteredArgs.data();

    try {
        runSensorsImpl(newArgc, newArgv, viewerType);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
