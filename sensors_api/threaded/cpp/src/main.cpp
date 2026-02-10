/**
 * @file main.cpp
 * @brief Multi-sensor streaming sample with ZED cameras and LiDAR
 *
 * This sample demonstrates how to use sl::Camera and sl::Lidar APIs to manage
 * multiple sensors with dedicated threads for each device. It supports:
 * - Live streaming from connected devices
 * - Reading from recorded files (SVO for cameras, OSF for LiDAR)
 * - Recording to files while streaming
 * - 3D visualization with OpenGL
 * - Object detection and body tracking on ZED cameras
 * - Reference frame modes (sensor, baselink, world)
 * - Cross-device timestamp synchronization (--sync)
 * - Headless mode for data processing without display
 *
 * Run with --help for full usage information.
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
#include <ctime>
#include <unordered_map>
#include <condition_variable>

#include <sl/Camera.hpp>
#include <sl/Lidar.hpp>

#include "GLViewer.hpp"
#include "json.hpp"
#include "Asynchronizer.hpp"

using json = nlohmann::json;

// =============================================================================
// Helper Types and Globals
// =============================================================================

enum class InputMode {
    Live,
    Playback
};

enum class DeviceType {
    LIDAR,
    ZED
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
};

static RecordingState g_recordingState;
static ReferenceMode g_referenceMode = ReferenceMode::Baselink;
static DetectionConfig g_detectionConfig;
static ViewerLayout g_viewerLayout = ViewerLayout::SplitRows;
static bool g_syncMode = false;

// Mutex to serialize device open() calls — TensorRT/CUDA library loading is not thread-safe
static std::mutex g_deviceOpenMutex;

// Helper function to generate timestamped recording filename
std::string generateRecordingFilename(const std::string& prefix, const std::string& extension) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm);
    return prefix + "_" + buffer + extension;
}

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
            // Support rotation (Rodrigues vector) + translation
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
            // Support rotation (Rodrigues vector) + translation
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
// Triple Buffer for Lock-Free Data Exchange
// =============================================================================

/**
 * @brief Thread-safe triple buffer for lock-free producer-consumer data exchange
 *
 * Uses atomic index swapping to allow producers to write without blocking consumers.
 * Provides the latest available data to consumers without waiting.
 */
template <typename T>
class TripleBuffer {
public:
    TripleBuffer()
        : writeIdx_(0)
        , readIdx_(1)
        , middleIdx_(2)
        , newDataAvailable_(false) { }

    T& getWriteBuffer() {
        return buffers_[writeIdx_.load(std::memory_order_relaxed)];
    }

    void publishWrite() {
        int oldMiddle = middleIdx_.exchange(writeIdx_.load(std::memory_order_relaxed), std::memory_order_acq_rel);
        writeIdx_.store(oldMiddle, std::memory_order_release);
        newDataAvailable_.store(true, std::memory_order_release);
    }

    // Try to get new data - returns true if new data was swapped in
    bool trySwapRead() {
        if (newDataAvailable_.exchange(false, std::memory_order_acq_rel)) {
            int oldMiddle = middleIdx_.exchange(readIdx_.load(std::memory_order_relaxed), std::memory_order_acq_rel);
            readIdx_.store(oldMiddle, std::memory_order_release);
            return true;
        }
        return false;
    }

    T& getReadBuffer() {
        return buffers_[readIdx_.load(std::memory_order_relaxed)];
    }

private:
    T buffers_[3];
    std::atomic<int> writeIdx_;
    std::atomic<int> readIdx_;
    std::atomic<int> middleIdx_;
    std::atomic<bool> newDataAvailable_;
};

/**
 * @brief Per-device state for threaded processing
 */
struct DeviceState {
    std::string id;
    DeviceType type;
    sl::Transform pose;
    bool hasPose = false;
    std::atomic<bool> running {true};
    std::atomic<bool> recording {false};
    std::atomic<bool> endOfFile {false};
    std::atomic<uint64_t> frameCount {0};
    std::atomic<int64_t> lastTimestampNs {0};

    // Triple buffers for lock-free data exchange
    TripleBuffer<ZedDataPacket> zedBuffer;
    TripleBuffer<LidarDataPacket> lidarBuffer;
};

// =============================================================================
// Playback Synchronizer
// =============================================================================

/**
 * @brief Synchronization controller for multi-SVO/OSF playback
 *
 * Implements timestamp-based synchronization to align playback across
 * multiple recorded files. Each device thread waits until the global
 * playback time catches up to their next frame's timestamp.
 */
class PlaybackSynchronizer {
public:
    PlaybackSynchronizer()
        : enabled_(false)
        , playbackStarted_(false)
        , realTimeMode_(false)
        , allDevicesOpen_(false) { }

    void setRealTimeMode(bool enabled) {
        realTimeMode_ = enabled;
    }
    bool isRealTimeMode() const {
        return realTimeMode_;
    }

    void enable(size_t deviceCount) {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = true;
        deviceCount_ = deviceCount;
        openCount_ = 0;
        allDevicesOpen_ = false;
        playbackStarted_ = true;
    }

    void disable() {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = false;
        allDevicesOpen_ = true;
        cv_.notify_all();
    }

    bool isEnabled() const {
        return enabled_.load();
    }
    bool hasStarted() const {
        return playbackStarted_.load();
    }

    // Called by each device thread after opening device, before starting grab loop
    // BLOCKS until all devices are open - ensures synchronized start
    void waitForAllDevicesOpen(const std::string& deviceId) {
        std::unique_lock<std::mutex> lock(mutex_);
        openCount_++;

        if (openCount_ >= deviceCount_) {
            allDevicesOpen_ = true;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this] {
                return allDevicesOpen_.load() || !enabled_.load();
            });
        }
    }

private:
    std::atomic<bool> enabled_;
    std::atomic<bool> playbackStarted_;
    std::atomic<bool> realTimeMode_;
    std::atomic<bool> allDevicesOpen_;
    size_t deviceCount_ = 0;
    size_t openCount_ = 0;
    std::mutex mutex_;
    std::condition_variable cv_;
};

// =============================================================================
// Worker Threads
// =============================================================================

/**
 * @brief ZED Camera worker thread
 *
 * Runs grab/retrieve loop in dedicated thread, pushing data to triple buffer.
 * Optimized for minimum latency with GPU-based retrieval when possible.
 */
void zedWorkerThread(
    DeviceState& state,
    const DeviceConfig& config,
    PlaybackSynchronizer& synchronizer,
    AsyncPacketSynchronizer* asyncSync,
    InputMode inputMode,
    bool enableOD,
    bool enableBT,
    bool enableTracking
) {
    std::cout << "[ZED " << state.id << "] Worker thread starting..." << std::endl;

    sl::Camera camera;

    // Configure initialization parameters
    sl::InitParameters initParams;
    initParams.input = config.input;
    initParams.depth_mode = sl::DEPTH_MODE::NEURAL;
    initParams.coordinate_units = sl::UNIT::METER;
    initParams.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    // Use SDK's real-time mode - handles timing internally for smooth playback
    initParams.svo_real_time_mode = (inputMode == InputMode::Live) || synchronizer.isRealTimeMode();
    initParams.async_grab_camera_recovery = false;
    initParams.depth_maximum_distance = 7;

    sl::ERROR_CODE err;
    {
        std::lock_guard<std::mutex> lock(g_deviceOpenMutex);
        err = camera.open(initParams);
    }
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "[ZED " << state.id << "] Failed to open: " << err << std::endl;
        state.running = false;
        return;
    }

    // Get actual serial number for ID
    auto info = camera.getCameraInformation();
    state.id = std::to_string(info.serial_number);

    // Register with async synchronizer if sync mode is enabled
    if (asyncSync) {
        asyncSync->setStreamRole(state.id, Role::Required);
    }

    std::cout << "[ZED " << state.id << "] Opened successfully. Resolution: " << info.camera_configuration.resolution.width << "x"
              << info.camera_configuration.resolution.height << std::endl;

    // Always enable positional tracking for gravity-based orientation correction
    {
        sl::PositionalTrackingParameters trackingParams;
        trackingParams.set_as_static = true;
        trackingParams.set_gravity_as_origin = true;
        auto trackErr = camera.enablePositionalTracking(trackingParams);
        if (trackErr != sl::ERROR_CODE::SUCCESS) {
            std::cerr << "[ZED " << state.id << "] Warning: Failed to enable tracking: " << trackErr << std::endl;
        }
    }

    // Enable object detection
    if (enableOD) {
        sl::ObjectDetectionParameters odParams;
        odParams.detection_model = g_detectionConfig.object_model;
        odParams.enable_tracking = true;
        odParams.enable_segmentation = false;
        auto odErr = camera.enableObjectDetection(odParams);
        if (odErr != sl::ERROR_CODE::SUCCESS) {
            std::cerr << "[ZED " << state.id << "] Failed to enable OD: " << odErr << std::endl;
            enableOD = false;
        } else {
            std::cout << "[ZED " << state.id << "] Object detection enabled" << std::endl;
        }
    }

    // Enable body tracking
    if (enableBT) {
        sl::BodyTrackingParameters btParams;
        btParams.detection_model = g_detectionConfig.body_model;
        btParams.body_format = g_detectionConfig.body_format;
        btParams.enable_tracking = true;
        btParams.enable_body_fitting = true;
        btParams.enable_segmentation = false;
        auto btErr = camera.enableBodyTracking(btParams);
        if (btErr != sl::ERROR_CODE::SUCCESS) {
            std::cerr << "[ZED " << state.id << "] Failed to enable BT: " << btErr << std::endl;
            enableBT = false;
        } else {
            std::cout << "[ZED " << state.id << "] Body tracking enabled" << std::endl;
        }
    }

    // Runtime parameters
    sl::RuntimeParameters runtimeParams;
    runtimeParams.enable_fill_mode = false;
    runtimeParams.confidence_threshold = 51;

    sl::ObjectDetectionRuntimeParameters odRuntimeParams;
    odRuntimeParams.detection_confidence_threshold = g_detectionConfig.object_detection_confidence;

    sl::BodyTrackingRuntimeParameters btRuntimeParams;
    btRuntimeParams.detection_confidence_threshold = g_detectionConfig.body_tracking_confidence;
    btRuntimeParams.skeleton_smoothing = 0.7f;

    // Wait for all devices to be open before starting grab loop
    synchronizer.waitForAllDevicesOpen(state.id);

    // Recording state
    std::string recordingPath;
    bool isRecording = false;

    while (state.running) {
        // Handle recording toggle
        bool wantRecording = state.recording.load();
        if (wantRecording != isRecording) {
            if (wantRecording && !isRecording) {
                recordingPath = generateRecordingFilename("zed_" + state.id, ".svo2");
                sl::RecordingParameters recParams;
                recParams.video_filename = sl::String(recordingPath.c_str());
                recParams.compression_mode = sl::SVO_COMPRESSION_MODE::H265;
                auto recErr = camera.enableRecording(recParams);
                if (recErr == sl::ERROR_CODE::SUCCESS) {
                    isRecording = true;
                    std::cout << "[ZED " << state.id << "] Recording to: " << recordingPath << std::endl;
                } else {
                    std::cerr << "[ZED " << state.id << "] Failed to start recording: " << recErr << std::endl;
                }
            } else if (!wantRecording && isRecording) {
                camera.disableRecording();
                isRecording = false;
                std::cout << "[ZED " << state.id << "] Recording stopped" << std::endl;
            }
        }

        // Grab frame
        auto grabErr = camera.grab(runtimeParams);

        if (grabErr == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            std::cout << "[ZED " << state.id << "] End of SVO file" << std::endl;
            state.endOfFile = true;
            break;
        }
        if (grabErr != sl::ERROR_CODE::SUCCESS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Fill data packet
        ZedDataPacket packet;
        packet.valid = true;
        packet.timestamp = camera.getTimestamp(sl::TIME_REFERENCE::IMAGE);
        packet.sensorPose = state.pose;
        packet.hasSensorPose = state.hasPose;

        // Retrieve image (GPU, native resolution)
        camera.retrieveImage(packet.image, sl::VIEW::LEFT, sl::MEM::GPU);

        // Retrieve point cloud (GPU, native resolution for GPU-accelerated transform)
        camera.retrieveMeasure(packet.pointCloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU);

        // Retrieve pose if tracking enabled
        if (enableTracking) {
            packet.hasTrackingPose = (camera.getPosition(packet.pose) == sl::POSITIONAL_TRACKING_STATE::OK);
        }

        // Retrieve object detection results
        if (enableOD) {
            camera.retrieveObjects(packet.objects, odRuntimeParams);
            packet.hasObjects = true;
        }

        // Retrieve body tracking results
        if (enableBT) {
            camera.retrieveBodies(packet.bodies, btRuntimeParams);
            packet.hasBodies = true;
        }

        // Publish data
        state.frameCount++;
        state.lastTimestampNs = packet.timestamp.getNanoseconds();
        if (asyncSync) {
            asyncSync->ingest(state.id, packet);
        } else {
            state.zedBuffer.getWriteBuffer() = std::move(packet);
            state.zedBuffer.publishWrite();
        }
    }

    // Cleanup
    if (isRecording) {
        camera.disableRecording();
    }
    if (enableBT)
        camera.disableBodyTracking();
    if (enableOD)
        camera.disableObjectDetection();
    camera.disablePositionalTracking();
    camera.close();

    std::cout << "[ZED " << state.id << "] Worker thread exiting. Frames: " << state.frameCount << std::endl;
}

/**
 * @brief LiDAR worker thread
 *
 * Runs grab/retrieve loop in dedicated thread, pushing data to triple buffer.
 */
void lidarWorkerThread(
    DeviceState& state,
    const DeviceConfig& config,
    PlaybackSynchronizer& synchronizer,
    AsyncPacketSynchronizer* asyncSync,
    InputMode inputMode
) {
    std::cout << "[LiDAR " << state.id << "] Worker thread starting..." << std::endl;

    sl::Lidar lidar;

    // Configure initialization parameters
    sl::InitLidarParameters initParams;
    initParams.input = config.input;
    initParams.mode = sl::LIDAR_MODE::AUTO;
    initParams.depth_minimum_distance = 1.0f;
    initParams.depth_maximum_distance = 80.0f;
    initParams.coordinate_units = sl::UNIT::METER;
    initParams.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    // Use SDK's real-time mode - handles timing internally for smooth playback
    initParams.svo_real_time_mode = (inputMode == InputMode::Live) || synchronizer.isRealTimeMode();

    sl::ERROR_CODE err;
    {
        std::lock_guard<std::mutex> lock(g_deviceOpenMutex);
        err = lidar.open(initParams);
    }
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "[LiDAR " << state.id << "] Failed to open: " << err << std::endl;
        state.running = false;
        return;
    }

    auto info = lidar.getLidarInformation();
    state.id = std::string(info.serial_number.c_str());
    std::cout << "[LiDAR " << state.id << "] Opened successfully" << std::endl;

    // Register with async synchronizer if sync mode is enabled
    if (asyncSync) {
        asyncSync->setStreamRole(state.id, Role::Optional);
    }

    // Wait for all devices to be open before starting grab loop
    synchronizer.waitForAllDevicesOpen(state.id);

    // Recording state
    std::string recordingPath;
    bool isRecording = false;

    while (state.running) {
        // Handle recording toggle
        bool wantRecording = state.recording.load();
        if (wantRecording != isRecording) {
            if (wantRecording && !isRecording) {
                recordingPath = generateRecordingFilename("lidar_" + state.id, ".osf");
                sl::RecordingLidarParameters recParams;
                recParams.output_filename = recordingPath;
                auto recErr = lidar.enableRecording(recParams);
                if (recErr == sl::ERROR_CODE::SUCCESS) {
                    isRecording = true;
                    std::cout << "[LiDAR " << state.id << "] Recording to: " << recordingPath << std::endl;
                } else {
                    std::cerr << "[LiDAR " << state.id << "] Failed to start recording: " << recErr << std::endl;
                }
            } else if (!wantRecording && isRecording) {
                lidar.disableRecording();
                isRecording = false;
                std::cout << "[LiDAR " << state.id << "] Recording stopped" << std::endl;
            }
        }

        // Grab frame
        auto grabErr = lidar.grab();

        if (grabErr == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            std::cout << "[LiDAR " << state.id << "] End of OSF file" << std::endl;
            state.endOfFile = true;
            break;
        }
        if (grabErr != sl::ERROR_CODE::SUCCESS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Fill data packet
        LidarDataPacket packet;
        packet.valid = true;
        packet.timestamp = lidar.getTimestamp(sl::TIME_REFERENCE::IMAGE);
        packet.sensorPose = state.pose;
        packet.hasPose = state.hasPose;

        // Retrieve point cloud with reflectivity view (XYZ + packed color in 4th channel)
        lidar.retrieveMeasure(packet.pointCloud, sl::LIDAR_MEASURE::XYZ_REFLECTIVITY_VIEW);

        // Retrieve intensity view image for visualization
        lidar.retrieveView(packet.intensityImage, sl::LIDAR_VIEW::SIGNAL);

        // Publish data
        state.frameCount++;
        state.lastTimestampNs = packet.timestamp.getNanoseconds();
        if (asyncSync) {
            asyncSync->ingest(state.id, packet);
        } else {
            state.lidarBuffer.getWriteBuffer() = std::move(packet);
            state.lidarBuffer.publishWrite();
        }
    }

    // Cleanup
    if (isRecording) {
        lidar.disableRecording();
    }
    lidar.close();

    std::cout << "[LiDAR " << state.id << "] Worker thread exiting. Frames: " << state.frameCount << std::endl;
}

// =============================================================================
// Main Implementation
// =============================================================================

void runImpl(int argc, char** argv, bool headless) {
    std::cout << "\n=== ZED + LiDAR Multi-Sensor Sample ===" << std::endl;

    std::vector<DeviceConfig> devices;
    InputMode inputMode = InputMode::Live;
    bool realTimePlayback = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--realtime" || arg == "-rt") {
            realTimePlayback = true;
            std::cout << "Real-time playback mode enabled" << std::endl;
            continue;
        }
        if (arg == "--baselink" || arg == "-b") {
            g_referenceMode = ReferenceMode::Baselink;
            continue;
        }
        if (arg == "--world" || arg == "-w") {
            g_referenceMode = ReferenceMode::World;
            continue;
        }
        if (arg == "--sensor" || arg == "-s") {
            g_referenceMode = ReferenceMode::Sensor;
            continue;
        }
        if (arg == "--object-detection" || arg == "-od") {
            g_detectionConfig.object_detection = true;
            continue;
        }
        if (arg == "--body-tracking" || arg == "-bt") {
            g_detectionConfig.body_tracking = true;
            continue;
        }

        // Parse input files
        if (arg.size() > 5 && arg.substr(arg.size() - 5) == ".json") {
            if (!parseJsonConfig(arg, devices))
                return;
            inputMode = InputMode::Playback;
        } else if (arg.size() > 4 && (arg.substr(arg.size() - 4) == ".svo" || arg.substr(arg.size() - 5) == ".svo2")) {
            DeviceConfig cfg;
            cfg.type = DeviceType::ZED;
            cfg.input.setFromSVOFile(sl::String(arg.c_str()));
            devices.push_back(cfg);
            inputMode = InputMode::Playback;
            std::cout << "Playing back SVO: " << arg << std::endl;
        } else if (arg.size() > 4 && arg.substr(arg.size() - 4) == ".osf") {
            DeviceConfig cfg;
            cfg.type = DeviceType::LIDAR;
            cfg.input.setFromSVOFile(sl::String(arg.c_str()));
            devices.push_back(cfg);
            inputMode = InputMode::Playback;
            std::cout << "Playing back OSF: " << arg << std::endl;
        }
    }

    // Auto-detect devices if none specified
    if (devices.empty()) {
        std::cout << "Auto-detecting devices..." << std::endl;

        auto zedDevices = sl::Camera::getDeviceList();
        for (const auto& dev : zedDevices) {
            DeviceConfig cfg;
            cfg.type = DeviceType::ZED;
            cfg.input.setFromSerialNumber(dev.serial_number);
            devices.push_back(cfg);
            std::cout << "  Found ZED: " << dev.serial_number << std::endl;
        }

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

    // Create device states
    std::vector<std::unique_ptr<DeviceState>> deviceStates;
    for (size_t i = 0; i < devices.size(); i++) {
        auto state = std::make_unique<DeviceState>();
        state->id = "device_" + std::to_string(i);
        state->type = devices[i].type;
        state->pose = devices[i].pose;
        state->hasPose = devices[i].has_pose;
        deviceStates.push_back(std::move(state));
    }

    // Initialize synchronizer for playback
    PlaybackSynchronizer synchronizer;
    if (inputMode == InputMode::Playback) {
        synchronizer.setRealTimeMode(realTimePlayback);
        synchronizer.enable(devices.size());
    }

    // Initialize async synchronizer if sync mode is enabled
    std::unique_ptr<AsyncPacketSynchronizer> asyncSync;
    if (g_syncMode) {
        // Buffer depth: small for live (frames arrive at real-time rate),
        // larger for playback (SVOs can burst faster than consumption)
        size_t bufferDepth = (inputMode == InputMode::Live) ? 4 : 60;
        asyncSync = std::make_unique<AsyncPacketSynchronizer>(15'000'000, bufferDepth); // tolerance ~half a frame at 30 FPS
        std::cout << "Sync mode enabled: using timestamp-based cross-device synchronization" << std::endl;
    }

    // Start worker threads
    std::vector<std::thread> workerThreads;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i].type == DeviceType::ZED) {
            bool enableOD = g_detectionConfig.object_detection || devices[i].enable_object_detection;
            bool enableBT = g_detectionConfig.body_tracking || devices[i].enable_body_tracking;
            bool enableTracking = (g_referenceMode == ReferenceMode::World) || enableOD || enableBT;

            workerThreads.emplace_back(
                zedWorkerThread,
                std::ref(*deviceStates[i]),
                std::ref(devices[i]),
                std::ref(synchronizer),
                asyncSync.get(),
                inputMode,
                enableOD,
                enableBT,
                enableTracking
            );
        } else {
            workerThreads.emplace_back(
                lidarWorkerThread,
                std::ref(*deviceStates[i]),
                std::ref(devices[i]),
                std::ref(synchronizer),
                asyncSync.get(),
                inputMode
            );
        }
    }

    // Initialize viewer
    std::unique_ptr<GLViewer> viewer;

    if (!headless) {
        viewer = std::make_unique<GLViewer>();
        viewer->setLayout(g_viewerLayout);
        viewer->setRecordingCallback(getRecordingStatus);
        viewer->setRecordingToggleCallback([]() {
            g_recordingState.toggleRequested = true;
        });
        viewer->init(argc, argv);

        if (!viewer->isAvailable()) {
            std::cerr << "Failed to initialize viewer!" << std::endl;
            for (auto& state : deviceStates)
                state->running = false;
            for (auto& t : workerThreads)
                if (t.joinable())
                    t.join();
            return;
        }
    } else {
        std::cout << "Running in headless mode (no display)" << std::endl;
    }

    std::cout << "\nStarting sensor streaming..." << std::endl;
    std::cout << "Controls: R=toggle recording, WASD=move, Mouse=look, Q=quit" << std::endl;

    int frameCount = 0;

    // Main loop - consume data from worker threads and update viewer
    while (!headless ? (viewer->isAvailable() && !viewer->isDone()) : true) {
        // Handle recording toggle
        if (g_recordingState.toggleRequested.exchange(false)) {
            bool newState = !g_recordingState.isRecording;
            g_recordingState.isRecording = newState;
            for (auto& state : deviceStates) {
                state->recording = newState;
            }
            updateRecordingStatus(newState ? "RECORDING" : "Recording stopped");
            std::cout << (newState ? "Recording started" : "Recording stopped") << std::endl;
        }

        // Check if all devices have ended (for playback)
        bool allEnded = true;
        for (const auto& state : deviceStates) {
            if (!state->endOfFile && state->running) {
                allEnded = false;
                break;
            }
        }
        if (allEnded && inputMode == InputMode::Playback) {
            std::cout << "All playback files completed" << std::endl;
            break;
        }

        // Process data from devices
        bool anyNewData = false;

        if (asyncSync) {
            // Sync mode: consume synchronized bundles from AsyncPacketSynchronizer
            auto bundleOpt = asyncSync->syncNext(MissingPolicy::EmitPartial);
            if (bundleOpt) {
                const Bundle& b = *bundleOpt;
                for (const auto& [serial, packetVar] : b.packets) {
                    if (std::holds_alternative<ZedPacketPtr>(packetVar)) {
                        const auto& zptr = std::get<ZedPacketPtr>(packetVar);
                        if (zptr->valid && !headless) {
                            sl::Mat* pcPtr = nullptr;
                            if (zptr->pointCloud.isInit()) {
                                if (zptr->hasSensorPose && g_referenceMode == ReferenceMode::Baselink) {
                                    sl::applyTransform(zptr->pointCloud, sl::Transform(zptr->sensorPose), sl::MEM::GPU);
                                }
                                pcPtr = &zptr->pointCloud;
                            }
                            viewer->updateZed(serial, zptr->image, pcPtr);
                            if (zptr->hasObjects) {
                                viewer->updateZedObjects(serial, zptr->objects);
                            }
                            if (zptr->hasBodies) {
                                viewer->updateZedBodies(serial, zptr->bodies);
                            }
                        }
                    } else if (std::holds_alternative<LidarPacketPtr>(packetVar)) {
                        const auto& lptr = std::get<LidarPacketPtr>(packetVar);
                        if (lptr->valid && !headless) {
                            if (lptr->hasPose && g_referenceMode == ReferenceMode::Baselink) {
                                sl::applyTransform(lptr->pointCloud, sl::Transform(lptr->sensorPose), sl::MEM::GPU);
                            }
                            viewer->updateLidar(serial, lptr->pointCloud, lptr->intensityImage);
                        }
                    }
                }
                anyNewData = true;
            }
        } else {
            // Default mode: consume from per-device triple buffers
            for (auto& state : deviceStates) {
                if (state->type == DeviceType::ZED) {
                    if (state->zedBuffer.trySwapRead()) {
                        ZedDataPacket& packet = state->zedBuffer.getReadBuffer();
                        if (packet.valid && !headless) {
                            sl::Mat* pcPtr = nullptr;
                            if (packet.pointCloud.isInit()) {
                                if (packet.hasSensorPose && g_referenceMode == ReferenceMode::Baselink) {
                                    sl::applyTransform(packet.pointCloud, sl::Transform(packet.sensorPose), sl::MEM::GPU);
                                }
                                pcPtr = &packet.pointCloud;
                            }
                            viewer->updateZed(state->id, packet.image, pcPtr);
                            if (packet.hasObjects) {
                                viewer->updateZedObjects(state->id, packet.objects);
                            }
                            if (packet.hasBodies) {
                                viewer->updateZedBodies(state->id, packet.bodies);
                            }
                        }
                        anyNewData = true;
                    }
                } else {
                    if (state->lidarBuffer.trySwapRead()) {
                        LidarDataPacket& packet = state->lidarBuffer.getReadBuffer();
                        if (packet.valid && !headless) {
                            if (packet.hasPose && g_referenceMode == ReferenceMode::Baselink) {
                                sl::applyTransform(packet.pointCloud, sl::Transform(packet.sensorPose), sl::MEM::GPU);
                            }
                            viewer->updateLidar(state->id, packet.pointCloud, packet.intensityImage);
                        }
                        anyNewData = true;
                    }
                }
            }
        }

        if (anyNewData) {
            frameCount++;
        }

        if (!anyNewData) {
            // Sync mode: 1ms polling (bundles arrive at frame rate)
            // Default mode: 100µs polling (tight triple-buffer reads)
            std::this_thread::sleep_for(asyncSync ? std::chrono::milliseconds(1) : std::chrono::microseconds(100));
        }
    }

    // Shutdown
    std::cout << "\nShutting down worker threads..." << std::endl;
    synchronizer.disable();
    for (auto& state : deviceStates) {
        state->running = false;
    }
    for (auto& t : workerThreads) {
        if (t.joinable())
            t.join();
    }

    std::cout << "\nSample finished. Main loop iterations: " << frameCount << std::endl;
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
    std::cout << "  " << progName << " cam.svo2 lidar.osf -rt             # Real-time playback" << std::endl;
    std::cout << "  " << progName << " cam.svo2 lidar.osf --sync           # Synchronized playback" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --baselink, -b          Use BASELINK reference frame (default, sensor poses applied)" << std::endl;
    std::cout << "  --world, -w             Use WORLD reference frame (enable positional tracking)" << std::endl;
    std::cout << "  --sensor, -s            Use SENSOR reference frame (raw data, no transform)" << std::endl;
    std::cout << "  --realtime, -rt         Play back at recorded frame rate" << std::endl;
    std::cout << "  --object-detection, -od Enable object detection on ZED cameras" << std::endl;
    std::cout << "  --body-tracking, -bt    Enable body tracking on ZED cameras" << std::endl;
    std::cout << "  --headless              Run without display (data processing only)" << std::endl;
    std::cout << "  --sync                  Enable timestamp-based cross-device synchronization" << std::endl;
    std::cout << "  --combined-layout, -cl  Use combined viewer layout (all sensors in one 3D view)" << std::endl;
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
    bool headless = false;

    // Process command-line options
    std::vector<char*> filteredArgs;
    filteredArgs.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--headless") {
            headless = true;
        } else if (arg == "--sync") {
            g_syncMode = true;
        } else if (arg == "--combined-layout" || arg == "-cl") {
            g_viewerLayout = ViewerLayout::Combined;
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
        runImpl(newArgc, newArgv, headless);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
