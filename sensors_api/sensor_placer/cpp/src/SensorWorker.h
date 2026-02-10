#ifndef SENSOR_WORKER_H
#define SENSOR_WORKER_H

#include <sl/Camera.hpp>
#include <sl/Lidar.hpp>

#include <thread>
#include <atomic>
#include <mutex>
#include <string>

enum class SensorType {
    ZED,
    LIDAR
};

class SensorWorker {
public:
    static constexpr sl::COORDINATE_SYSTEM COORD_SYS = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    static constexpr sl::UNIT UNIT_REF = sl::UNIT::METER;

    SensorWorker();
    ~SensorWorker();

    // ── Open from various sources ──
    bool openZedSerial(int serial);
    bool openZedSVO(const std::string& path);
    bool openZedStream(const std::string& ip);
    bool openZedInput(sl::InputType input);

    bool openLidarIP(const std::string& ip);
    bool openLidarOSF(const std::string& path);

    void close();

    // ── Accessors ──
    SensorType type() const {
        return type_;
    }
    bool isOpen() const {
        return running_;
    }
    bool needsUpdate();

    int serial() const {
        return serial_;
    }
    std::string ip() const {
        return ip_;
    }

    sl::Mat getPC() {
        return pointCloud_;
    }
    sl::Mat getLidarPC();

    CUstream getCUDAStream();
    sl::CameraParameters getCamParams();
    sl::InitParameters getInitParams() {
        return zedInit_;
    }

    // IMU orientation
    sl::Transform getOrientation();
    bool getIMUPose(sl::Transform& pose);

    // Floor plane
    bool findFloorPlane(sl::Transform& pose);

    // Edge-only rendering
    void setTextureOnly(bool state);

private:
    bool initZed();
    bool initLidar();
    void start();
    void stop();
    void runZed();
    void runLidar();

    SensorType type_ = SensorType::ZED;

    // ZED
    sl::Camera zed_;
    sl::InitParameters zedInit_;
    sl::RuntimeParameters zedRtp_;
    sl::CameraInformation camInfo_;

    // LiDAR
    sl::Lidar lidar_;
    sl::InitLidarParameters lidarInit_;

    // Shared
    int serial_ = 0;
    std::string ip_;
    sl::Mat pointCloud_; // GPU, F32_C4
    sl::Mat lidarPC_;    // GPU, for LiDAR grab results
    sl::Resolution pcRes_ {720, 404};

    std::thread worker_;
    std::atomic<bool> running_ {false};
    std::atomic<bool> updated_ {false};

    // IMU
    std::atomic<bool> waitIMU_ {true};
    sl::Transform imuPose_;

    // Floor plane
    std::atomic<bool> askFloor_ {false};
    std::atomic<bool> foundFloor_ {false};
    sl::Transform floorReset_;

    std::mutex lidarMtx_;
};

#endif // SENSOR_WORKER_H
