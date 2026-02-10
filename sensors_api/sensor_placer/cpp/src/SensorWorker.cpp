#include "SensorWorker.h"
#include <iostream>

SensorWorker::SensorWorker() {
    zedInit_.depth_mode = sl::DEPTH_MODE::NEURAL;
    zedInit_.camera_fps = 15;
    zedInit_.coordinate_system = COORD_SYS;
    zedInit_.coordinate_units = UNIT_REF;
    zedInit_.open_timeout_sec = 10;
}

SensorWorker::~SensorWorker() {
    close();
}

void SensorWorker::close() {
    stop();
    pointCloud_.free();
    if (type_ == SensorType::ZED)
        zed_.close();
    else {
        lidarPC_.free();
        lidar_.close();
    }
}

// ─── ZED openers ─────────────────────────────────────────────────────────────

bool SensorWorker::openZedSerial(int serial) {
    type_ = SensorType::ZED;
    zedInit_.input.setFromSerialNumber(serial);
    return initZed();
}

bool SensorWorker::openZedSVO(const std::string& path) {
    type_ = SensorType::ZED;
    zedInit_.input.setFromSVOFile(path.c_str());
    return initZed();
}

bool SensorWorker::openZedStream(const std::string& ip) {
    type_ = SensorType::ZED;
    zedInit_.input.setFromStream(ip.c_str());
    return initZed();
}

bool SensorWorker::openZedInput(sl::InputType input) {
    type_ = SensorType::ZED;
    zedInit_.input = input;
    return initZed();
}

bool SensorWorker::initZed() {
    auto err = zed_.open(zedInit_);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "ZED open error: " << err << std::endl;
        return false;
    }
    pointCloud_.alloc(pcRes_, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);
    camInfo_ = zed_.getCameraInformation(pcRes_);
    serial_ = camInfo_.serial_number;
    std::cout << "ZED opened: S/N " << serial_ << std::endl;
    zedRtp_.confidence_threshold = 50;
    start();
    return true;
}

// ─── LiDAR openers ──────────────────────────────────────────────────────────

bool SensorWorker::openLidarIP(const std::string& ip) {
    type_ = SensorType::LIDAR;
    ip_ = ip;
    lidarInit_.input.setFromStream(sl::String(ip.c_str()));
    return initLidar();
}

bool SensorWorker::openLidarOSF(const std::string& path) {
    type_ = SensorType::LIDAR;
    lidarInit_.input.setFromSVOFile(sl::String(path.c_str()));
    return initLidar();
}

bool SensorWorker::initLidar() {
    lidarInit_.mode = sl::LIDAR_MODE::AUTO;
    lidarInit_.depth_minimum_distance = 1.0f;
    lidarInit_.depth_maximum_distance = 80.0f;
    lidarInit_.coordinate_units = UNIT_REF;
    lidarInit_.coordinate_system = COORD_SYS;

    auto err = lidar_.open(lidarInit_);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "LiDAR open error: " << err << std::endl;
        return false;
    }
    auto info = lidar_.getLidarInformation();
    serial_ = 0; // LiDARs use IP, not serial
    // ip_ is already set by openLidarIP(); LidarInformation doesn't carry it
    std::cout << "LiDAR opened: " << info.serial_number << " (" << ip_ << ")" << std::endl;
    start();
    return true;
}

// ─── Thread management ──────────────────────────────────────────────────────

void SensorWorker::start() {
    if (!running_) {
        running_ = true;
        if (type_ == SensorType::ZED)
            worker_ = std::thread(&SensorWorker::runZed, this);
        else
            worker_ = std::thread(&SensorWorker::runLidar, this);
    }
}

void SensorWorker::stop() {
    if (running_) {
        running_ = false;
        if (worker_.joinable())
            worker_.join();
    }
}

void SensorWorker::runZed() {
    while (running_) {
        auto err = zed_.grab(zedRtp_);
        if (err == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            zed_.setSVOPosition(0);
            continue;
        }
        if (err == sl::ERROR_CODE::SUCCESS) {
            if (askFloor_) {
                sl::Plane floor;
                foundFloor_ = (zed_.findFloorPlane(floor, floorReset_) == sl::ERROR_CODE::SUCCESS);
                askFloor_ = false;
            }
            if (waitIMU_) {
                sl::SensorsData sd;
                zed_.getSensorsData(sd, sl::TIME_REFERENCE::IMAGE);
                imuPose_ = sd.imu.pose;
                waitIMU_ = false;
            }
            zed_.retrieveMeasure(pointCloud_, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pcRes_);
            updated_ = true;
        }
    }
}

void SensorWorker::runLidar() {
    while (running_) {
        auto err = lidar_.grab();
        if (err == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            lidar_.setSVOPosition(0);
            continue;
        }
        if (err == sl::ERROR_CODE::SUCCESS) {
            std::lock_guard<std::mutex> lk(lidarMtx_);
            lidar_.retrieveMeasure(lidarPC_, sl::LIDAR_MEASURE::XYZ_REFLECTIVITY_VIEW, sl::Resolution(0, 0), sl::MEM::GPU);
            updated_ = true;
        }
    }
}

// ─── Accessors ───────────────────────────────────────────────────────────────

bool SensorWorker::needsUpdate() {
    bool tmp = updated_;
    updated_ = false;
    return tmp;
}

sl::Mat SensorWorker::getLidarPC() {
    std::lock_guard<std::mutex> lk(lidarMtx_);
    return lidarPC_;
}

CUstream SensorWorker::getCUDAStream() {
    if (type_ == SensorType::ZED)
        return zed_.getCUDAStream();
    return nullptr;
}

sl::CameraParameters SensorWorker::getCamParams() {
    return camInfo_.camera_configuration.calibration_parameters.left_cam;
}

sl::Transform SensorWorker::getOrientation() {
    int count = 0;
    while (waitIMU_) {
        sl::sleep_ms(2);
        if (count++ > 2000)
            break;
    }
    return imuPose_;
}

bool SensorWorker::getIMUPose(sl::Transform& pose) {
    if (type_ != SensorType::ZED)
        return false;
    // Request a fresh IMU sample
    waitIMU_ = true;
    int count = 0;
    while (waitIMU_) {
        sl::sleep_ms(2);
        if (count++ > 2000)
            return false;
    }
    // imuPose_ contains the IMU-derived orientation (gravity-aligned)
    sl::float3 imu_angles = imuPose_.getEulerAngles();
    sl::float3 current_angles = pose.getEulerAngles();
    sl::Translation current_translation = pose.getTranslation();

    // Override pitch (X) and roll (Z) from IMU, keep current yaw (Y)
    sl::float3 new_angles(imu_angles.x, current_angles.y, imu_angles.z);
    sl::Rotation new_rotation;
    new_rotation.setEulerAngles(new_angles);

    pose.setOrientation(new_rotation);
    pose.setTranslation(current_translation);
    return true;
}

bool SensorWorker::findFloorPlane(sl::Transform& pose) {
    if (type_ != SensorType::ZED)
        return false;
    askFloor_ = true;
    while (askFloor_)
        sl::sleep_us(500);
    if (foundFloor_) {
        // Extract Euler angles from the floor reset transform (pitch and roll)
        sl::float3 floor_angles = floorReset_.getEulerAngles();

        // Extract Euler angles from the current pose (to preserve yaw)
        sl::float3 current_angles = pose.getEulerAngles();

        // Extract current translation to preserve it
        sl::Translation current_translation = pose.getTranslation();

        // Create new transform with floor pitch/roll and current yaw (Y axis)
        sl::float3 new_angles(floor_angles.x, current_angles.y, floor_angles.z);
        sl::Rotation new_rotation;
        new_rotation.setEulerAngles(new_angles);

        // Apply the new rotation and keep original translation
        pose.setOrientation(new_rotation);
        pose.setTranslation(current_translation);
    }
    return foundFloor_;
}

void SensorWorker::setTextureOnly(bool state) {
    zedRtp_.texture_confidence_threshold = state ? 75 : 100;
    zedRtp_.confidence_threshold = state ? 95 : 50; // for edge only set the confidence higher
}
