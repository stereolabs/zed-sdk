///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2026, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

//
// Multi camera positional tracking: the pose of a rig of ZED cameras mounted on one rigid body, frame after frame.
//
// The Sensors API tracks the cameras as that one body: they feed a single positional tracking and build a single shared map,
// and the pose of each camera is the pose of the rig moved by the pose of the camera on the rig.
//
// The rig mixes ZED stereo cameras and ZED One cameras (ZED X One). A ZED stereo camera drives the tracking, so the rig needs
// at least one; each ZED One adds its images to the map.
//
// The ZED API flow, in main() and the functions right below it:
//
//   init() -> add() / setSensorPose() for each camera -> enablePositionalTracking()
//          -> process() / getPosition() -> close()
//
// The helper files next to this one hold the rest:
//
//   RigConfig           the rig file: the cameras, their inputs and their poses on the rig
//   SampleArguments     command line
//   SampleConsole       console output
//   MapIO               area files, on disk
//   TrajectoryRecorder  trajectory (TUM) and per-frame tracking status (CSV) output
//   LandmarkOverlay     drawing the tracked landmarks on the image
//   SvoPlaybackClock    replaying the SVOs at the speed they were recorded at
//   RigView             the OpenGL viewer
//

// ZED includes
#include <sl/Sensors.hpp>

// Sample includes
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "LandmarkOverlay.hpp"
#include "MapIO.hpp"
#include "RigConfig.hpp"
#include "RigView.hpp"
#include "SampleArguments.hpp"
#include "SampleConsole.hpp"
#include "SvoPlaybackClock.hpp"
#include "TrajectoryRecorder.hpp"

// Using the sl namespace
using namespace sl;

/// Coordinate system of every pose of the sample.
constexpr COORDINATE_SYSTEM SAMPLE_COORDINATE_SYSTEM = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
/// Unit of every pose of the sample.
constexpr UNIT SAMPLE_UNIT = UNIT::METER;

/**
 * @brief What the tracking gives for one frame.
 */
struct RigFrame {
    /// Pose of each camera in the world frame, in rig order.
    std::vector<Pose> cameraPoses;
    /// Tracking state of each camera, in rig order.
    std::vector<POSITIONAL_TRACKING_STATE> cameraStates;
    /// Pose of the rig (BASELINK) in the world frame.
    Pose rigPose;
    /// Status of the tracking of the rig.
    PositionalTrackingStatus trackingStatus;
};

/**
 * @brief The camera image shown in the viewer, and what retrieving it requires.
 */
struct CameraImageDisplay {
    /// The image, filled for the camera the viewer shows.
    Mat image;
    /// Size the image is retrieved at.
    Resolution displayResolution;
    /// Size of the camera images, which 2D landmarks are expressed in.
    Resolution cameraResolution;
    /// Whether each camera has a fisheye lens, in rig order.
    std::vector<bool> isFisheye;
};

InitSensorsParameters buildInitSensorsParameters();
InitParameters buildInitParameters(const RigConfig& rig, const RigCamera& camera);
Transform buildSensorPose(const RigCamera& camera);
PositionalTrackingSensorsParameters buildTrackingParameters(const Arguments& args, SensorDeviceIdentifier trackingCamera);
std::optional<std::vector<SensorDeviceIdentifier>> openRig(Sensors& sensors, const RigConfig& rig);
std::optional<size_t> selectTrackingCamera(const RigConfig& rig, const std::vector<SensorDeviceIdentifier>& cameraIds);
bool enableRigTracking(
    Sensors& sensors,
    const PositionalTrackingSensorsParameters& trackingParameters,
    const RigConfig& rig,
    const std::vector<SensorDeviceIdentifier>& cameraIds
);
void trackRig(
    int argc,
    char** argv,
    Sensors& sensors,
    const Arguments& args,
    const RigConfig& rig,
    const std::vector<SensorDeviceIdentifier>& cameraIds,
    size_t trackingCamera
);
void exportKeyframeGroups(Sensors& sensors, const std::string& path);

/**
 * @brief Tracks the rig described by the rig file given on the command line.
 * @param argc Number of command line arguments.
 * @param argv The command line arguments.
 * @return 0 on success, 1 on an invalid command line, rig file or camera, or when the tracking cannot start.
 */
int main(int argc, char** argv) {
    printHeader("ZED Positional Tracking - Multi Camera");

    if (argc > 1 && std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return 0;
    }

    Arguments args;
    if (!parseArgs(argc, argv, args)) {
        std::cout << "\n";
        printUsage(argv[0]);
        return 1;
    }

    printArgs(args);

    RigConfig rig;
    if (!loadRigConfig(args.rigFile, rig)) {
        return 1;
    }

    Sensors sensors;
    const std::optional<std::vector<SensorDeviceIdentifier>> cameraIds = openRig(sensors, rig);
    if (!cameraIds) {
        sensors.close();
        return 1;
    }

    const std::optional<size_t> selectedTrackingCamera = selectTrackingCamera(rig, *cameraIds);
    if (!selectedTrackingCamera) {
        sensors.close();
        return 1;
    }
    const size_t trackingCamera = selectedTrackingCamera.value();
    printRig(sensors, rig, *cameraIds, trackingCamera);

    const PositionalTrackingSensorsParameters trackingParameters = buildTrackingParameters(args, (*cameraIds)[trackingCamera]);
    if (!enableRigTracking(sensors, trackingParameters, rig, *cameraIds)) {
        sensors.close();
        return 1;
    }

    printTrackingParameters(trackingParameters);
    if (!args.noGui) {
        printInteraction();
    }

    trackRig(argc, argv, sensors, args, rig, *cameraIds, trackingCamera);

    if (args.keyframeTUMFile) {
        exportKeyframeGroups(sensors, args.keyframeTUMFile.value());
    }

    if (args.map) {
        saveAreaMap(sensors, (*cameraIds)[trackingCamera], args.outputAreaFile.value());
    }

    sensors.close();

    return 0;
}

/**
 * @brief Builds the parameters of the Sensors instance.
 * @return The parameters: every pose in SAMPLE_COORDINATE_SYSTEM and SAMPLE_UNIT, whatever the parameters of each camera say.
 */
InitSensorsParameters buildInitSensorsParameters() {
    InitSensorsParameters initSensorsParameters;
    initSensorsParameters.coordinate_units = SAMPLE_UNIT;
    initSensorsParameters.coordinate_system = SAMPLE_COORDINATE_SYSTEM;
    initSensorsParameters.verbose = 1;
    return initSensorsParameters;
}

/**
 * @brief Builds the parameters of one camera of the rig. The tracking runs on the images alone, so no depth is computed.
 *
 * The same parameters serve a ZED One: Sensors::add() opens it as a ZED One, with the settings that apply to it.
 * @param rig The rig, for the capture settings shared by its cameras.
 * @param camera The camera.
 * @return The parameters of the camera.
 */
InitParameters buildInitParameters(const RigConfig& rig, const RigCamera& camera) {
    InitParameters initParameters;

    initParameters.input = camera.input;
    initParameters.depth_mode = DEPTH_MODE::NONE;
    initParameters.sdk_verbose = 1;

    if (rig.resolution) {
        initParameters.camera_resolution = rig.resolution.value();
    }
    if (rig.fps > 0) {
        initParameters.camera_fps = rig.fps;
    }

    // Every frame of a recording is processed, SvoPlaybackClock paces them.
    initParameters.svo_real_time_mode = false;

    return initParameters;
}

/**
 * @brief Converts the pose of a camera on the rig from the rig file convention to the one of the Sensors API.
 * @param camera The camera, whose pose is in meters on the camera axes (X right, Y down, Z forward).
 * @return The pose in SAMPLE_COORDINATE_SYSTEM.
 */
Transform buildSensorPose(const RigCamera& camera) {
    Transform sensorPose = camera.cameraToRig;
    convertCoordinateSystem(sensorPose, COORDINATE_SYSTEM::IMAGE, SAMPLE_COORDINATE_SYSTEM);
    return sensorPose;
}

/**
 * @brief Builds the tracking parameters: the cameras tracked together as one rigid body, with one shared map.
 * @param args The options, for the area file to load.
 * @param trackingCamera Camera that drives the tracking, a ZED stereo camera: its IMU is the one used.
 * @return The tracking parameters.
 */
PositionalTrackingSensorsParameters buildTrackingParameters(const Arguments& args, SensorDeviceIdentifier trackingCamera) {
    PositionalTrackingSensorsParameters trackingParameters;

    trackingParameters.multi_sensor_mode = MULTI_SENSOR_TRACKING_MODE::SINGLE_BODY;
    trackingParameters.tracking_camera_id = trackingCamera;
    trackingParameters.mode = POSITIONAL_TRACKING_MODE::GEN_3;

    // An area file saved by the same rig: poses are given in the frame of that area once the rig recognizes it.
    if (args.inputAreaFile) {
        trackingParameters.area_file_path = args.inputAreaFile.value().c_str();
    }

    trackingParameters.set_floor_as_origin = false;
    trackingParameters.set_gravity_as_origin = true;
    trackingParameters.enable_area_memory = true;
    trackingParameters.enable_pose_smoothing = false;
    trackingParameters.enable_imu_fusion = true;
    trackingParameters.set_as_static = false;
    trackingParameters.depth_min_range = -1;

    return trackingParameters;
}

/**
 * @brief Opens every camera of the rig, sets its pose on the rig, and aligns the recordings in time.
 * @param sensors The Sensors instance, not yet initialized.
 * @param rig The rig.
 * @return The identifier of each camera, in rig order, or nothing, after printing why, when a camera cannot be opened.
 */
std::optional<std::vector<SensorDeviceIdentifier>> openRig(Sensors& sensors, const RigConfig& rig) {
    assert(!rig.cameras.empty());

    SENSORS_ERROR_CODE status = sensors.init(buildInitSensorsParameters());
    if (status > SENSORS_ERROR_CODE::SUCCESS) {
        print("Failed to initialize the Sensors API", status);
        return std::nullopt;
    }

    std::vector<SensorDeviceIdentifier> cameraIds;
    for (const RigCamera& camera : rig.cameras) {
        SensorDeviceIdentifier cameraId;
        status = sensors.add(buildInitParameters(rig, camera), cameraId);
        if (status > SENSORS_ERROR_CODE::SUCCESS) {
            print("Failed to open camera '" + camera.label + "' (" + camera.inputDescription + ")", status);
            return std::nullopt;
        }

        // Must be set before the tracking is enabled.
        sensors.setSensorPose(buildSensorPose(camera), cameraId);
        cameraIds.push_back(cameraId);
    }

    // Starts every recording at the first instant they all cover.
    if (rig.allFromSVO()) {
        status = sensors.syncSVO();
        if (status > SENSORS_ERROR_CODE::SUCCESS) {
            print("Failed to align the recordings in time", status);
            return std::nullopt;
        }
    }

    return cameraIds;
}

/**
 * @brief Selects the camera that drives the tracking: the one the rig file names, or the first ZED stereo camera.
 * @param rig The rig.
 * @param cameraIds Identifier of each camera, in rig order, which tells a ZED stereo camera from a ZED One.
 * @return Its index in rig order, or nothing, after printing why, when it is a ZED One or the rig has no ZED stereo camera.
 */
std::optional<size_t> selectTrackingCamera(const RigConfig& rig, const std::vector<SensorDeviceIdentifier>& cameraIds) {
    assert(cameraIds.size() == rig.cameras.size());

    const auto isStereo = [&cameraIds](size_t index) {
        return cameraIds[index].type == SENSORS_TYPE::CAMERA;
    };

    const std::optional<size_t> named = rig.trackingCameraIndex();
    if (named) {
        if (!isStereo(named.value())) {
            print(
                "tracking_camera '" + rig.trackingCamera + "' is a ZED One: the tracking is driven by a ZED stereo camera",
                ERROR_CODE::FAILURE,
                false
            );
            return std::nullopt;
        }
        return named;
    }

    for (size_t i = 0; i < cameraIds.size(); i++) {
        if (isStereo(i)) {
            return i;
        }
    }
    print("The rig needs at least one ZED stereo camera to drive the tracking", ERROR_CODE::FAILURE, false);
    return std::nullopt;
}

/**
 * @brief Enables the tracking of the rig and starts the exports of its map.
 * @param sensors The Sensors instance the cameras are opened in.
 * @param trackingParameters The tracking parameters.
 * @param rig The rig, for the camera labels.
 * @param cameraIds Identifier of each camera, in rig order.
 * @return False, after printing why, when the tracking cannot start.
 */
bool enableRigTracking(
    Sensors& sensors,
    const PositionalTrackingSensorsParameters& trackingParameters,
    const RigConfig& rig,
    const std::vector<SensorDeviceIdentifier>& cameraIds
) {
    const SENSORS_ERROR_CODE status = sensors.enablePositionalTracking(trackingParameters);
    if (status > SENSORS_ERROR_CODE::SUCCESS) {
        print("Failed to enable positional tracking", status);
        printCameraErrors(sensors, rig, cameraIds, false);
        return false;
    }

    // The keyframe groups and the 2D landmarks are exported from the frame that follows the first request.
    std::map<uint64_t, KeyFrameGroup> keyframeGroups;
    BatchedData<std::vector<Landmark2D>> landmarks2D;
    sensors.getPositionalTrackingKeyframeGroups(keyframeGroups);
    sensors.getPositionalTrackingLandmarks2D(landmarks2D);
    return true;
}

/**
 * @brief Grabs one frame from every camera and runs the tracking of the rig on them.
 * @param sensors The Sensors instance tracking the rig.
 * @param rig The rig, for the camera labels.
 * @param cameraIds Identifier of each camera, in rig order.
 * @return False at the end of the recordings, or, after printing why, when a camera fails.
 */
bool processFrame(Sensors& sensors, const RigConfig& rig, const std::vector<SensorDeviceIdentifier>& cameraIds) {
    const SENSORS_ERROR_CODE status = sensors.process();

    if (status == SENSORS_ERROR_CODE::END_OF_SVOFILE_REACHED) {
        return false;
    }
    if (status > SENSORS_ERROR_CODE::SUCCESS) {
        print("Failed to process the cameras", status);
        printCameraErrors(sensors, rig, cameraIds, true);
        return false;
    }
    return true;
}

/**
 * @brief Retrieves the poses of the last processed frame.
 * @param sensors The Sensors instance tracking the rig.
 * @param cameraIds Identifier of each camera, in rig order.
 * @param trackingCamera Index of the camera that drives the tracking.
 * @param rigToTrackingCamera Inverse of the pose of the tracking camera on the rig.
 * @return The pose of every camera and of the rig, and the tracking status.
 */
RigFrame retrieveRigFrame(
    Sensors& sensors,
    const std::vector<SensorDeviceIdentifier>& cameraIds,
    size_t trackingCamera,
    const Transform& rigToTrackingCamera
) {
    assert(trackingCamera < cameraIds.size());

    RigFrame frame;
    frame.cameraPoses.resize(cameraIds.size());
    for (size_t i = 0; i < cameraIds.size(); i++) {
        frame.cameraStates.push_back(sensors.getPosition(frame.cameraPoses[i], SENSORS_REFERENCE_FRAME::WORLD, cameraIds[i]));
    }

    frame.rigPose = frame.cameraPoses[trackingCamera];
    frame.rigPose.pose_data = frame.cameraPoses[trackingCamera].pose_data * rigToTrackingCamera;
    frame.trackingStatus = sensors.getPositionalTrackingStatus(cameraIds[trackingCamera]);
    return frame;
}

/**
 * @brief Prepares the camera image of the viewer.
 * @param sensors The Sensors instance the cameras are opened in.
 * @param cameraIds Identifier of each camera, in rig order. The cameras of a rig share one resolution.
 * @return The image settings, the image itself empty.
 */
CameraImageDisplay prepareCameraImageDisplay(Sensors& sensors, const std::vector<SensorDeviceIdentifier>& cameraIds) {
    assert(!cameraIds.empty());

    CameraImageDisplay display;
    for (const SensorDeviceIdentifier& cameraId : cameraIds) {
        const CameraInformation information = sensors.getCameraInformation(Resolution(0, 0), cameraId);
        display.cameraResolution = information.camera_configuration.resolution;
        display.isFisheye.push_back(
            information.camera_configuration.calibration_parameters_raw.left_cam.lens_distortion_model == LENS_DISTORTION_MODEL::FISHEYE
        );
    }

    const int displayWidth = std::min(1280, static_cast<int>(display.cameraResolution.width));
    display.displayResolution = Resolution(displayWidth, display.cameraResolution.height * displayWidth / display.cameraResolution.width);
    return display;
}

/**
 * @brief Gives the viewer the poses and the tracking status of one frame.
 * @param view The viewer.
 * @param frame The frame.
 */
void updateView(RigView& view, const RigFrame& frame) {
    for (size_t i = 0; i < frame.cameraPoses.size(); i++) {
        view.updateCameraPose(i, frame.cameraPoses[i].pose_data, frame.cameraStates[i]);
    }
    view.updateRigPose(frame.rigPose.pose_data);
    view.updatePositionalTrackingStatus(frame.trackingStatus);
}

/**
 * @brief Gives the viewer the current map of the rig: its landmarks and the keyframes of each camera.
 * @param view The viewer.
 * @param sensors The Sensors instance tracking the rig.
 * @param cameraIds Identifier of each camera, in rig order.
 */
void refreshMap(RigView& view, Sensors& sensors, const std::vector<SensorDeviceIdentifier>& cameraIds) {
    std::map<uint64_t, Landmark> landmarks;
    sensors.getPositionalTrackingLandmarks(landmarks);
    view.updateLandmarks(landmarks);

    BatchedData<std::map<uint64_t, KeyFrame>> keyframes;
    sensors.getPositionalTrackingKeyframes(keyframes);

    std::vector<std::map<uint64_t, KeyFrame>> keyframesPerCamera(cameraIds.size());
    for (size_t i = 0; i < cameraIds.size(); i++) {
        auto cameraKeyframes = keyframes.find(cameraIds[i]);
        if (cameraKeyframes != keyframes.end()) {
            keyframesPerCamera[i] = cameraKeyframes->second;
        }
    }
    view.updateKeyframes(keyframesPerCamera);
}

/**
 * @brief Retrieves the image of the camera the viewer shows, with the landmarks it tracks when the viewer displays them.
 * @param view The viewer.
 * @param sensors The Sensors instance tracking the rig.
 * @param cameraIds Identifier of each camera, in rig order.
 * @param[in,out] display The camera image display, whose image is filled.
 */
void showCameraImage(RigView& view, Sensors& sensors, const std::vector<SensorDeviceIdentifier>& cameraIds, CameraImageDisplay& display) {
    const size_t shown = view.displayedCamera();
    assert(shown < cameraIds.size());

    const VIEW imageView = display.isFisheye[shown] ? VIEW::LEFT_UNRECTIFIED : VIEW::LEFT;
    sensors.retrieveImage(display.image, imageView, MEM::CPU, display.displayResolution, cameraIds[shown]);

    if (view.isLandmarkModeEnabled()) {
        BatchedData<std::vector<Landmark2D>> landmarks2D;
        sensors.getPositionalTrackingLandmarks2D(landmarks2D);
        auto shownLandmarks = landmarks2D.find(cameraIds[shown]);
        if (shownLandmarks != landmarks2D.end()) {
            drawLandmarks2D(display.image, shownLandmarks->second, display.cameraResolution);
        }
    }
}

/**
 * @brief Tracks the rig frame after frame, recording and displaying it, until the window is closed or the recordings end.
 * @param argc Number of command line arguments, for the viewer.
 * @param argv The command line arguments, for the viewer.
 * @param sensors The Sensors instance tracking the rig.
 * @param args The options: viewer and output files.
 * @param rig The rig.
 * @param cameraIds Identifier of each camera, in rig order.
 * @param trackingCamera Index of the camera that drives the tracking.
 */
void trackRig(
    int argc,
    char** argv,
    Sensors& sensors,
    const Arguments& args,
    const RigConfig& rig,
    const std::vector<SensorDeviceIdentifier>& cameraIds,
    size_t trackingCamera
) {
    assert(cameraIds.size() == rig.cameras.size());
    assert(trackingCamera < cameraIds.size());

    const Transform rigToTrackingCamera = Transform::inverse(buildSensorPose(rig.cameras[trackingCamera]));

    CameraImageDisplay display = prepareCameraImageDisplay(sensors, cameraIds);
    std::unique_ptr<RigView> view;
    if (!args.noGui) {
        view = std::make_unique<RigView>(argc, argv, &display.image, rig.labels(), trackingCamera);
        if (rig.allFromSVO()) {
            view->enablePauseButton();
        }
    }

    RigRecorder recorder(args.trajectoryTUMFile.value_or(""), args.statusLogFile.value_or(""), rig.labels());
    SvoPlaybackClock svoPlayback;
    uint64_t lastMapRefresh = 0;

    while (!view || view->isOpen()) {
        if (view && view->isPaused()) {
            view->render();
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            continue;
        }

        if (!processFrame(sensors, rig, cameraIds)) {
            break;
        }

        const RigFrame frame = retrieveRigFrame(sensors, cameraIds, trackingCamera, rigToTrackingCamera);
        recorder.record(frame.rigPose, frame.trackingStatus, frame.cameraPoses);

        const Timestamp imageTimestamp = frame.rigPose.timestamp;
        if (rig.allFromSVO()) {
            svoPlayback.waitForRealTime(imageTimestamp);
            svoPlayback.markProcessed(imageTimestamp);
        }

        if (!view) {
            continue;
        }

        updateView(*view, frame);
        // The map holds many points and changes slowly: refreshed once per second.
        if (imageTimestamp.getSeconds() - lastMapRefresh > 1) {
            lastMapRefresh = imageTimestamp.getSeconds();
            refreshMap(*view, sensors, cameraIds);
        }
        showCameraImage(*view, sensors, cameraIds, display);
        view->render();
    }

    recorder.close();

    // Before sensors.close(): the OpenGL resources must go before the CUDA context of the SDK.
    if (view) {
        view->close();
    }
}

/**
 * @brief Writes the pose of the rig at every keyframe group of the map, as left by the last loop closure.
 * @param sensors The Sensors instance tracking the rig.
 * @param path Path of the TUM file.
 */
void exportKeyframeGroups(Sensors& sensors, const std::string& path) {
    std::map<uint64_t, KeyFrameGroup> keyframeGroups;
    const SENSORS_ERROR_CODE status = sensors.getPositionalTrackingKeyframeGroups(keyframeGroups);

    if (status > SENSORS_ERROR_CODE::SUCCESS) {
        print("Failed to retrieve the keyframe groups, " + path + " was not written", status);
    } else if (!writeKeyframeGroupFile(path, keyframeGroups)) {
        print("Failed to write the keyframe groups to " + path, ERROR_CODE::FAILURE, false);
    } else {
        print("Exported " + std::to_string(keyframeGroups.size()) + " keyframe groups to " + path);
    }
}
