///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2025, STEREOLABS.
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
// Positional tracking: where the camera is, in a world frame, image after image.
//
// The whole ZED API flow lives in main() below:
//
//   open() -> enablePositionalTracking() -> grab() / getPosition() -> close()
//
// Everything else the sample does (parsing the command line, printing, drawing, saving maps and
// trajectories) is in the helper files next to this one, so that flow stays readable:
//
//   SampleArguments   command line
//   SampleConsole     console output
//   MotionPriorFile   reading and replaying the motion measured by an external source
//   MotionPriorChannel  the same motion, recorded in the SVO next to the images
//   MapIO             area files, on disk
//   TrajectoryRecorder  trajectory (TUM) and per-image tracking status (CSV) output
//   LandmarkOverlay   drawing the tracked landmarks on the image
//   SvoPlaybackClock  replaying an SVO at the speed it was recorded at
//   SLAMView          the OpenGL viewer
//

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "LandmarkOverlay.hpp"
#include "MapIO.hpp"
#include "MotionPriorChannel.hpp"
#include "MotionPriorFile.hpp"
#include "SLAMView.hpp"
#include "SampleArguments.hpp"
#include "SampleConsole.hpp"
#include "SvoPlaybackClock.hpp"
#include "TrajectoryRecorder.hpp"

// Using the sl namespace
using namespace sl;

// Defined right after main(): the camera and tracking configuration the options translate into.
InitParameters buildInitParameters(const Arguments& args);
PositionalTrackingParameters buildTrackingParameters(const Arguments& args);

//
// Main
//
int main(int argc, char** argv) {
    printHeader("ZED Positional Tracking");

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

    //
    // Open the camera
    //
    Camera zed;
    ERROR_CODE status = zed.open(buildInitParameters(args));

    if (status > ERROR_CODE::SUCCESS) {
        print("Failed to open the camera", status);
        zed.close();
        return 1;
    }

    // Optionally read the motion measured by an external source (wheel odometry, an external VIO,
    // a GNSS-derived displacement), handed to the tracking image by image in the main loop. It comes
    // from a JSON file, or from the SVO itself when it was recorded with one.
    MotionPriorTrace motionPrior;
    bool useMotionPrior = false;

    if (args.motionPriorFile) {
        if (!loadMotionPriorFile(args.motionPriorFile.value(), motionPrior)) {
            print("Failed to load motion prior file: " + args.motionPriorFile.value(), ERROR_CODE::FAILURE, false);
            zed.close();
            return 1;
        }
        useMotionPrior = true;
    } else if (args.svoFile && !args.noMotionPrior && hasMotionPriorChannel(zed)) {
        if (!loadMotionPriorChannel(zed, motionPrior)) {
            zed.close();
            return 1;
        }
        print("Using the motion prior recorded in the SVO (--no-motion-prior ignores it)");
        useMotionPrior = true;
    }

    // The sensor transform is read in the coordinate system the camera was opened with, so it is brought
    // there before it reaches the tracking or the viewer.
    if (useMotionPrior && !expressSensorTransformIn(motionPrior, zed.getInitParameters().coordinate_system)) {
        zed.close();
        return 1;
    }

    if (useMotionPrior) {
        finalizeMotionPriorTrace(motionPrior, args.motionPriorDelayMs);
    } else if (args.motionPriorDelayMs > 0.) {
        print("No motion prior to delay: --motion-prior-delay ignored");
    }

    // Optionally ignore part of the image, for instance the vehicle the camera is mounted on:
    // pixels that move with the camera would otherwise be trusted as static landmarks.
    if (args.roiFile) {
        Mat roi;
        roi.read(args.roiFile->c_str());
        zed.setRegionOfInterest(roi);
    }

    //
    // Enable positional tracking
    //
    PositionalTrackingParameters trackingParameters = buildTrackingParameters(args);
    status = zed.enablePositionalTracking(trackingParameters);

    if (status > ERROR_CODE::SUCCESS) {
        print("Failed to enable positional tracking", status);
        zed.close();
        return 1;
    }

    printTrackingParameters(trackingParameters);

    // A low depth confidence keeps the noisiest measurements out of the tracking.
    RuntimeParameters runtimeParameters;
    runtimeParameters.confidence_threshold = 30;

    //
    // Prepare the display and the optional output files
    //
    Resolution displayResolution = zed.getRetrieveMeasureResolution();
    Mat leftImage;

    // The viewer maps this Mat into OpenGL: CUDA-GL interop for GPU memory, an upload for CPU memory.
    // Voxel mode returns a cloud whose size changes every frame, so the SDK reallocates the Mat and the
    // fixed interop mapping cannot follow it, so CPU memory there.
    Mat pointCloud(displayResolution, MAT_TYPE::F32_C4, args.voxelMode ? MEM::CPU : MEM::GPU);

    // Held by pointer so --no-gui can leave it unbuilt: the constructor is what opens the window.
    std::unique_ptr<SLAMView> view;
    if (!args.noGui) {
        view = std::make_unique<SLAMView>(argc, argv, &leftImage, &pointCloud, zed.getCUDAStream());

        if (useMotionPrior) {
            view->enableMotionPrior(motionPrior.source, motionPrior.sensorTransform);
        }

        // A recording can wait: pausing it freezes the replay where it is, to look at the map at leisure.
        if (args.svoFile) {
            view->enablePauseButton();
        }
    }

    TrajectoryRecorder recorder;
    if (args.trajectoryTUMFile) {
        recorder.openTrajectoryFile(args.trajectoryTUMFile.value());
    }
    if (args.statusLogFile) {
        recorder.openStatusFile(args.statusLogFile.value());
    }

    // Fixed for the lifetime of the camera, so read once rather than on every image.
    const CameraInformation cameraInformation = zed.getCameraInformation();
    const Resolution cameraResolution = cameraInformation.camera_configuration.resolution;

    // On a fisheye camera the tracking runs on the raw fisheye image, so that is the one to show.
    const bool isFisheye = cameraInformation.camera_configuration.calibration_parameters_raw.left_cam.lens_distortion_model
        == LENS_DISTORTION_MODEL::FISHEYE;

    SvoPlaybackClock svoPlayback;
    std::map<uint64_t, Landmark> landmarks;
    std::vector<Landmark2D> landmarks2D;
    std::map<uint64_t, KeyFrame> keyframes;
    std::map<uint64_t, MotionPriorLink> motionPriorLinks;
    uint64_t lastLandmarkUpdate = 0; // in the image clock, like every comparison against it below
    Timestamp previousImageTimestamp = 0;

    //
    // Main loop
    //
    // Without the viewer there is no window to close, so the end of the recording is what ends the run.
    while (!view || view->isOpen()) {
        // Paused: no image is read, so the tracking waits too. The viewer keeps rendering, so the map can
        // still be turned around and clicked.
        if (view && view->isPaused()) {
            view->render();
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            continue;
        }

        // Read the next image without processing it yet: the external motion samples covering it
        // have to reach the tracking before grab() consumes it.
        ERROR_CODE readStatus = zed.read();

        if (readStatus == ERROR_CODE::END_OF_SVOFILE_REACHED) {
            break;
        } else if (readStatus > ERROR_CODE::SUCCESS) {
            print("Failed to read the next image frame", readStatus);
            break;
        }

        Timestamp imageTimestamp = zed.getTimestamp(TIME_REFERENCE::IMAGE);

        if (args.svoFile) {
            svoPlayback.waitForRealTime(imageTimestamp);
        }

        if (useMotionPrior) {
            const size_t firstNew = motionPrior.next;
            ingestMotionPriorUpTo(zed, motionPrior, imageTimestamp);

            if (view) {
                for (size_t i = firstNew; i < motionPrior.next; i++) {
                    view->addMotionPriorSample(motionPrior.samples[i], motionPrior.arrivalTimestamps[i]);
                }
            }
        }

        // Process the image: depth, then the tracking, run here.
        ERROR_CODE grabStatus = zed.grab(runtimeParameters);

        if (args.svoFile) {
            svoPlayback.markProcessed(imageTimestamp);
        }

        if (grabStatus == ERROR_CODE::END_OF_SVOFILE_REACHED) {
            break;
        } else if (grabStatus > ERROR_CODE::SUCCESS) {
            print("Failed to grab image frame", grabStatus);
            break;
        }

        // The image and the point cloud are only ever drawn, so nothing retrieves them without a viewer.
        if (view) {
            // Retrieve the left image
            zed.retrieveImage(leftImage, isFisheye ? VIEW::LEFT_UNRECTIFIED : VIEW::LEFT, MEM::CPU, displayResolution);

            // Retrieve the calculated point cloud
            if (args.voxelMode) {
                zed.retrieveVoxelMeasure(pointCloud, MEASURE::XYZBGRA, MEM::CPU);
            } else {
                zed.retrieveMeasure(pointCloud, MEASURE::XYZBGRA, MEM::GPU, displayResolution);
            }
        }

        // Retrieve the calculated camera pose, and how much the tracking trusts it
        Pose pose;
        zed.getPosition(pose);

        PositionalTrackingStatus trackingStatus = zed.getPositionalTrackingStatus();

        recorder.record(pose, trackingStatus);

        if (useMotionPrior) {
            recordRelativePoseFusionStatus(motionPrior, trackingStatus.relative_pose_fusion_status);
        }

        if (!view) {
            continue;
        }

        // Update the display
        view->updatePoseTransform(pose.pose_data);
        view->updatePositionalTrackingStatus(trackingStatus, imageTimestamp);

        // The landmarks and keyframes are the map the tracking builds. They change slowly, and
        // there are a lot of them, so refresh the display copy once per second.
        if (imageTimestamp.getSeconds() - lastLandmarkUpdate > 1) {
            lastLandmarkUpdate = imageTimestamp.getSeconds();

            zed.getPositionalTrackingLandmarks(landmarks);
            view->updateLandmarks(landmarks);

            zed.getPositionalTrackingKeyframes(keyframes);
            view->updateKeyframes(keyframes);

            // Which keyframes the motion prior holds together, drawn as lines between them.
            if (useMotionPrior) {
                zed.getPositionalTrackingMotionPriorLinks(motionPriorLinks);
                view->updateMotionPriorLinks(motionPriorLinks);
            }
        }

        if (view->isLandmarkModeEnabled()) {
            zed.getPositionalTrackingLandmarks2D(landmarks2D);
            drawLandmarks2D(leftImage, landmarks2D, cameraResolution);
        }

        view->render();
    }

    //
    // Close everything, saving what was asked for on the way out
    //
    recorder.close();

    // The keyframes are read again here rather than reused from the display copy: their poses keep
    // being refined by loop closures until the very last image.
    if (args.keyframeTUMFile) {
        const std::string& keyframeFile = args.keyframeTUMFile.value();
        ERROR_CODE keyframeStatus = zed.getPositionalTrackingKeyframes(keyframes);

        if (keyframeStatus != ERROR_CODE::SUCCESS) {
            print("Failed to retrieve the keyframes, " + keyframeFile + " was not written", keyframeStatus);
        } else if (!recorder.writeKeyframeFile(keyframeFile, keyframes)) {
            print("Failed to write the keyframes to " + keyframeFile, ERROR_CODE::FAILURE, false);
        } else {
            print("Exported " + std::to_string(keyframes.size()) + " keyframes to " + keyframeFile);
        }
    }

    // How much of the trace reached the tracking, and what the tracking did with it.
    if (useMotionPrior) {
        printMotionPriorSummary(motionPrior);
    }

    // Must run before zed.close(): GLUT's atexit destroys the window after main returns,
    // which would unmap CUDA-GL resources after the SDK CUDA context is already gone.
    if (view) {
        view->close();
    }
    pointCloud.free();

    if (args.map) {
        saveAreaMap(zed, args.outputAreaFile.value());
    }

    zed.close();

    return 0;
}

//
// The camera configuration. The tracking runs on the depth, so the depth mode and the coordinate
// system chosen here are also the ones the poses come back in.
//
InitParameters buildInitParameters(const Arguments& args) {
    InitParameters initParameters;

    initParameters.depth_mode = DEPTH_MODE::NEURAL;
    initParameters.coordinate_units = UNIT::METER;
    initParameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    initParameters.sdk_verbose = 1;

    if (args.resolution) {
        initParameters.camera_resolution = args.resolution.value();
    }

    // Where the images come from: a live camera by default, otherwise a recording or a stream.
    if (args.svoFile) {
        initParameters.input.setFromSVOFile(args.svoFile.value().c_str());
    } else if (args.streamIP && args.streamPort) {
        initParameters.input.setFromStream(args.streamIP.value().c_str(), args.streamPort.value());
    } else if (args.streamIP) {
        initParameters.input.setFromStream(args.streamIP.value().c_str());
    }

    return initParameters;
}

//
// The tracking configuration. These are the knobs a positional tracking application tunes; the
// values below are the ones this sample runs with, not necessarily the SDK defaults.
//
PositionalTrackingParameters buildTrackingParameters(const Arguments& args) {
    PositionalTrackingParameters trackingParameters;

    // Loading an area file lets the tracking recognize a place it has already been in, and give
    // poses in that area's frame instead of starting a new one.
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
    trackingParameters.enable_2d_ground_mode = args.enable2dGroundMode;
    trackingParameters.enable_localization_only = false;
    trackingParameters.mode = POSITIONAL_TRACKING_MODE::GEN_3;

    //
    // Start the tracking from a known pose instead of the origin. Useful when the camera starts at
    // a known place of a previously mapped area.
    //
    if (args.customInitialPose) {
        // clang-format off

        // Homogeneous transformation matrix for the initial pose:
        // For example, we have a translation to (2.0, 4.0, 1.0) in the right-most column,
        // and a rotation of 45° about the Z-axis in the upper-left 3x3 portion of the matrix.
        float initialPose[16] = {
            0.7071f, -0.7071f, 0.0f, 2.0f,
            0.7071f,  0.7071f, 0.0f, 4.0f,
            0.0f,     0.0f,    1.0f, 1.0f,
            0.0f,     0.0f,    0.0f, 1.0f
        };
        // clang-format on

        trackingParameters.initial_world_transform = Transform(initialPose);
    }

    return trackingParameters;
}
