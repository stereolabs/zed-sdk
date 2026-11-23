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
// Motion prior recording: record an SVO that carries a motion prior next to the images, so the
// Positional Tracking sample can replay both together, each sample reaching the tracking at the same
// point it reached the application live.
//
// The motion source here is the rotation of the camera between two images, pre-integrated from the raw
// gyroscope of its IMU. It stands in for your own source (wheel odometry, an external VIO): replace
// integrateGyro() with whatever reads it, the rest stays the same.
//
//   open() -> enableRecording() -> [ read() -> motion sample -> ingestDataIntoSVO() -> grab() ] -> close()
//

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include <csignal>
#include <iostream>
#include <string>
#include <vector>

#include "MotionPriorChannel.hpp"

using namespace sl;

namespace {

    // Per-axis uncertainty of the rotation over one sample, in radians: 0.01 rad, about half a degree.
    // Between two images the integrated gyroscope and the images usually agree within a quarter of a
    // degree, and the margin covers fast motion. Too small a value gets every disagreement with the images rejected by the tracking.
    constexpr float rotationStd = 0.01f;

    volatile std::sig_atomic_t stopRequested = 0;

    void onSignal(int) {
        stopRequested = 1;
    }

    void print(const std::string& message, ERROR_CODE status = ERROR_CODE::SUCCESS) {
        std::cout << "[Sample]" << (status > ERROR_CODE::SUCCESS ? "[Error] " : " ") << message;
        if (status > ERROR_CODE::SUCCESS) {
            std::cout << " | " << toString(status) << ": " << toVerbose(status);
        }
        std::cout << std::endl;
    }

    //
    // The motion source: the rotation of the IMU between two images, pre-integrated from the raw
    // gyroscope, along the axes of the IMU.
    //
    struct GyroIntegrator {
        Timestamp lastTimestamp = 0;    // the last gyroscope measurement integrated
        sl::float3 lastAngularVelocity; // its angular velocity, in rad/s
    };

    // Longest hole between two gyroscope measurements integrated across. A longer one (dropped data, a
    // paused input) leaves the motion over it unknown, so the interval restarts after it.
    constexpr uint64_t maxGyroGapNs = 100000000; // 100 ms

    constexpr float degreesToRadians = 3.14159265358979f / 180.f;

    // Integrates every gyroscope measurement received since the previous call into one rotation, and
    // fills sample with it. Returns false when there is nothing to give yet: on the first call, which
    // only starts the interval, and when no new measurement arrived.
    bool integrateGyro(Camera& zed, GyroIntegrator& gyro, RelativePoseData& sample) {
        // Right after read(), the batch holds the IMU measurements taken between the previous image and
        // the one just read: chained call after call, they cover the whole recording without a gap.
        std::vector<SensorsData> batch;
        if (zed.getSensorsDataBatch(batch) != ERROR_CODE::SUCCESS || batch.empty()) {
            return false;
        }

        Timestamp start = gyro.lastTimestamp;
        Transform rotation; // identity: no rotation yet

        for (const SensorsData& measurement : batch) {
            const uint64_t timestamp = measurement.imu.timestamp.getNanoseconds();
            if (!measurement.imu.is_available || timestamp <= gyro.lastTimestamp.getNanoseconds()) {
                continue;
            }

            // Calibrated angular velocity (bias, scale and misalignment corrected), in deg/s.
            const sl::float3 angularVelocity = measurement.imu.angular_velocity * degreesToRadians;

            if (gyro.lastTimestamp.getNanoseconds() > 0) {
                const uint64_t elapsed = timestamp - gyro.lastTimestamp.getNanoseconds();

                if (elapsed > maxGyroGapNs) {
                    rotation = Transform();
                    start = Timestamp(timestamp);
                } else {
                    // The rate at both ends of the step averaged, then turned into the rotation it gives
                    // over that step. Composed on the right: each step turns about the axes the IMU had
                    // at its start, so the result is the end orientation in the frame of the start.
                    const sl::float3 step = (angularVelocity + gyro.lastAngularVelocity) * (0.5f * elapsed * 1e-9f);
                    Transform increment;
                    increment.setRotationVector(step);
                    rotation = Transform(rotation * increment);
                }
            }

            gyro.lastTimestamp = Timestamp(timestamp);
            gyro.lastAngularVelocity = angularVelocity;
        }

        if (start.getNanoseconds() == 0 || gyro.lastTimestamp.getNanoseconds() <= start.getNanoseconds()) {
            return false;
        }

        sample = RelativePoseData();
        sample.start_timestamp = start;
        sample.end_timestamp = gyro.lastTimestamp;
        sample.motion = rotation;
        // A gyroscope says nothing about how far the camera moved: only the rotation is measured.
        sample.measured_axes = POSE_AXIS::ROTATION;
        // Its uncertainty, in the [tx, ty, tz, rx, ry, rz] order: the rotation block only, the translation
        // being ignored with its axes.
        for (int axis = 3; axis < 6; axis++) {
            sample.covariance[axis * 6 + axis] = rotationStd * rotationStd;
        }
        return true;
    }

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]).find(".svo") == std::string::npos) {
        std::cout << "Usage: " << argv[0] << " <output.svo2> [input.svo2]\n"
                  << "  Records the camera, or re-records an SVO2, with the camera rotation integrated from its gyroscope\n"
                  << "  stored as a motion prior. Stop a live recording with Ctrl-C.\n";
        return EXIT_FAILURE;
    }
    const std::string outputFile = argv[1];
    const bool fromSvo = argc > 2;

    //
    // Open the camera. Nothing is computed from the images, so no depth.
    //
    InitParameters initParameters;
    initParameters.depth_mode = DEPTH_MODE::NONE;
    // The sensor transform below is written in this coordinate system, and the channel says so.
    initParameters.coordinate_system = COORDINATE_SYSTEM::IMAGE;
    initParameters.coordinate_units = UNIT::METER;
    if (fromSvo) {
        initParameters.input.setFromSVOFile(argv[2]);
        initParameters.svo_real_time_mode = false;
    }

    Camera zed;
    ERROR_CODE status = zed.open(initParameters);
    if (status > ERROR_CODE::SUCCESS) {
        print("Failed to open the camera", status);
        return EXIT_FAILURE;
    }

    const SensorsConfiguration sensorsConfiguration = zed.getCameraInformation().sensors_configuration;
    if (!sensorsConfiguration.gyroscope_parameters.isAvailable) {
        print("This camera has no IMU to measure the rotation with", ERROR_CODE::SENSORS_NOT_AVAILABLE);
        zed.close();
        return EXIT_FAILURE;
    }

    //
    // Start recording. The motion prior goes into the same SVO as the images.
    //
    RecordingParameters recordingParameters;
    recordingParameters.video_filename.set(outputFile.c_str());
    recordingParameters.compression_mode = SVO_COMPRESSION_MODE::H265;
    status = zed.enableRecording(recordingParameters);
    if (status > ERROR_CODE::SUCCESS) {
        print("Failed to start recording " + outputFile, status);
        zed.close();
        return EXIT_FAILURE;
    }

    std::signal(SIGINT, onSignal);
    print("Recording " + outputFile + " with the camera IMU rotation as motion prior" + (fromSvo ? "" : ", Ctrl-C to stop"));

    MotionPriorChannel channel;
    GyroIntegrator gyro;
    Timestamp lastImageTimestamp = 0;
    size_t images = 0;
    size_t samples = 0;
    size_t failures = 0;

    while (!stopRequested) {
        // Read the next image without processing it yet. In an application running the positional
        // tracking, this is where the motion prior is handed over with ingestMotionPrior(): before the
        // grab() that processes the image it covers.
        status = zed.read();
        if (status == ERROR_CODE::END_OF_SVOFILE_REACHED) {
            break;
        } else if (status > ERROR_CODE::SUCCESS) {
            print("Failed to read the next image", status);
            break;
        }

        const Timestamp imageTimestamp = zed.getTimestamp(TIME_REFERENCE::IMAGE);

        // Once, before the first sample: where the source sits. The gyroscope measures along the axes of the
        // IMU, so its sensor transform is where the IMU is in the camera.
        if (images == 0) {
            if (channel.writeConfig(zed, sensorsConfiguration.camera_imu_transform, imageTimestamp) != ERROR_CODE::SUCCESS) {
                failures++;
            }
        }

        RelativePoseData sample;
        if (integrateGyro(zed, gyro, sample)) {
            // Stamped with when the sample reached the application. Live, that is now, on the same clock as
            // the images; re-recording an SVO, the image it arrived with stands for it.
            const Timestamp arrival = fromSvo ? imageTimestamp : getCurrentTimeStamp();

            // The last image read before this one: the replay hands the sample over before this image.
            const ERROR_CODE written = channel.writeSample(zed, sample, lastImageTimestamp, arrival);
            if (written == ERROR_CODE::SUCCESS) {
                samples++;
            } else if (failures++ == 0) {
                print("Failed to write a motion prior sample into the SVO", written);
            }
        }

        // The image itself goes into the SVO here.
        status = zed.grab();
        if (status > ERROR_CODE::SUCCESS && status != ERROR_CODE::END_OF_SVOFILE_REACHED) {
            print("Failed to record the image", status);
            break;
        }

        lastImageTimestamp = imageTimestamp;
        images++;
    }

    zed.disableRecording();
    zed.close();

    print("Recorded " + std::to_string(images) + " images and " + std::to_string(samples) + " motion prior samples into " + outputFile);
    if (failures > 0) {
        print(std::to_string(failures) + " motion prior messages could not be written", ERROR_CODE::FAILURE);
        return EXIT_FAILURE;
    }

    print("Replay it with: ZED_Positional_Tracking --svo " + outputFile);
    return EXIT_SUCCESS;
}
