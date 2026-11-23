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

#include "SampleConsole.hpp"

#include <cassert>
#include <iostream>
#include <sstream>

using namespace sl;

namespace {

    /**
     * @brief Colors a boolean for the console.
     * @param value The boolean.
     * @return "true" in green or "false" in red.
     */
    std::string colorBool(bool value) {
        return value ? "\033[32mtrue\033[0m" : "\033[31mfalse\033[0m";
    }

    /**
     * @brief Colors a value for the console.
     * @param value The value, as text.
     * @return The value in cyan.
     */
    std::string colorValue(const std::string& value) {
        return "\033[36m" + value + "\033[0m";
    }

    /**
     * @brief Colors a number for the console, formatted like std::cout does ("-1", not "-1.000000").
     * @param value The number.
     * @return The number in cyan.
     */
    std::string colorValue(float value) {
        std::ostringstream oss;
        oss << value;
        return colorValue(oss.str());
    }

    /**
     * @brief Severity of a printed line, which sets its tag.
     */
    enum class Severity {
        INFO,
        WARNING,
        ERROR
    };

    /**
     * @brief Prints a line prefixed with [Sample] and the tag of its severity.
     * @param severity Severity of the line.
     * @param message The line.
     * @param detail Text appended after a separator, empty for none.
     */
    void printLine(Severity severity, const std::string& message, const std::string& detail) {
        std::cout << "\033[36m[Sample]\033[0m";

        if (severity == Severity::ERROR) {
            std::cout << " \033[31m[Error]\033[0m ";
        } else if (severity == Severity::WARNING) {
            std::cout << " \033[33m[Warning]\033[0m ";
        } else {
            std::cout << " ";
        }

        std::cout << message;

        if (!detail.empty()) {
            std::cout << " | " << detail;
        }

        std::cout << "\n";
    }

} // namespace

void printHeader(const std::string& text) {
    const int width = 80;
    const std::string border(width, '=');
    int padding = (width - text.length()) / 2;

    std::cout << "\n" << border << "\n";
    std::cout << std::string(padding, ' ') << text << "\n";
    std::cout << border << "\n\n";
}

void print(const std::string& message, std::optional<ERROR_CODE> errorCode, bool showErrorDetail) {
    if (errorCode && errorCode > ERROR_CODE::SUCCESS) {
        const std::string detail
            = showErrorDetail ? std::string(toString(errorCode.value()).c_str()) + ": " + toVerbose(errorCode.value()).c_str() : "";
        printLine(Severity::ERROR, message, detail);
    } else if (errorCode && errorCode < ERROR_CODE::SUCCESS) {
        printLine(Severity::WARNING, message, "");
    } else {
        printLine(Severity::INFO, message, "");
    }
}

void print(const std::string& message, SENSORS_ERROR_CODE errorCode) {
    if (errorCode > SENSORS_ERROR_CODE::SUCCESS) {
        printLine(Severity::ERROR, message, toString(errorCode).c_str());
    } else if (errorCode < SENSORS_ERROR_CODE::SUCCESS) {
        printLine(Severity::WARNING, message, toString(errorCode).c_str());
    } else {
        printLine(Severity::INFO, message, "");
    }
}

void printCameraErrors(Sensors& sensors, const RigConfig& rig, const std::vector<SensorDeviceIdentifier>& cameraIds, bool afterProcess) {
    assert(cameraIds.size() == rig.cameras.size());

    BatchedData<ERROR_CODE> errors;
    if (afterProcess) {
        sensors.getProcessErrorCodes(errors);
    } else {
        sensors.getErrorCodes(errors);
    }

    for (size_t i = 0; i < cameraIds.size(); i++) {
        auto error = errors.find(cameraIds[i]);
        if (error != errors.end() && error->second != ERROR_CODE::SUCCESS) {
            print("  camera '" + rig.cameras[i].label + "'", error->second);
        }
    }
}

void printRig(Sensors& sensors, const RigConfig& rig, const std::vector<SensorDeviceIdentifier>& cameraIds, size_t trackingCamera) {
    assert(cameraIds.size() == rig.cameras.size());
    assert(trackingCamera < cameraIds.size());

    printHeader("Rig");

    for (size_t i = 0; i < cameraIds.size(); i++) {
        const CameraInformation information = sensors.getCameraInformation(Resolution(0, 0), cameraIds[i]);
        const Resolution resolution = information.camera_configuration.resolution;

        Transform pose;
        sensors.getSensorPose(pose, cameraIds[i]);
        const Translation translation = pose.getTranslation();

        std::ostringstream line;
        line << rig.cameras[i].label << (i == trackingCamera ? " (drives the tracking)" : "") << ": " << information.camera_model << " SN "
             << information.serial_number << ", " << resolution.width << "x" << resolution.height << " at "
             << information.camera_configuration.fps << " FPS, at (" << translation.tx << ", " << translation.ty << ", " << translation.tz
             << ") m on the rig";
        print(line.str());
    }
}

void printTrackingParameters(const PositionalTrackingSensorsParameters& trackingParameters) {
    printHeader("Positional Tracking Parameters");
    print("multi_sensor_mode     " + colorValue(std::string(toString(trackingParameters.multi_sensor_mode).c_str())));
    print("set_floor_as_origin   " + colorBool(trackingParameters.set_floor_as_origin));
    print("set_gravity_as_origin " + colorBool(trackingParameters.set_gravity_as_origin));
    print("enable_area_memory    " + colorBool(trackingParameters.enable_area_memory));
    print("enable_pose_smoothing " + colorBool(trackingParameters.enable_pose_smoothing));
    print("enable_imu_fusion     " + colorBool(trackingParameters.enable_imu_fusion));
    print("set_as_static         " + colorBool(trackingParameters.set_as_static));
    print("depth_min_range       " + colorValue(trackingParameters.depth_min_range));
    print("mode                  " + colorValue(std::string(toString(trackingParameters.mode).c_str())));
}

void printInteraction() {
    printHeader("Interaction");
    print("'space' to toggle camera view visibility");
    print("'n' to show the image of the next camera of the rig");
    print("'d' to switch background color from dark to light");
    print("'l' to enable / disable landmark display");
    print("'c' to show / hide the rig path");
    print("'k' or the Pause button to pause / resume an SVO replay");
    print("'f' to follow the rig");
    print("'z' to reset the view");
    print("'ctrl' + drag to rotate");
    print("'esc' to exit");
    std::cout << "\n";
}
