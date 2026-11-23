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

#include "SampleConsole.hpp"

#include <iostream>
#include <sstream>

using namespace sl;

namespace {

    std::string colorBool(bool value) {
        return value ? "\033[32mtrue\033[0m" : "\033[31mfalse\033[0m";
    }

    std::string colorValue(const std::string& value) {
        return "\033[36m" + value + "\033[0m";
    }

    // Formatted like std::cout would, so a whole number stays "-1" rather than "-1.000000".
    std::string colorValue(float value) {
        std::ostringstream oss;
        oss << value;
        return colorValue(oss.str());
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
    std::cout << "\033[36m[Sample]\033[0m";

    if (errorCode && errorCode > ERROR_CODE::SUCCESS) {
        std::cout << " \033[31m[Error]\033[0m ";
    } else if (errorCode && errorCode < ERROR_CODE::SUCCESS) {
        std::cout << " \033[33m[Warning]\033[0m ";
    } else {
        std::cout << " ";
    }

    std::cout << message;

    if (errorCode && errorCode > ERROR_CODE::SUCCESS && showErrorDetail) {
        std::cout << " | " << toString(errorCode.value()) << ": " << toVerbose(errorCode.value());
    }

    std::cout << "\n";
}

void printTrackingParameters(const PositionalTrackingParameters& trackingParameters) {
    printHeader("Positional Tracking Parameters");
    print("set_floor_as_origin   " + colorBool(trackingParameters.set_floor_as_origin));
    print("set_gravity_as_origin " + colorBool(trackingParameters.set_gravity_as_origin));
    print("enable_area_memory    " + colorBool(trackingParameters.enable_area_memory));
    print("enable_pose_smoothing " + colorBool(trackingParameters.enable_pose_smoothing));
    print("enable_imu_fusion     " + colorBool(trackingParameters.enable_imu_fusion));
    print("set_as_static         " + colorBool(trackingParameters.set_as_static));
    print("depth_min_range       " + colorValue(trackingParameters.depth_min_range));
    print("enable_2d_ground_mode " + colorBool(trackingParameters.enable_2d_ground_mode));
    print("mode                  " + colorValue(std::string(toString(trackingParameters.mode).c_str())));

    printHeader("Interaction");
    print("'space' to toggle camera view visibility");
    print("'d' to switch background color from dark to light");
    print("'p' to enable / disable current live point cloud display");
    print("'l' to enable / disable landmark display");
    print("'c' to show / hide the camera path");
    print("'m' to show / hide the motion prior (hides the camera path while shown)");
    print("'t' to show / hide the motion prior timeline");
    print("'k' or the Pause button to pause / resume an SVO replay");
    print("'f' to follow the camera");
    print("'z' to reset the view");
    print("'ctrl' + drag to rotate");
    print("'esc' to exit");
    std::cout << "\n";
}
