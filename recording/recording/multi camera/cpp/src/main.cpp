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

/****************************************************************************************
** This sample shows how to record video in Stereolabs SVO format                      **
** SVO video files can be played with the ZED API and used with its different modules  **
*****************************************************************************************/

// Standard includes
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// ZED includes
#include <sl/Camera.hpp>
#ifndef _WIN32
#include <sl/CameraOne.hpp>
#endif

// Sample includes
#include "utils.hpp"

/// \brief Acquisition function template
/// \tparam CameraType Type of the camera (e.g., sl::Camera, sl::CameraOne)
/// \param zed Reference to the camera object
template <typename CameraType>
void acquisition(CameraType& zed) {
    auto infos = zed.getCameraInformation();

    while (!exit_app) {
        if (zed.grab() <= sl::ERROR_CODE::SUCCESS) {
            // If needed, add more processing here
        }
    }

    std::cout << infos.camera_model << "[" << infos.serial_number << "] QUIT \n";

    // disable Streaming
    zed.disableStreaming();
    // close the Camera
    zed.close();
}

/// Function to set the depth mode in InitParameters
inline void setDepthMode(sl::InitParameters &ip) {
    ip.depth_mode = sl::DEPTH_MODE::NONE; // No depth mode for this example
}

/// Function to set the depth mode in InitParametersOne
inline void setDepthMode(sl::InitParametersOne &ip) {
    // NA
}

/// \brief Open a camera with the given serial number, and enable recording
/// \tparam CameraType Type of the camera (e.g., sl::Camera, sl::CameraOne)
/// \tparam IP Type of the InitParameters (e.g., sl::InitParameters, sl::InitParametersOne)
/// \param zed Reference to the camera object
/// \param sn Serial number of the camera
/// \param camera_fps Desired camera frame rate (default is 30)
template<typename CameraType, typename IP>
bool openCamera(CameraType& zed, const int sn, const int camera_fps = 30) {

    IP init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION::AUTO;
    setDepthMode(init_parameters);
    init_parameters.input.setFromSerialNumber(sn);
    init_parameters.camera_fps = camera_fps;

    // Open the camera
    const sl::ERROR_CODE open_err = zed.open(init_parameters);
    if (open_err <= sl::ERROR_CODE::SUCCESS) {
        std::cout << toString(zed.getCameraInformation().camera_model) << "_SN" << sn << " Opened" << std::endl;
    } else {
        std::cout << "ZED SN:" << sn << " Error: " << open_err << std::endl;
        zed.close();
        return false;
    }

    // Enable streaming
    sl::RecordingParameters recording_params;
    std::string svo_filename = std::string(sl::toString(zed.getCameraInformation().camera_model)) + "_SN" + std::to_string(sn) + ".svo2";;
    svo_filename.erase(std::remove(svo_filename.begin(), svo_filename.end(), ' '), svo_filename.end()); // Remove spaces from the filename
    recording_params.video_filename.set(svo_filename.c_str()); 
    recording_params.compression_mode = sl::SVO_COMPRESSION_MODE::H264;
    const sl::ERROR_CODE recording_err = zed.enableRecording(recording_params);
    if (recording_err <= sl::ERROR_CODE::SUCCESS) {
        std::cout << toString(zed.getCameraInformation().camera_model) << "_SN" << sn << " Enabled recording" << std::endl;
    } else {
        std::cout << "ZED SN:" << sn << " Recording initialization error: " << recording_err << std::endl;
        zed.close();
        return false;
    }

    std::cout << "Recording SVO " << recording_params.video_filename << std::endl;
    return true;
}

/// Function to print device information
void printDeviceInfo(const std::vector<sl::DeviceProperties> & devs) {
    for (const auto &dev : devs)
        std::cout << "ID : " << dev.id << ", model : " << dev.camera_model
            << " , S/N : " << dev.serial_number
            << " , state : " << dev.camera_state
            << std::endl;
}

int main(int argc, char **argv) {
    // Get the list of available ZED cameras
    const std::vector<sl::DeviceProperties> dev_stereo_list = sl::Camera::getDeviceList();
    printDeviceInfo(dev_stereo_list);

#ifndef _WIN32
    const std::vector<sl::DeviceProperties> dev_one_list = sl::CameraOne::getDeviceList();
    printDeviceInfo(dev_one_list);
#else
    const std::vector<sl::DeviceProperties> dev_one_list;
#endif

    const int nb_one = dev_one_list.size();
    const int nb_stereo = dev_stereo_list.size();
    if (nb_one + nb_stereo == 0) {
        std::cout << "No ZED Detected, exit program" << std::endl;
        return EXIT_FAILURE;
    }

    bool zed_open = false;

    // Open the Stereo cameras
    std::vector<sl::Camera> zeds_stereo(nb_stereo);
    for (int z = 0; z < nb_stereo; ++z) {
        zed_open |= openCamera<sl::Camera, sl::InitParameters>(zeds_stereo[z], dev_stereo_list[z].serial_number);
    }

#ifndef _WIN32
    // Open the Mono cameras
    std::vector<sl::CameraOne> zeds_one(nb_one);
    for (int z = 0; z < nb_one; ++z) {
        zed_open |= openCamera<sl::CameraOne, sl::InitParametersOne>(zeds_one[z], dev_one_list[z].serial_number);
    }
#endif

    if (!zed_open) {
        std::cout << "No ZED opened, exit program" << std::endl;
        return EXIT_FAILURE;
    }


    // Create a grab thread for each opened camera
    std::vector<std::thread> thread_pool(nb_stereo + nb_one); // compute threads
    for (int z = 0; z < nb_stereo; z++) {
        if (zeds_stereo[z].isOpened())
            thread_pool[z] = std::thread(acquisition<sl::Camera>, std::ref(zeds_stereo[z]));
    }
#ifndef _WIN32
    for (int z = 0; z < nb_one; z++) {
        if (zeds_one[z].isOpened())
            thread_pool[nb_stereo + z] = std::thread(acquisition<sl::CameraOne>, std::ref(zeds_one[z]));
    }
#endif

    // Ctrl+C to close
    SetCtrlHandler();
    std::cout << "Press Ctrl+C to exit" << std::endl;
    while (!exit_app) { // main loop
        sl::sleep_ms(20);
    }

    // stop all running threads
    std::cout << "Exit signal, closing ZEDs" << std::endl;
    sl::sleep_ms(100);

    // Wait for every thread to be stopped
    for (auto& th : thread_pool)
        if (th.joinable())
            th.join();

    std::cout << "Program exited" << std::endl;
    return EXIT_SUCCESS;
}


