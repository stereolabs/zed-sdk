///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2024, STEREOLABS.
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

/***************************************************************************
 ** This sample shows how to use global localization on real-world map    **
 **************************************************************************/

#include <sl/Camera.hpp>

#include "gnss_reader/IGNSSReader.h"
#include "gnss_reader/GPSDReader.hpp"
#include "exporter/KMLExporter.h"
#include "exporter/GNSSSaver.h"

bool exit_app = false;

// Handle the CTRL-C keyboard signal
#ifdef _WIN32
#include <Windows.h>

void CtrlHandler(DWORD fdwCtrlType) {
    exit_app = (fdwCtrlType == CTRL_C_EVENT);
}
#else
#include <signal.h>
void nix_exit_handler(int s) {
    std::cout << "Receive CTRL + C will stop application" << std::endl;
    exit_app = true;
}
#endif

// Set the function to handle the CTRL-C
void SetCtrlHandler() {
#ifdef _WIN32
    SetConsoleCtrlHandler((PHANDLER_ROUTINE) CtrlHandler, TRUE);
#else // unix
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = nix_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
#endif
}

int main(int argc, char **argv) {
    // Open the camera
    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.depth_mode = sl::DEPTH_MODE::NONE;
    init_params.sdk_verbose = 1;
    sl::ERROR_CODE camera_open_error = zed.open(init_params);
    if (camera_open_error != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "[ZED][ERROR] Can't open ZED camera" << std::endl;
        return EXIT_FAILURE;
    }

    // Enable SVO recording:
    std::string svo_path = "ZED_SN" + std::to_string(zed.getCameraInformation().serial_number) + "_" + getCurrentDatetime() + ".svo2";
    sl::String path_output(svo_path.c_str());
    auto returned_state = zed.enableRecording(sl::RecordingParameters(path_output, sl::SVO_COMPRESSION_MODE::H265_LOSSLESS));
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "Recording ZED : " << returned_state << std::endl;
        zed.close();
        return EXIT_FAILURE;
    }

    // Handle CTRL + C
    SetCtrlHandler();

    // Enable GNSS data producing:
    GPSDReader gnss_reader;
    gnss_reader.initialize(&exit_app);

    std::cout << "Start grabbing data... Global localization data will be displayed on the Live Server" << std::endl;

    GNSSSaver gnss_data_saver(&zed);
    while (!exit_app) {
        sl::ERROR_CODE err = zed.grab();
        if(err != sl::ERROR_CODE::SUCCESS)
            std::cout << "ZED has error: " << err << std::endl;
            
        // Get GNSS data:
        sl::GNSSData input_gnss;
        if (gnss_reader.grab(input_gnss) == sl::ERROR_CODE::SUCCESS) {
            if(input_gnss.ts.getNanoseconds() == 0)
                continue;
            // Save current GNSS data to KML file:
            saveKMLData("raw_gnss.kml", input_gnss);
            // Save GNSS data into JSON:
            gnss_data_saver.addGNSSData(input_gnss);
        }
    }
    zed.close();

    closeAllKMLWriter();
    return EXIT_SUCCESS;
}
