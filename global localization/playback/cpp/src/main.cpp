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
 ** API with pre-recorded GNSS data                                       **
 **************************************************************************/

#include <iostream>
#include <future>

#include <opencv2/opencv.hpp>

#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>
#include "display/GenericDisplay.h"
#include "exporter/KMLExporter.h"
#include "GNSSReplay.hpp"

std::vector<std::string> split(const std::string &s, const std::string &delimiter);
cv::Mat slMat2cvMat(sl::Mat& input);

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage ./ZED_GNSS_playback <svo-file> <gnss-data-file-path optional if SVO2 include GNSS data> <optional_mask.png> <opt_gnss_antenna_pos without spaces like 0.5,0.1,-1 >" << std::endl;
        return EXIT_FAILURE;
    }

    // The GNSS data are either extracted from a external json OR preferably from the SVO v2 custom data

    std::string svo_name, gnss_file, mask_file, gnss_antenna_position_str;
    for(int i = 1; i<argc; i++){
        std::string arg(argv[i]);
        if(arg.find(".svo") != std::string::npos)
            svo_name = arg;
        if(arg.find(".json") != std::string::npos)
            gnss_file = arg;
        if(arg.find(".png") != std::string::npos || arg.find(".jpg") != std::string::npos)
            mask_file = arg;
        if(arg.find(",") != std::string::npos)
            gnss_antenna_position_str = arg;
    }
    sl::float3 gnss_antenna_position;
    // GNSS input parsing
    if(!gnss_antenna_position_str.empty()){
        auto str_list = split(gnss_antenna_position_str, ",");
        if(str_list.size() == 3){
            gnss_antenna_position.x = std::stof(str_list[0]);
            gnss_antenna_position.y = std::stof(str_list[1]);
            gnss_antenna_position.z = std::stof(str_list[2]);

            std::cout << "GNSS antenna position: [" << gnss_antenna_position.x << "," << gnss_antenna_position.y << "," << gnss_antenna_position.z << "]" << std::endl;
        } else std::cerr << "Invalid GNSS Position input, ignoring" << std::endl;
    }
    
    // Open the camera
    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_params.coordinate_units = sl::UNIT::METER;
    init_params.input.setFromSVOFile(svo_name.c_str());
    sl::ERROR_CODE camera_open_error = zed.open(init_params);
    if (camera_open_error != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "[ZED][ERROR] Can't open ZED camera" << std::endl;
        return EXIT_FAILURE;
    }

    // Set Region of Interest
    if(!mask_file.empty()) { 
        std::cout << "Loading Region of interest file" << std::endl;
        sl::Mat mask_roi;
        auto err = mask_roi.read(mask_file.c_str());
        if(err == sl::ERROR_CODE::SUCCESS)
            zed.setRegionOfInterest(mask_roi, {sl::MODULE::ALL});
        else
            std::cout << "Error loading Region of Interest file: " << err << std::endl;
    }

    // Enable positional tracking:
    sl::PositionalTrackingParameters pose_tracking_params;
    pose_tracking_params.mode = sl::POSITIONAL_TRACKING_MODE::GEN_2;
    pose_tracking_params.enable_area_memory = false;
    auto positional_init = zed.enablePositionalTracking(pose_tracking_params);
    if (positional_init != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "[ZED][ERROR] Can't start tracking of camera" << std::endl;
        return EXIT_FAILURE;
    }

    // Display
    sl::Resolution display_resolution(1280, 720);
    sl::Mat left_img;

    // Create Fusion object:
    sl::Fusion fusion;
    sl::InitFusionParameters init_fusion_param;
    init_fusion_param.coordinate_system = init_params.coordinate_system; //sl::COORDINATE_SYSTEM::IMAGE;
    init_fusion_param.coordinate_units = init_params.coordinate_units; //sl::UNIT::METER;
    init_fusion_param.verbose = true;
    sl::FUSION_ERROR_CODE fusion_init_code = fusion.init(init_fusion_param);
    if (fusion_init_code != sl::FUSION_ERROR_CODE::SUCCESS) {
        std::cerr << "[Fusion][ERROR] Failed to initialize fusion, error: " << fusion_init_code << std::endl;
        return EXIT_FAILURE;
    }

    // Enable odometry publishing:
    zed.startPublishing();
    /// Run a first grab for starting sending data:
    while (zed.grab() != sl::ERROR_CODE::SUCCESS) {
    }

    // Subscribe to Odometry
    sl::CameraIdentifier uuid(zed.getCameraInformation().serial_number);
    fusion.subscribe(uuid);
    // Enable positional tracking for Fusion object
    sl::PositionalTrackingFusionParameters positional_tracking_fusion_parameters;
    sl::GNSSCalibrationParameters gnss_calibration_parameter;
    gnss_calibration_parameter.enable_reinitialization = false;
    gnss_calibration_parameter.enable_translation_uncertainty_target = false;
    gnss_calibration_parameter.gnss_vio_reinit_threshold = 5;
    gnss_calibration_parameter.target_yaw_uncertainty = 1e-2;
    // Set the antenna position relative to the camera system here:
    gnss_calibration_parameter.gnss_antenna_position = gnss_antenna_position;
    positional_tracking_fusion_parameters.gnss_calibration_parameters = gnss_calibration_parameter;
    positional_tracking_fusion_parameters.enable_GNSS_fusion = true;
    sl::FUSION_ERROR_CODE tracking_error_code = fusion.enablePositionalTracking(positional_tracking_fusion_parameters);
    if(tracking_error_code != sl::FUSION_ERROR_CODE::SUCCESS){
        std::cout << "[Fusion][ERROR] Could not start tracking, error: " << tracking_error_code << std::endl;
        return EXIT_FAILURE;
    }

    // Setup viewer:
    GenericDisplay viewer;
    viewer.init(argc, argv);
    std::cout << "Start grabbing data... Global localization data will be displayed on the Live Server" << std::endl;
    std::cout << "To run the Live Server (web interface), go to 'map server' folder and run 'python3 -m http.server 8000' then open a browser to 'http://0.0.0.0:8000/'" << std::endl << std::endl;
    sl::Pose zed_pose;


    GNSSReplay gnss_replay(gnss_file, &zed);
    while (viewer.isAvailable()) {
        // Grab camera:
        auto zed_status = zed.grab();
        if (zed_status == sl::ERROR_CODE::SUCCESS) {
            // You can still use the classical getPosition for your application, just not that the position returned by this method
            // is the position without any GNSS/cameras fusion
            zed.getPosition(zed_pose, sl::REFERENCE_FRAME::WORLD);
            zed.retrieveImage(left_img, sl::VIEW::LEFT, sl::MEM::CPU, display_resolution);
            cv::imshow("left", slMat2cvMat(left_img));
            cv::waitKey(10);
        }

        if (zed_status == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
            break;

        // Get GNSS data:
        sl::GNSSData input_gnss;
        sl::GNSSData input_gnss_sync;

        if (gnss_replay.grab(input_gnss, zed_pose.timestamp.getNanoseconds()) == sl::FUSION_ERROR_CODE::SUCCESS) {
            // Display it on the Live Server:
            auto ingest_error = fusion.ingestGNSSData(input_gnss);
            saveKMLData("raw_gnss.kml", input_gnss);
            if (ingest_error != sl::FUSION_ERROR_CODE::SUCCESS) {
                std::cout << "Ingest error occurred when ingesting GNSSData: " << ingest_error << std::endl;
            }
        }
        
        // Process data and compute positions:
        if (fusion.process() == sl::FUSION_ERROR_CODE::SUCCESS) {
            
            sl::Pose fused_position;
            // Get position into the ZED CAMERA coordinate system:
            sl::POSITIONAL_TRACKING_STATE current_state = fusion.getPosition(fused_position);
            if (current_state == sl::POSITIONAL_TRACKING_STATE::OK) {
                // Display it on OpenGL:
                sl::FusedPositionalTrackingStatus fused_status = fusion.getFusedPositionalTrackingStatus();
                viewer.updatePoseData(fused_position.pose_data, fused_status);
            }

            fusion.getCurrentGNSSData(input_gnss_sync);
            viewer.updateRawGeoPoseData(input_gnss_sync);

            // Get position into the GNSS coordinate system - this needs a initialization between CAMERA 
            // and GNSS. When the initialization is finish the getGeoPose will return sl::POSITIONAL_TRACKING_STATE::OK
            sl::GeoPose current_geopose;
            auto current_geopose_status = fusion.getGeoPose(current_geopose);
            if (current_geopose_status == sl::GNSS_FUSION_STATUS::OK) {
                // Display it on the Live Server:
                viewer.updateGeoPoseData(current_geopose);

                sl::Transform current_calibration = fusion.getGeoTrackingCalibration();
            } 

            if (0) {
                // GNSS coordinate system to ZED coordinate system is not initialize yet
                // The initialisation between the coordinates system is basically an optimization problem that
                // Try to fit the ZED computed path with the GNSS computed path. In order to do it just move
                // your system by the distance you specified in positional_tracking_fusion_parameters.gnss_initialisation_distance
                float yaw_std;
                sl::float3 position_std;
                fusion.getCurrentGNSSCalibrationSTD(yaw_std, position_std);

                if(yaw_std != -1.f)
                    std::cout << "GNSS State: " << current_geopose_status << ": calibration uncertainty yaw_std " << yaw_std << " rad position_std " << position_std[0] << " m, " << position_std[1] << " m, " << position_std[2] << " m\t\t\t\r";
            }
        }
    }

    closeAllKMLWriter();
    fusion.close();
    zed.close();
    
    return 0;
}

std::vector<std::string> split(const std::string &s, const std::string &delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0, end = 0;
    while ((end = s.find(delimiter, start)) != std::string::npos) {
        tokens.push_back(s.substr(start, end - start));
        start = end + delimiter.length();
    }
    tokens.push_back(s.substr(start));
    return tokens;
}

cv::Mat slMat2cvMat(sl::Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cvType = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1: cvType = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cvType = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cvType = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cvType = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cvType = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cvType = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cvType = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cvType = CV_8UC4; break;
        default: break;
    }

    // Convert to OpenCV matrix
    return cv::Mat(input.getHeight(), input.getWidth(), cvType, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}