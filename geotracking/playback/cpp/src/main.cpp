///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2023, STEREOLABS.
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
 ** This sample shows how to use geotracking for global scale             **
 ** localization on real-world map API with pre-recorded GNSS data        **
 **************************************************************************/

#include <iostream>
#include <future>

#include <opencv2/opencv.hpp>

#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>
#include "display/GenericDisplay.h"
#include "exporter/KMLExporter.h"
#include "GNSSReplay.hpp"

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage ./ZED_GNSS_playback <svo-file> <gnss-data-file-path>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string svo_name, gnss_file;
    for(int i = 1; i<3; i++){
        std::string arg(argv[i]);
        if(arg.find(".svo") != std::string::npos)
            svo_name = arg;
        if(arg.find(".json") != std::string::npos)
            gnss_file = arg;
    }
        // Open the camera
    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_params.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE;
    init_params.coordinate_units = sl::UNIT::METER;
    init_params.input.setFromSVOFile(svo_name.c_str());
    sl::ERROR_CODE camera_open_error = zed.open(init_params);
    if (camera_open_error != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "[ZED][ERROR] Can't open ZED camera" << std::endl;
        return EXIT_FAILURE;
    }
    // Enable positional tracking:
    sl::PositionalTrackingParameters pose_tracking_params;
    pose_tracking_params.mode = sl::POSITIONAL_TRACKING_MODE::QUALITY;
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
    init_fusion_param.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE;
    init_fusion_param.coordinate_units = sl::UNIT::METER;
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
    gnss_calibration_parameter.enable_rolling_calibration = false;
    positional_tracking_fusion_parameters.gnss_calibration_parameters = gnss_calibration_parameter;
    positional_tracking_fusion_parameters.enable_GNSS_fusion = true;
    fusion.enablePositionalTracking(positional_tracking_fusion_parameters);

    // Setup viewer:
    GenericDisplay viewer;
    viewer.init(argc, argv);
    std::cout << "Start grabbing data ... the geo-tracking will be displayed in ZEDHub map section" << std::endl;
    sl::Pose zed_pose;

    GNSSReplay gnss_replay(gnss_file);
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
        if(gnss_replay.grab(input_gnss, zed_pose.timestamp.getNanoseconds()) == sl::FUSION_ERROR_CODE::SUCCESS){
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
                std::stringstream ss_translation;
                ss_translation << fused_position.pose_data.getTranslation();
                std::string translation_message = ss_translation.str();
                std::stringstream ss_rotation;
                ss_rotation << fused_position.pose_data.getEulerAngles();
                std::string rotation_message = ss_rotation.str();
                // Display it on OpenGL:
                viewer.updatePoseData(fused_position.pose_data, translation_message, rotation_message, current_state);
            }
            // Get position into the GNSS coordinate system - this needs a initialization between CAMERA 
            // and GNSS. When the initialization is finish the getGeoPose will return sl::POSITIONAL_TRACKING_STATE::OK
            sl::GeoPose current_geopose;
            auto current_geopose_satus = fusion.getGeoPose(current_geopose);
            if (current_geopose_satus == sl::GNSS_CALIBRATION_STATE::CALIBRATED) {
                // Display it on ZED Hub:
                viewer.updateGeoPoseData(current_geopose, zed.getTimestamp(sl::TIME_REFERENCE::CURRENT));

                sl::Transform current_calibration = fusion.getGeoTrackingCalibration();
            } 
            
            {
                // GNSS coordinate system to ZED coordinate system is not initialize yet
                // The initialisation between the coordinates system is basicaly an optimization problem that
                // Try to fit the ZED computed path with the GNSS computed path. In order to do it just move
                // your system by the distance you specified in positional_tracking_fusion_parameters.gnss_initialisation_distance
                float yaw_std;
                sl::float3 position_std;
                fusion.getCurrentGNSSCalibrationSTD(yaw_std, position_std);
                if(yaw_std != -1.f)
                    std::cout << "GNSS State"<<current_geopose_satus<< ": calibration uncertainty yaw_std " << yaw_std << " position_std " << position_std[0] << ", " << position_std[1] << ", " << position_std[2] << "\r";
            }
        }
    }

    closeAllKMLWriter();
    fusion.close();
    zed.close();
    
    return 0;
}
