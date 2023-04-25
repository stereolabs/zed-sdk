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
#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>
#include "display/GenericDisplay.h"
#include "json.hpp"

using json = nlohmann::json;

/**
 * @brief Function used for getting GNSS data;
 *
 * @return sl::GNSSData
 */
bool getNextGNSSData(std::string gnss_file_path, uint64_t current_timestamp, sl::GNSSData & out);

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage ./ZED_GNSS_playback <svo-file> <gnss-data-file-path>" << std::endl;
        return EXIT_FAILURE;
    }
    // Open the camera
    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.sdk_verbose = 1;
    init_params.input.setFromSVOFile(argv[1]);
    sl::ERROR_CODE camera_open_error = zed.open(init_params);
    if (camera_open_error != sl::ERROR_CODE::SUCCESS)
    {
        std::cerr << "[ZED][ERROR] Can't open ZED camera" << std::endl;
        return EXIT_FAILURE;
    }
    // Enable positional tracking:
    auto positional_init = zed.enablePositionalTracking();
    if (positional_init != sl::ERROR_CODE::SUCCESS)
    {
        std::cerr << "[ZED][ERROR] Can't start tracking of camera" << std::endl;
        return EXIT_FAILURE;
    }

    // Create Fusion object:
    sl::Fusion fusion;
    sl::InitFusionParameters init_fusion_param;
    init_fusion_param.coordinate_units = sl::UNIT::METER;
    sl::FUSION_ERROR_CODE fusion_init_code = fusion.init(init_fusion_param);
    if (fusion_init_code != sl::FUSION_ERROR_CODE::SUCCESS)
    {
        std::cerr << "[Fusion][ERROR] Failed to initialize fusion, error: " << fusion_init_code << std::endl;
        return EXIT_FAILURE;
    }

    // Enable odometry publishing:
    zed.startPublishing();
    /// Run a first grab for starting sending data:
    while (zed.grab() != sl::ERROR_CODE::SUCCESS)
    {
    }

    // Subscribe to Odometry
    sl::CameraIdentifier uuid(zed.getCameraInformation().serial_number);
    fusion.subscribe(uuid);
    // Enable positional tracking for Fusion object
    sl::PositionalTrackingFusionParameters positional_tracking_fusion_parameters;
    positional_tracking_fusion_parameters.enable_GNSS_fusion = true;
    fusion.enablePositionalTracking(positional_tracking_fusion_parameters);

    // Setup viewer:
    GenericDisplay viewer;
    viewer.init(argc, argv);
    std::cout << "Start grabbing data ... the geo-tracking will be displayed in ZEDHub map section" << std::endl;
    sl::Pose zed_pose;
    while (viewer.isAvailable())
    {
        // Grab camera:
        if (zed.grab() == sl::ERROR_CODE::SUCCESS)
        {
            // You can still use the classical getPosition for your application, just not that the position returned by this method
            // is the position without any GNSS/cameras fusion
            zed.getPosition(zed_pose, sl::REFERENCE_FRAME::WORLD);
        }

        // Get GNSS data:
        sl::GNSSData input_gnss;
        if(getNextGNSSData(argv[2], zed_pose.timestamp.getNanoseconds(), input_gnss)){
            // Publish GNSS data to Fusion
            auto ingest_error = fusion.ingestGNSSData(input_gnss);
            if(ingest_error != sl::FUSION_ERROR_CODE::SUCCESS){
                std::cout << "Ingest error occurred when ingesting GNSSData: " << ingest_error << std::endl;
            }
        }   
        
            
        // Process data and compute positions:
        if (fusion.process() == sl::FUSION_ERROR_CODE::SUCCESS)
        {
            sl::Pose fused_position;
            // Get position into the ZED CAMERA coordinate system:
            sl::POSITIONAL_TRACKING_STATE current_state = fusion.getPosition(fused_position);
            if (current_state == sl::POSITIONAL_TRACKING_STATE::OK)
            {
                std::stringstream ss;
                ss << fused_position.pose_data.getTranslation();
                std::string translation_message = ss.str();
                ss.clear();
                ss << fused_position.pose_data.getEulerAngles();
                std::string rotation_message = ss.str();
                // Display it on OpenGL:
                viewer.updatePoseData(fused_position.pose_data, translation_message, rotation_message, current_state);
            }
            // Get position into the GNSS coordinate system - this needs a initialization between CAMERA 
            // and GNSS. When the initialization is finish the getGeoPose will return sl::POSITIONAL_TRACKING_STATE::OK
            sl::GeoPose current_geopose;
            sl::POSITIONAL_TRACKING_STATE current_geopose_satus = fusion.getGeoPose(current_geopose);
            if (current_geopose_satus == sl::POSITIONAL_TRACKING_STATE::OK)
            {
                // Display it on ZED Hub:
                viewer.updateGeoPoseData(current_geopose, zed.getTimestamp(sl::TIME_REFERENCE::CURRENT));
            }
            else
            {
                // GNSS coordinate system to ZED coordinate system is not initialize yet
                // The initialisation between the coordinates system is basicaly an optimization problem that
                // Try to fit the ZED computed path with the GNSS computed path. In order to do it just move
                // your system by the distance you specified in positional_tracking_fusion_parameters.gnss_initialisation_distance
            }
        }
    }
    fusion.close();
    zed.close();
    return 0;
}

json readGNSSJsonFile(std::string gnss_file_path)
{

    std::ifstream gnss_file_data;
    gnss_file_data.open(gnss_file_path);
    if (!gnss_file_data.is_open())
    {
        std::cerr << "Unable to open " << gnss_file_path << std::endl;
        exit(EXIT_FAILURE);
    }
    json out = json::parse(gnss_file_data);
    return out;
}

bool getNextGNSSData(std::string gnss_file_path, uint64_t current_timestamp, sl::GNSSData & out)
{
    static json gnss_data;
    static unsigned current_gnss_idx = 0;
    if (current_gnss_idx == 0)
        gnss_data = readGNSSJsonFile(gnss_file_path);
    if (current_gnss_idx < gnss_data["GNSS"].size())
    {
        json current_gnss_data_json = gnss_data["GNSS"][current_gnss_idx];
        // Check inputs:
        if(current_gnss_data_json["coordinates"].is_null())
            return false;
        if(current_gnss_data_json["coordinates"]["latitude"].is_null())
            return false;
        if(current_gnss_data_json["coordinates"]["longitude"].is_null())
            return false;
        if(current_gnss_data_json["coordinates"]["altitude"].is_null())
            return false;
        if(current_gnss_data_json["coordinates"]["longitude_std"].is_null())
            return false;
        if(current_gnss_data_json["coordinates"]["latitude_std"].is_null())
            return false;
        if(current_gnss_data_json["coordinates"]["altitude_std"].is_null())
            return false;
        if(current_gnss_data_json["ts"].is_null())
            return false;
        for (unsigned i = 0; i < 9; i++)
            if(current_gnss_data_json["position_covariance"][i].is_null())
                return false;


        sl::GNSSData current_gnss_data;
        // Fill out coordinates:
        current_gnss_data.setCoordinates(current_gnss_data_json["coordinates"]["latitude"].get<float>(), current_gnss_data_json["coordinates"]["longitude"].get<float>(), current_gnss_data_json["coordinates"]["altitude"].get<float>(), false);
        // Fill out default standard deviation:
        current_gnss_data.longitude_std = current_gnss_data_json["longitude_std"];
        current_gnss_data.latitude_std = current_gnss_data_json["latitude_std"];
        current_gnss_data.altitude_std = current_gnss_data_json["altitude_std"];
        // Fill out covariance [must be not null]
        std::array<double, 9> position_covariance;
        for (unsigned i = 0; i < 9; i++)
            position_covariance[i] = current_gnss_data_json["position_covariance"][i].get<float>();
        current_gnss_data.position_covariance = position_covariance;
        // Fill out timestamp:
        current_gnss_data.ts.setNanoseconds(current_gnss_data_json["ts"].get<uint64_t>());

        // Verify that the current GNSS data and camera timestamp are within a reasonable time offset of each other to ensure accurate alignment of spatial and visual data:
        int64_t current_zed_timestamp = current_timestamp;
        sl::Timestamp timestamp_difference(abs(current_zed_timestamp - (int64_t)current_gnss_data_json["ts"].get<uint64_t>()));
        if(timestamp_difference.getSeconds() > 10)
            return false;


        current_gnss_idx++;
        out =  current_gnss_data;
        return true;
    }
    return false;
}
