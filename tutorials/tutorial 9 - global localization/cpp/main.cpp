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
 ** This sample demonstrates how to use previous GNSS recorded data and   **
 ** How to fused it with ZED camera                                       **
 **************************************************************************/

#include <iostream>
#include <future>
#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>

/**
 * @brief Function used for getting GNSS data;
 *
 * @return sl::GNSSData
 */
sl::GNSSData getGNSSData();

int main(int argc, char **argv)
{
    //////////////////////////////
    //                          //
    //      Setup camera        //
    //                          //
    //////////////////////////////
    // Open camera:
    sl::InitParameters init_params;
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_params.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE;
    init_params.coordinate_units = sl::UNIT::METER;
    init_params.camera_resolution = sl::RESOLUTION::AUTO;
    init_params.camera_fps = 60;
    sl::Camera zed;
    sl::ERROR_CODE camera_open_error = zed.open(init_params);
    if (camera_open_error != sl::ERROR_CODE::SUCCESS)
    {
        std::cerr << "[ZED][ERROR] Can't open ZED camera" << std::endl;
        return EXIT_FAILURE;
    }
    // Enable positional tracking:
    sl::PositionalTrackingParameters ptp;
    ptp.initial_world_transform = sl::Transform::identity();
    ptp.enable_imu_fusion = true;     // Enable IMU (for having the gravity direction)
    ptp.set_gravity_as_origin = true; // Set gravity as origin for allowing GNSS to Camera initialization
    auto positional_init = zed.enablePositionalTracking(ptp);
    if (positional_init != sl::ERROR_CODE::SUCCESS)
    {
        std::cerr << "[ZED][ERROR] Can't start tracking of camera" << std::endl;
        return EXIT_FAILURE;
    }
    /// Enable camera publishing for fusion:
    sl::CommunicationParameters communication_parameters;
    communication_parameters.setForSharedMemory();
    zed.startPublishing(communication_parameters);
    /// Run a first grab for starting sending data:
    while (zed.grab() != sl::ERROR_CODE::SUCCESS)
        ;

    //////////////////////////////
    //                          //
    //      Setup Fusion        //
    //                          //
    //////////////////////////////
    // Create fusion object:
    sl::InitFusionParameters init_multi_cam_parameters;
    init_multi_cam_parameters.coordinate_units = sl::UNIT::METER;
    init_multi_cam_parameters.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE;
    init_multi_cam_parameters.output_performance_metrics = true;
    init_multi_cam_parameters.verbose = true;
    sl::Fusion fusion;
    sl::FUSION_ERROR_CODE fusion_init_code = fusion.init(init_multi_cam_parameters);
    if (fusion_init_code != sl::FUSION_ERROR_CODE::SUCCESS)
    {
        std::cerr << "[Fusion][ERROR] Failed to initialize fusion, error: " << fusion_init_code << std::endl;
        return EXIT_FAILURE;
    }
    // Subscribe to camera:
    sl::CameraIdentifier uuid(zed.getCameraInformation().serial_number);
    fusion.subscribe(uuid, communication_parameters, sl::Transform::identity());
    // Enable positional tracking:
    sl::PositionalTrackingFusionParameters ptfp;
    ptfp.enable_GNSS_fusion = true;
    fusion.enablePositionalTracking(ptfp);

    //////////////////////////////
    //                          //
    //      Grab data           //
    //                          //
    //////////////////////////////
    std::cout << "Start grabbing data ... " << std::endl;
    // Setup future callback:
    auto gnss_async = std::async(std::launch::async, getGNSSData);
    unsigned number_detection = 0;
    while (number_detection < 200)
    {
        // Grab camera:
        if (zed.grab() == sl::ERROR_CODE::SUCCESS)
        {
            sl::Pose zed_pose;
            // You can still use the classical getPosition for your application, just not that the position returned by this method
            // is the position without any GNSS/cameras fusion
            zed.getPosition(zed_pose, sl::REFERENCE_FRAME::WORLD);
        }

        // Get GNSS data:
        if (gnss_async.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            sl::GNSSData input_gnss = gnss_async.get();
            // Here we set the GNSS timestamp to the current timestamp of zed camera
            // This is because we use synthetic data and not real one. 
            input_gnss.ts = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);
            if(input_gnss.ts!=sl::Timestamp(0))
                fusion.ingestGNSSData(input_gnss);
            gnss_async = std::async(std::launch::async, getGNSSData);
        }
        // Process fusion
        if (fusion.process() == sl::FUSION_ERROR_CODE::SUCCESS)
        {
            /// Get position:
            sl::GeoPose current_geopose;
            sl::GNSS_FUSION_STATUS current_geopose_status = fusion.getGeoPose(current_geopose);
            /// Get fused position:
            sl::Pose fused_position;
            sl::POSITIONAL_TRACKING_STATE current_state = fusion.getPosition(fused_position);
            std::string translation_message = std::to_string(fused_position.pose_data.getTranslation().tx) + ", " + std::to_string(fused_position.pose_data.getTranslation().ty) + ", " + std::to_string(fused_position.pose_data.getTranslation().tz);
            std::string rotation_message = std::to_string(fused_position.pose_data.getEulerAngles()[0]) + ", " + std::to_string(fused_position.pose_data.getEulerAngles()[1]) + ", " + std::to_string(fused_position.pose_data.getEulerAngles()[2]);
            if (current_state == sl::POSITIONAL_TRACKING_STATE::OK)
            {
                std::cout << "get position translation  = " << translation_message << ", rotation_message = " <<  rotation_message << std::endl;
            }

            // Display it
            if (current_geopose_status == sl::GNSS_FUSION_STATUS::OK)
            {
                number_detection++;
                double latitude, longitude, altitude;
                current_geopose.latlng_coordinates.getCoordinates(latitude, longitude, altitude, false);
                std::cout << "get world map coordinates latitude = " << latitude << ", longitude = " <<  longitude << ", altitude = " << altitude << std::endl;
            }
            else
            {
                // GNSS coordinate system to ZED coordinate system is not initialize yet
                // The initialisation between the coordinates system is basically an optimization problem that
                // Try to fit the ZED computed path with the GNSS computed path. In order to do it just move 
                // your system by the distance you specified in positional_tracking_fusion_parameters.gnss_initialisation_distance
            }
        }
    }
    fusion.close();
    zed.close();
}

sl::GNSSData getGNSSData()
{
    static double x = 0;
    x = x + 0.00000001;
    sl::GNSSData out;
    out.setCoordinates(x, 0, 0);

    // N.B. For illustrate how to use Global Localization API we generated "fake" GNSS data.
    // If you use a real GNSS sensor you must provide all GNSSData:
    // coordinates, ts, position_covariance, latitude_std, longitude_std, altitude_std
    return out;
}
