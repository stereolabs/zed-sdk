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
 ** with ZED camera                                                       **
 **************************************************************************/

#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>

#include "display/GenericDisplay.h"
#include "gnss_reader/IGNSSReader.h"
#include "gnss_reader/GPSDReader.hpp"

int main(int argc, char **argv)
{
    // Open the camera
    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.sdk_verbose = 1;
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
    // Enable GNSS data producing:
    GPSDReader gnss_reader;
    gnss_reader.initialize();

    // Subscribe to Odometry
    sl::CameraIdentifier uuid(zed.getCameraInformation().serial_number);
    fusion.subscribe(uuid);
    // Enable positional tracking for Fusion object
    sl::PositionalTrackingFusionParameters positional_tracking_fusion_parameters;
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
    while (viewer.isAvailable())
    {
        // Grab camera:
        if (zed.grab() == sl::ERROR_CODE::SUCCESS)
        {
            sl::Pose zed_pose;
            // You can still use the classical getPosition for your application, just not that the position returned by this method
            // is the position without any GNSS/cameras fusion
            zed.getPosition(zed_pose, sl::REFERENCE_FRAME::CAMERA);
        }
        // Get GNSS data:
        sl::GNSSData input_gnss;
        if (gnss_reader.grab(input_gnss) == sl::ERROR_CODE::SUCCESS)
        {
            // Display it on the Live Server:
            viewer.updateRawGeoPoseData(input_gnss);

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

            sl::FusedPositionalTrackingStatus fused_status = fusion.getFusedPositionalTrackingStatus();
         
            // Display it on OpenGL:
            viewer.updatePoseData(fused_position.pose_data, fused_status);
            
            // Get position into the GNSS coordinate system - this needs a initialization between CAMERA 
            // and GNSS. When the initialization is finish the getGeoPose will return sl::POSITIONAL_TRACKING_STATE::OK
            sl::GeoPose current_geopose;
            auto current_geopose_satus = fusion.getGeoPose(current_geopose);
            if (current_geopose_satus == sl::GNSS_FUSION_STATUS::OK)
            {
                // Display it on the Live Server:
                viewer.updateGeoPoseData(current_geopose);
            }
            else
            {
                // GNSS coordinate system to ZED coordinate system is not initialize yet
                // The initialization between the coordinates system is an optimization problem that
                // Try to fit the ZED computed path with the GNSS computed path. In order to do it just move
                // your system by the distance you specified in positional_tracking_fusion_parameters.gnss_initialisation_distance
            }
        }
    }
    fusion.close();
    zed.close();

    return EXIT_SUCCESS;
}
