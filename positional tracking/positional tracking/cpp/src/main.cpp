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

/**********************************************************************************
 ** This sample demonstrates how to capture a live 3D reconstruction of a scene  **
 ** as a fused point cloud and display the result in an OpenGL window.           **
 **********************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"

#include <opencv2/opencv.hpp>

// Using std and sl namespaces
using namespace std;
using namespace sl;

void parse_args(int argc, char **argv,InitParameters& param, sl::Mat &roi);

void print(std::string msg_prefix, sl::ERROR_CODE err_code = sl::ERROR_CODE::SUCCESS, std::string msg_suffix = "");

int main(int argc, char **argv) {

    Camera zed;
    // Set configuration parameters for the ZED
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::NEURAL;
    init_parameters.coordinate_units = UNIT::METER;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed

    sl::Mat roi;
    parse_args(argc, argv, init_parameters, roi);

    // Open the camera
    auto returned_state = zed.open(init_parameters);

    if (returned_state > ERROR_CODE::SUCCESS) {// Quit if an error occurred
        print("Open Camera", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    if(roi.isInit()){
        auto state = zed.setRegionOfInterest(roi, {sl::MODULE::POSITIONAL_TRACKING});
        std::cout<<"Applied ROI "<<state<<"\n";
    }else{
        // If the region of interest is not loaded from a file, the auto detection can be enabled
        if(0){
            sl::RegionOfInterestParameters roi_param;
            roi_param.auto_apply_module = {sl::MODULE::DEPTH, sl::MODULE::POSITIONAL_TRACKING};
            zed.startRegionOfInterestAutoDetection(roi_param);
            print("Region Of Interest auto detection is running.");
        }
    }

    REGION_OF_INTEREST_AUTO_DETECTION_STATE roi_state = REGION_OF_INTEREST_AUTO_DETECTION_STATE::NOT_ENABLED;

    /* Print shortcuts*/
    std::cout<<"Shortcuts\n";
    std::cout<<"\t- 'l' to enable/disable current live point cloud display\n";
    std::cout<<"\t- 'm' to enable/disable landmark display\n";
    std::cout<<"\t- 'd' to switch background color from dark to light\n";
    std::cout<<"\t- 'f' to follow the camera\n";
    std::cout<<"\t- 'Shift' for soft mouse control / 'alt' for regular control / 'Ctrl' for strong control \n";
    std::cout<<"\t- 'space' to switch camera view\n";

    auto camera_infos = zed.getCameraInformation();

    // Setup and start positional tracking
    Pose pose;
    POSITIONAL_TRACKING_STATE tracking_state = POSITIONAL_TRACKING_STATE::OFF;

    sl::PositionalTrackingParameters ptp;
    ptp.mode = sl::POSITIONAL_TRACKING_MODE::GEN_3;
    returned_state = zed.enablePositionalTracking(ptp);
    if (returned_state > ERROR_CODE::SUCCESS) {
        print("Enabling positional tracking failed: ", returned_state);
        zed.close();
        return EXIT_FAILURE;
    }

    // Setup runtime parameters
    RuntimeParameters runtime_parameters;
    // Use low depth confidence to avoid introducing noise in the constructed model
    runtime_parameters.confidence_threshold = 30;

    auto resolution = camera_infos.camera_configuration.resolution;

    // Define display resolution and check that it fit at least the image resolution
    
    sl::Resolution display_resolution = zed.getRetrieveMeasureResolution();

    Mat image(display_resolution, MAT_TYPE::U8_C4, sl::MEM::GPU);
    Mat point_cloud(display_resolution, MAT_TYPE::F32_C4, sl::MEM::GPU);
    
    // Point cloud viewer
    GLViewer viewer;

    viewer.init(argc, argv, image, point_cloud, zed.getCUDAStream());

    std::map<uint64_t, sl::Landmark> map_lm;
    std::vector<sl::float3> map_lm_tracked;
    std::vector<sl::Landmark2D> map_lm2d;
    auto last_lm_update = sl::getCurrentTimeStamp().getSeconds();

    // Start the main loop
    while (viewer.isAvailable()) {
        // Grab a new image
        sl::ERROR_CODE grab_result = zed.grab(runtime_parameters);

        switch (grab_result) {
            case sl::ERROR_CODE::SUCCESS:
            case sl::ERROR_CODE::CORRUPTED_FRAME:
                // Retrieve the left image
                zed.retrieveImage(image, VIEW::LEFT, MEM::GPU, display_resolution);
                zed.retrieveMeasure(point_cloud, MEASURE::XYZBGRA, MEM::GPU, display_resolution);
                // Retrieve the camera pose data
                zed.getPosition(pose);

                viewer.updateCameraPose(pose.pose_data, zed.getPositionalTrackingStatus());

                if(sl::getCurrentTimeStamp().getSeconds() - last_lm_update > 1) {
                    zed.getPositionalTrackingLandmarks(map_lm);
                    viewer.pushLM(map_lm);
                    last_lm_update = sl::getCurrentTimeStamp().getSeconds();
                }

                if(map_lm.size()){
                    zed.getPositionalTrackingLandmarks2D(map_lm2d);
                    map_lm_tracked.clear();
                    for(auto &it: map_lm2d){
                        if(map_lm.find(it.id) != map_lm.end())
                            map_lm_tracked.push_back(map_lm[it.id].position);
                    }
                    if(map_lm_tracked.size()) viewer.pushTrackedLM(map_lm_tracked);
                }

                // If the region of interest auto detection is running, the resulting mask can be saved and reloaded for later use
                if(roi_state == sl::REGION_OF_INTEREST_AUTO_DETECTION_STATE::RUNNING &&
                        zed.getRegionOfInterestAutoDetectionStatus() == sl::REGION_OF_INTEREST_AUTO_DETECTION_STATE::READY) {
                    sl::String roi_name = "roi_mask.jpg";
                    std::cout << "Region Of Interest detection done! Saving into " << roi_name << std::endl;
                    zed.getRegionOfInterest(roi, sl::Resolution(0,0), sl::MODULE::POSITIONAL_TRACKING);
                    roi.write(roi_name);
                }
                roi_state = zed.getRegionOfInterestAutoDetectionStatus();
                break;
            default:
                break;
        }
    }

    // Free allocated memory before closing the camera
    image.free();
    point_cloud.free();
    // Close the ZED
    zed.close();

    return 0;
}

void parse_args(int argc, char **argv,InitParameters& param, sl::Mat &roi)
{
    if(argc == 1) return;
    for(int id = 1; id < argc; id ++) {
        std::string arg(argv[id]);
        if(arg.find(".svo")!=string::npos) {
            // SVO input mode
            param.input.setFromSVOFile(arg.c_str());
            cout<<"[Sample] Using SVO File input: "<<arg<<endl;
        }

        unsigned int a,b,c,d,port;
        if (sscanf(arg.c_str(),"%u.%u.%u.%u:%d", &a, &b, &c, &d,&port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a)+"."+to_string(b)+"."+to_string(c)+"."+to_string(d);
            param.input.setFromStream(String(ip_adress.c_str()),port);
            cout<<"[Sample] Using Stream input, IP : "<<ip_adress<<", port : "<<port<<endl;
        }
        else  if (sscanf(arg.c_str(),"%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(String(argv[1]));
            cout<<"[Sample] Using Stream input, IP : "<<argv[1]<<endl;
        }
        else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        }else if (arg.find("HD1200") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1200;
            cout << "[Sample] Using Camera in resolution HD1200" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        }else if (arg.find("SVGA") != string::npos) {
            param.camera_resolution = RESOLUTION::SVGA;
            cout << "[Sample] Using Camera in resolution SVGA" << endl;
        }else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }else if ((arg.find(".png") != string::npos) || ((arg.find(".jpg") != string::npos))) {
            roi.read(arg.c_str());
            cout << "[Sample] Using Region of intererest from "<< arg << endl;
        }
    }
}

void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    cout <<"[Sample]";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout<<" ";
    cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}
