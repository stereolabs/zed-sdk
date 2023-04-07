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

#include "GLViewer.hpp"

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

#include "CostTraversability.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

void parseArgs(int argc, char** argv, InitParameters& param);

#define ENABLE_SPATIAL_MAPPING_VIEW 1

int main(int argc, char** argv) {

    // Terrain mapping parameters
    const float TERRAIN_RANGE = 10; // dimension of the terrain chunk
    // In camera centric mode, the terrain map only occupy 1 chunk
    const float TERRAIN_RES = 0.15; // Terrain mapping resolution
    // Enable traversability cost computation, depends on agent parameters (see below)
    const bool ENABLE_TRAVERSIBILITY = true;

    // Both reference frames are supported
    //const REFERENCE_FRAME terrain_ref = REFERENCE_FRAME::WORLD; // provides a global map
    const REFERENCE_FRAME terrain_ref = REFERENCE_FRAME::CAMERA; // local map, centered on the camera

    
    InitParameters ip;
    ip.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    ip.coordinate_units = UNIT::METER;
    ip.depth_mode = DEPTH_MODE::NEURAL_FAST;
    ip.depth_maximum_distance = TERRAIN_RANGE * 0.75;
    ip.sdk_verbose = 1;
    parseArgs(argc, argv, ip);

    Camera zed;
    if (zed.open(ip) != ERROR_CODE::SUCCESS) {
        std::cout << "error open " << std::endl;
        return -1;
    }


    PositionalTrackingParameters ptp;
    ptp.set_floor_as_origin = true;
    zed.enablePositionalTracking(ptp);

    bool mappingEnable = false;
#if ENABLE_SPATIAL_MAPPING_VIEW
    SpatialMappingParameters smp;
    smp.resolution_meter = TERRAIN_RES * 2.f;
    smp.range_meter = TERRAIN_RANGE * 0.5f;
    smp.use_chunk_only = true;
    smp.map_type = SpatialMappingParameters::SPATIAL_MAP_TYPE::FUSED_POINT_CLOUD;
#endif

    // Create grid map.
    TerrainMappingParameters tmp;
    tmp.setGridResolution(UNIT::METER, TERRAIN_RES);
    tmp.setGridRange(UNIT::METER, TERRAIN_RANGE);
    tmp.setCameraHeightThreshold(UNIT::METER, 1.);

    RuntimeParameters rtp;
    rtp.confidence_threshold = 50;
    rtp.measure3D_reference_frame = terrain_ref;

    Mat imL;
    Terrain myTerrain;

    Mat terrainView;
    TerrainMesh terrainMesh;
    sl::Mat low_res_point_cloud(Resolution(192, 128), sl::MAT_TYPE::F32_C4, sl::MEM::GPU);
    Pose pose;

    // Traversability cost computation
    AgentParameters agent; // Agent traversability capabilities
    agent.radius = 0.25f; // meters
    agent.step_max = 0.1f; // meters
    agent.slope_max = 20; // degrees
    agent.roughness_max = 0.1f;
    TraversabilityParameters traversability_param;
    sl::Terrain traversabilityTerrain;
    sl::Mat terrain_cost_view;

#if ENABLE_SPATIAL_MAPPING_VIEW
    sl::FusedPointCloud pointCloud;
    bool needPcUpdate = true;
#endif

    GLViewer viewer;
    viewer.init(low_res_point_cloud);

    const std::string wndName("ELEVATION");
    cv::namedWindow(wndName, cv::WINDOW_NORMAL);
    cv::setWindowProperty(wndName, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);

    const std::string wndCostName("COST");
    if (ENABLE_TRAVERSIBILITY)
        cv::namedWindow(wndCostName, cv::WINDOW_NORMAL);

    bool drawH = true;
    bool drawMap = true;
    bool pause = false;
    int key = 0;
    int wait_time = 5;
    int f = 0;

    
    POSITIONAL_TRACKING_STATE track_state = POSITIONAL_TRACKING_STATE::OK, track_state_prev = POSITIONAL_TRACKING_STATE::OK;

    // Work with copy of image in a loop.
    while (key != 'q') {
        // Initialize.
        auto state = zed.grab(rtp);
        if (state == ERROR_CODE::SUCCESS) {

            track_state = zed.getPosition(pose);
            if (track_state == POSITIONAL_TRACKING_STATE::OK) {

                if (!mappingEnable) {
#if ENABLE_SPATIAL_MAPPING_VIEW
                    zed.enableSpatialMapping(smp);
#endif
                    std::cout << "Starting Mapping " << std::endl;
                    mappingEnable = true;
                    //zed.enableTerrainMapping(tmp);
                    if (ENABLE_TRAVERSIBILITY)
                        initCostTraversibily(traversabilityTerrain, tmp);
                }

                zed.retrieveImage(imL, VIEW::LEFT, MEM::CPU, Resolution(720, 404));
                cv::Mat cvImage(imL.getHeight(), imL.getWidth(), CV_8UC4, imL.getPtr<sl::uchar1>());
                zed.retrieveMeasure(low_res_point_cloud, MEASURE::XYZBGRA, MEM::GPU, low_res_point_cloud.getResolution());

                cv::imshow("Image", cvImage);

                if (mappingEnable) {
                    //if (zed.retrieveTerrain(myTerrain, terrain_ref) == sl::ERROR_CODE::SUCCESS) 
                    {

                        if (ENABLE_TRAVERSIBILITY) {
                            computeCost(myTerrain, traversabilityTerrain, TERRAIN_RES, agent, traversability_param);
                            normalization(traversabilityTerrain, TRAVERSABILITY_COST /* OCCUPANCY*/ , terrain_cost_view);
                            cv::Mat cvCostView(terrain_cost_view.getHeight(), terrain_cost_view.getWidth(), CV_8UC3, terrain_cost_view.getPtr<sl::uchar1>());
                            cv::imshow(wndCostName, cvCostView);
                        }

                        state = myTerrain.retrieveView(terrainView, MAT_TYPE::U8_C4, drawH ? LayerName::ELEVATION : LayerName::COLOR);
                        if (state == sl::ERROR_CODE::SUCCESS && terrainView.isInit()) {
                            cv::Mat cvTerrain(terrainView.getHeight(), terrainView.getWidth(), CV_8UC4, terrainView.getPtr<sl::uchar1>());
                            cv::imshow(wndName, cvTerrain);
                        }

                        myTerrain.retrieveMesh(terrainMesh, drawH ? LayerName::ELEVATION : LayerName::COLOR, true);
                        viewer.updateMesh(terrainMesh, terrain_ref);
                    }

#if ENABLE_SPATIAL_MAPPING_VIEW
                    if(drawMap) {
                        if (needPcUpdate && (f % 30) == 0) {
                            zed.requestSpatialMapAsync();
                            needPcUpdate = false;
                        }

                        if (!needPcUpdate && (zed.getSpatialMapRequestStatusAsync() == ERROR_CODE::SUCCESS)) {
                            zed.retrieveSpatialMapAsync(pointCloud);
                            viewer.updatePc(pointCloud);
                            needPcUpdate = true;
                        }
                    }
#endif

                    viewer.updateCameraPose(pose.pose_data);
                }
            } else if (track_state_prev != track_state)
                std::cout << "Positional Tracking state: " << track_state << std::endl;
        } else break;

        key = cv::waitKey(wait_time);
        if (key == 'c') drawH = !drawH;
        else if (key == 'd') {
            drawMap = !drawMap;
            if(!drawMap) {
                pointCloud.clear();
                viewer.updatePc(pointCloud);
            }
        } else if (key == ' '){
            pause = !pause;
            if(pause) wait_time = 0;
            else wait_time = 5;
        }

        if (!viewer.isAvailable())
            break;

        track_state_prev = track_state;
    }

    viewer.exit();
    zed.close();

    return 0;
}

void parseArgs(int argc, char** argv, InitParameters& param) {
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        cout << "[Sample] Using SVO File input: " << argv[1] << endl;
    } else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        } else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        } else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }
    } else {
        // Default
    }
}
