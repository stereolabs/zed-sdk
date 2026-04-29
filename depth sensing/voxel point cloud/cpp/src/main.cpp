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

/*********************************************************************
 ** This sample demonstrates voxel-decimated 3D point cloud capture **
 ** with the ZED SDK, with interactive controls for voxel size,     **
 ** resolution mode, and scale factor.                              **
 *********************************************************************/

#include <sl/Camera.hpp>
#include "GLViewer.hpp"

using namespace std;
using namespace sl;

void parseArgs(int argc, char** argv, sl::InitParameters& param);
void printControls();

int main(int argc, char** argv) {
    Camera zed;
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::NEURAL;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_parameters.coordinate_units = UNIT::MILLIMETER;
    init_parameters.sdk_verbose = 1;
    // init_parameters.svo_real_time_mode = true; // enable SVO real-time mode (non-blocking grab with frame drop) for smoother playback
    // when processing can't keep up with the camera FPS init_parameters.depth_maximum_distance = 500000.f; // explicitly set max depth to
    // 50 m
    parseArgs(argc, argv, init_parameters);

    auto returned_state = zed.open(init_parameters);
    if (returned_state > ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    auto camera_config = zed.getCameraInformation().camera_configuration;
    auto stream = zed.getCUDAStream();

    // Point cloud viewer (reuses GLViewer from depth sensing sample)
    GLViewer viewer;
    sl::Resolution res = zed.getRetrieveMeasureResolution();
    sl::Resolution preview_res(640, 360);
    GLenum errgl = viewer.init(argc, argv, camera_config.calibration_parameters.left_cam, stream, res, preview_res);
    if (errgl != GLEW_OK) {
        print("Error OpenGL: " + std::string((char*)glewGetErrorString(errgl)));
        return EXIT_FAILURE;
    }

    // Initialize voxel parameters on the viewer (keyboard-interactive)
    viewer.voxelParams_.voxel_size = 50.f; // 50 mm
    viewer.voxelParams_.centroid = true;
    viewer.voxelParams_.resolution_mode = VOXELIZATION_MODE::STEREO_UNCERTAINTY;
    viewer.voxelParams_.resolution_scale = 0.2f;

    Mat point_cloud, left_image;
    RuntimeParameters run_parameters;
    int svo_position = 0;

    printControls();

    while (viewer.isAvailable()) {
        run_parameters.confidence_threshold = viewer.confidenceThreshold_;

        // SVO seek (arrow keys: +/- 200 frames)
        if (viewer.seekOffset_ != 0) {
            svo_position = std::max(0, svo_position + viewer.seekOffset_);
            zed.setSVOPosition(svo_position);
            viewer.seekOffset_ = 0;
        }

        // SVO pause: re-seek to the same frame before grabbing
        if (viewer.isPaused()) {
            zed.setSVOPosition(svo_position);
        }

        auto grab_status = zed.grab(run_parameters);
        // Loop SVO when end is reached
        if (grab_status == ERROR_CODE::END_OF_SVOFILE_REACHED) {
            zed.setSVOPosition(0);
            svo_position = 0;
            continue;
        }
        if (grab_status <= ERROR_CODE::SUCCESS) {
            if (!viewer.isPaused())
                svo_position = zed.getSVOPosition();

            auto& vp = viewer.voxelParams_;
            if (viewer.useVoxels()) {
                zed.retrieveVoxelMeasure(point_cloud, MEASURE::XYZRGBA, MEM::GPU, vp, stream);
            } else {
                zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA, MEM::GPU, res, stream);
            }
            zed.retrieveImage(left_image, VIEW::LEFT, MEM::GPU, preview_res);
            viewer.updatePointCloud(point_cloud);
            viewer.updateImage(left_image);

            int pt_count = point_cloud.getWidth();
            const char* pause_tag = viewer.isPaused() ? " PAUSED |" : "";
            if (viewer.useVoxels()) {
                std::cout << "\rFPS: " << zed.getCurrentFPS() << " |" << pause_tag << " VOXEL | pts: " << pt_count
                          << " | voxel: " << vp.voxel_size << " mm"
                          << " | mode: " << toString(vp.resolution_mode) << " | scale: " << vp.resolution_scale
                          << " | conf: " << viewer.confidenceThreshold_ << " | " << (vp.centroid ? "centroid" : "grid center") << "        "
                          << std::flush;
            } else {
                std::cout << "\rFPS: " << zed.getCurrentFPS() << " |" << pause_tag << " FULL PC | pts: " << pt_count
                          << " | conf: " << viewer.confidenceThreshold_ << "        " << std::flush;
            }

            if (viewer.shouldSaveData()) {
                sl::Mat pc_save;
                if (viewer.useVoxels())
                    zed.retrieveVoxelMeasure(pc_save, MEASURE::XYZRGBA, MEM::CPU, vp);
                else
                    zed.retrieveMeasure(pc_save, MEASURE::XYZRGBA, MEM::CPU);
                auto err = pc_save.write("VoxelPointcloud.ply");
                if (err <= sl::ERROR_CODE::SUCCESS)
                    std::cout << "\nSaved VoxelPointcloud.ply (" << pc_save.getWidth() << " points)" << std::endl;
                else
                    std::cout << "\nFailed to save .ply file" << std::endl;
            }
        }
    }

    point_cloud.free();
    zed.close();
    return EXIT_SUCCESS;
}

void printControls() {
    std::cout << "\n=== Voxel Point Cloud Viewer ===" << std::endl;
    std::cout << "  +/-     Increase/decrease voxel size" << std::endl;
    std::cout << "  1/2/3   Mode: FIXED / STEREO_UNCERTAINTY / LINEAR" << std::endl;
    std::cout << "  ,/.     Decrease/increase resolution scale" << std::endl;
    std::cout << "  c       Toggle centroid / grid center" << std::endl;
    std::cout << "  p/P     Decrease/increase point size" << std::endl;
    std::cout << "  v       Toggle voxel / full point cloud" << std::endl;
    std::cout << "  Space   Pause / resume (SVO)" << std::endl;
    std::cout << "  Left/Right  Rewind / fast-forward 200 frames (SVO)" << std::endl;
    std::cout << "  s       Save point cloud to .ply" << std::endl;
    std::cout << "  q/Esc   Quit" << std::endl;
    std::cout << "================================\n" << std::endl;
}

void parseArgs(int argc, char** argv, sl::InitParameters& param) {
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        param.input.setFromSVOFile(argv[1]);
        cout << "[Sample] Using SVO File input: " << argv[1] << endl;
    } else if (argc > 1) {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            string ip = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(sl::String(ip.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip << ", port : " << port << endl;
        } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            param.input.setFromStream(sl::String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        }
    }
}
