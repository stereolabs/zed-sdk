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

/************************************************************
** This sample demonstrates how to read a SVO video file. **
** We use OpenCV to display the video.					   **
*************************************************************/

// ZED include
#include <sl/Camera.hpp>

// Sample includes
#include <opencv2/opencv.hpp>
#include "utils.hpp"

// Using namespace
using namespace sl;
using namespace std;

int main(int argc, char **argv)
{

    if (argc <= 1)
    {
        cout << "Usage: \n";
        cout << "$ ZED_SVO_Playback <SVO_file> ...\n";
        cout << "  ** SVO file is mandatory in the application ** \n\n";
        return EXIT_FAILURE;
    }

    std::map<int, std::string> svo_files;
    std::vector<std::shared_ptr<Camera>> cameras;
    for (int i = 1; i < argc; ++i)
    {
        auto svo_path = std::string(argv[i]);
        if (svo_path.find(".svo") == std::string::npos)
        {
            std::cerr << "Path " << svo_path << " is not a valid SVO path, skipping..." << std::endl;
            continue;
        }

        // Create ZED objects
        auto zed = std::make_shared<Camera>();
        InitParameters init_parameters;
        init_parameters.svo_real_time_mode = true;
        init_parameters.input.setFromSVOFile(argv[i]);
        init_parameters.depth_mode = sl::DEPTH_MODE::NONE;

        // Open the camera
        auto returned_state = zed->open(init_parameters);
        if (returned_state != ERROR_CODE::SUCCESS)
        {
            std::cerr << "Unable to open camera" << i << returned_state << ". Exit program." << std::endl;
            return EXIT_FAILURE;
        }

        cameras.push_back(zed);
        svo_files.insert(std::make_pair(i - 1, init_parameters.input.getConfiguration()));
    }

    if (svo_files.empty())
    {
        std::cerr << "No SVO files opened, exiting." << std::endl;
        return EXIT_FAILURE;
    }

    bool enable_svo_sync = (svo_files.size() > 1);
    if (enable_svo_sync)
    {
        std::cout << "Starting SVO sync process..." << std::endl;
        std::map<int, int> cam_idx_to_svo_frame_idx = syncDATA(svo_files);

        for (auto &it: cam_idx_to_svo_frame_idx)
        {
            std::cout << "Setting camera " << it.first << " to frame " << it.second << std::endl;
            cameras[it.first]->setSVOPosition(it.second);
        }
    }

    // Setup key, images, times
    char key = ' ';
    cout << " Press 'q' to exit..." << endl;

    // Start SVO playback
    sl::Mat mat;
    sl::ERROR_CODE returned_state;
    while (key != 'q')
    {
        for (auto i = 0; i < cameras.size(); ++i)
        {
            auto zed = cameras[i];
            auto resolution = zed->getCameraInformation().camera_configuration.resolution;
            // Define OpenCV window size (resize to max 720/404)
            sl::Resolution low_resolution(min(720, (int) resolution.width) * 2, min(404, (int) resolution.height));

            returned_state = zed->grab();
            if (returned_state == ERROR_CODE::SUCCESS)
            {
                // Get the side by side image
                zed->retrieveImage(mat, VIEW::SIDE_BY_SIDE, MEM::CPU, low_resolution);
                cv::Mat cv_mat = slMat2cvMat(mat);

                // Display the frame
                cv::imshow("View " + std::to_string(zed->getCameraInformation().serial_number), cv_mat);
            } else if (returned_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
            {
                std::cout <<"SVO end has been reached. Closing camera." << std::endl;
                cv::destroyAllWindows();
                zed->close();
                cameras.erase(cameras.begin() + i);
            } else
            {
                std::cout << "Grab ZED : " << returned_state << std::endl;
                break;
            }
        }
        key = cv::waitKey(1);
    }

    for (auto &zed: cameras)
        zed->close();

    return EXIT_SUCCESS;
}