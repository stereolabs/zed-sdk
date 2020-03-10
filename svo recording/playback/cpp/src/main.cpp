///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
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

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");

int main(int argc, char **argv) {

    if (argc<=1)  {
        cout << "Usage: \n";
        cout << "$ ZED_SVO_Playback <SVO_file> \n";
        cout << "  ** SVO file is mandatory in the application ** \n\n";
        return EXIT_FAILURE;
    }

    // Create ZED objects
    Camera zed;
    InitParameters init_parameters;
    init_parameters.input.setFromSVOFile(argv[1]);
	init_parameters.camera_disable_self_calib = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE;

    // Open the camera
    ERROR_CODE zed_open_state = zed.open(init_parameters);
    if (zed_open_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", zed_open_state, "Exit program.");
        return EXIT_FAILURE;
    }

    auto resolution = zed.getCameraInformation().camera_configuration.resolution;
    cv::Size size(resolution.width, resolution.height);
    cv::Size size_sbs(size.width * 2, size.height);

    // Define OpenCV window size (resize to max 720/404)
    int width = min(720, size.width);
    int height = min(404, size.height);
    Mat svo_image(width * 2, height, MAT_TYPE::U8_C4, MEM::CPU);
    cv::Mat svo_image_ocv = slMat2cvMat(svo_image);

    // Setup key, images, times
    char key = ' ';
    cout << " Press 's' to save SVO image as a PNG" << endl;
    cout << " Press 'f' to jump forward in the video" << endl;
    cout << " Press 'b' to jump backward in the video" << endl;
    cout << " Press 'q' to exit..." << endl;

    int svo_frame_rate = zed.getInitParameters().camera_fps;
    int nb_frames = zed.getSVONumberOfFrames();
    print("[Info] SVO contains " +to_string(nb_frames)+" frames");

    // Start SVO playback

	zed.setSVOPosition(0);

     while (key != 'q') {
        sl::ERROR_CODE err = zed.grab();
        if (err == ERROR_CODE::SUCCESS) {

            // Get the side by side image
            zed.retrieveImage(svo_image, VIEW::SIDE_BY_SIDE, MEM::CPU, sl::Resolution(2*width, height));
            int svo_position = zed.getSVOPosition();

            // Display the frame
            cv::imshow("View", svo_image_ocv);
            key = cv::waitKey(10);
            
            switch (key) {
            case 's':
                svo_image.write(("capture_" + to_string(svo_position) + ".png").c_str());
                break;
            case 'f':
                zed.setSVOPosition(svo_position + svo_frame_rate);
                break;
            case 'b':
                zed.setSVOPosition(svo_position - svo_frame_rate);
                break;
            }

            ProgressBar((float)(svo_position / (float)nb_frames), 30);
        }
        else if (err == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
        {
            print("SVO end has been reached. Looping back to 0\n");
            zed.setSVOPosition(0);
        }
        else {
            print("Grab ZED : ",err);
            break;
        }
     } 
    zed.close();
    return EXIT_SUCCESS;
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout <<"[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout<<" ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}
