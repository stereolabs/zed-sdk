///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, STEREOLABS.
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

/***********************************************************************
** This sample demonstrates how to read a SVO file 		    		  **
** and convert it into an AVI file (LEFT + RIGHT) or (LEFT + DEPTH)   **
************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include <opencv2/opencv.hpp>
#include "utils.hpp"

// Using namespace
using namespace sl;
using namespace std;

enum APP_TYPE {
    LEFT_AND_RIGHT,
    LEFT_AND_DEPTH
};

int main(int argc, char **argv) {

    if ((argc != 3) && (argc != 4)) {
        cout << "Please specify:\n";
        cout << "- SVO file path (input)\n";
        cout << "- AVI file path (output)\n";
        cout << "- (optional) (bool) Export LEFT+DEPTH image instead of LEFT+RIGHT image.\n";
        return 1;
    }

    string svo_input_path(argv[1]);
    string avi_output_path(argv[2]);
    APP_TYPE app_type = argc == 4 ? LEFT_AND_DEPTH : LEFT_AND_RIGHT;

    // Create ZED objects
    Camera zed;

    // Specify SVO path parameter
    InitParameters initParameters;
    initParameters.svo_input_filename.set(svo_input_path.c_str());

    // Open the SVO file specified as a parameter
    ERROR_CODE err = zed.open(initParameters);
    if (err != SUCCESS) {
        cout << errorCode2str(err) << endl;
        zed.close();
        return 1; // Quit if an error occurred
    }

    // Get image size
    Resolution image_size = zed.getResolution();
    int width = image_size.width;
    int height = image_size.height;
    int width_sbs = image_size.width * 2;

    // Prepare side by side image containers
    cv::Size image_size_sbs(width_sbs, height); // Size of the side by side image
    cv::Mat svo_image_sbs_rgba(image_size_sbs, CV_8UC4); // Container for ZED RGBA side by side image
    cv::Mat ocv_image_sbs_rgb(image_size_sbs, CV_8UC3); // Container for OpenCV RGB side by side image

    Mat left_image(width, height, MAT_TYPE_8U_C4);
    cv::Mat left_image_ocv = slMat2cvMat(left_image);

    Mat  right_image(width, height, MAT_TYPE_8U_C4);
    cv::Mat right_image_ocv = slMat2cvMat(right_image);

    // Create video writer
    int fourcc = CV_FOURCC('M', '4', 'S', '2'); // MPEG-4 part 2 codec
    int frame_rate = fmax(zed.getCameraFPS(), 25); // Minimum write rate in OpenCV is 25
    cv::VideoWriter video_writer(avi_output_path, fourcc, frame_rate, image_size_sbs);

    if (!video_writer.isOpened()) {
        cout << "OpenCV video writer cannot be opened. Please check the .avi file path and write permissions." << endl;
        zed.close();
        return 1;
    }

    RuntimeParameters rt_param;
    rt_param.sensing_mode = SENSING_MODE_FILL;

    // Start SVO conversion to AVI
    cout << "Converting SVO... Use Ctrl-C to interrupt conversion." << endl;

    int nb_frames = zed.getSVONumberOfFrames();
    int svo_position = 0;

    SetCtrlHandler();

    while (!exit_app) {
        if (zed.grab(rt_param) == SUCCESS) {
            svo_position = zed.getSVOPosition();

            // Retrieve SVO images
            zed.retrieveImage(left_image, VIEW_LEFT);

            switch (app_type) {
                case LEFT_AND_RIGHT:
                zed.retrieveImage(right_image, VIEW_RIGHT);
                break;
                case  LEFT_AND_DEPTH:
                zed.retrieveImage(right_image, VIEW_DEPTH);
                break;
                default:
                break;
            }
            // Copy the left image to the left side of SBS image
            left_image_ocv.copyTo(svo_image_sbs_rgba(cv::Rect(0, 0, width, height)));

            // Copy the right image to the right side of SBS image
            right_image_ocv.copyTo(svo_image_sbs_rgba(cv::Rect(width, 0, width, height)));

            // Convert SVO image from RGBA to RGB
            cv::cvtColor(svo_image_sbs_rgba, ocv_image_sbs_rgb, CV_RGBA2RGB);

            // Write the RGB image
            video_writer.write(ocv_image_sbs_rgb);

            // Display progress
            ProgressBar((float) (svo_position / (float) nb_frames), 30);

            // Check if we have reached the end of the video
            if (svo_position >= (nb_frames - 1)) { // End of SVO
                cout << "\nSVO end has been reached. Exiting now.\n";
                exit_app = true;
            }
        }
    }
    // Close the video writer
    video_writer.release();

    zed.close();
    return 0;
}

