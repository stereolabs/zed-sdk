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
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "utils.hpp"

// Using namespace
using namespace sl;
using namespace std;

enum APP_TYPE {
    LEFT_AND_RIGHT,
    LEFT_AND_DEPTH,
    LEFT_AND_DEPTH_16
};

int main(int argc, char **argv) {
 
    if (argc != 4 ) {
        cout << "Usage: \n\n";
        cout << "    ZED_SVO_Export A B C \n\n";
        cout << "Please use the following parameters from the command line:\n";
        cout << " A - SVO file path (input) : \"path/to/file.svo\"\n";
        cout << " B - AVI file path (output) or image sequence folder(output) : \"path/to/output/file.avi\" or \"path/to/output/folder\"\n";
        cout << " C - Export mode:  0=Export LEFT+RIGHT AVI.\n";
        cout << "                   1=Export LEFT+DEPTH_VIEW AVI.\n";
        cout << "                   2=Export LEFT+RIGHT image sequence.\n";
        cout << "                   3=Export LEFT+DEPTH_VIEW image sequence.\n";
        cout << "                   4=Export LEFT+DEPTH_16Bit image sequence.\n";
        cout << " A and B need to end with '/' or '\\'\n\n";
        cout << "Examples: \n";
        cout << "  (AVI LEFT+RIGHT)   ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/file.avi\" 0\n";
        cout << "  (AVI LEFT+DEPTH)   ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/file.avi\" 1\n";
        cout << "  (SEQUENCE LEFT+RIGHT)   ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/folder\" 2\n";
        cout << "  (SEQUENCE LEFT+DEPTH)   ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/folder\" 3\n";
        cout << "  (SEQUENCE LEFT+DEPTH_16Bit)   ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/folder\" 4\n";
        cout << "\nPress [Enter] to continue";
        cin.ignore();
        return 1;
    }

    // Get input parameters
    string svo_input_path(argv[1]);
    string output_path(argv[2]);
    bool output_as_video = true;
    APP_TYPE app_type = LEFT_AND_RIGHT;
    if (!strcmp(argv[3],"1") || !strcmp(argv[3], "3"))
        app_type = LEFT_AND_DEPTH;
    if (!strcmp(argv[3], "4"))
        app_type = LEFT_AND_DEPTH_16;

    // Check if exporting to AVI or SEQUENCE
    if (strcmp(argv[3], "0") && strcmp(argv[3], "1"))
        output_as_video = false;

    if (!output_as_video && !directoryExists(output_path)) {
        cout << "Input directory doesn't exist. Check permissions or create it.\n" << output_path << "\n";
        return 1;
    }

    if(!output_as_video && output_path.back() != '/' && output_path.back() != '\\') {
        cout << "Output folder needs to end with '/' or '\\'.\n" << output_path << "\n";
        return 1;
    }

    // Create ZED objects
    Camera zed;

    // Specify SVO path parameter
    InitParameters initParameters;
    initParameters.svo_input_filename.set(svo_input_path.c_str());
    initParameters.coordinate_units = UNIT_MILLIMETER;

    // Open the SVO file specified as a parameter
    ERROR_CODE err = zed.open(initParameters);
    if (err != SUCCESS) {
        cout << toString(err) << endl;
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

    Mat  depth_image(width, height, MAT_TYPE_32F_C1);
    cv::Mat depth_image_ocv = slMat2cvMat(depth_image);

    // Create video writer
    cv::VideoWriter* video_writer;
    if (output_as_video) {
        int fourcc = CV_FOURCC('M', '4', 'S', '2'); // MPEG-4 part 2 codec
        int frame_rate = fmax(zed.getCameraFPS(), 25); // Minimum write rate in OpenCV is 25
        video_writer = new cv::VideoWriter(output_path, fourcc, frame_rate, image_size_sbs);
        if (!video_writer->isOpened()) {
            cout << "OpenCV video writer cannot be opened. Please check the .avi file path and write permissions." << endl;
            zed.close();
            return 1;
        }
    }

    RuntimeParameters rt_param;
    rt_param.sensing_mode = SENSING_MODE_FILL;

    // Start SVO conversion to AVI/SEQUENCE
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
            case  LEFT_AND_DEPTH_16:
                zed.retrieveMeasure(depth_image, MEASURE_DEPTH);
                break;
            default:
                break;
            }

            if (output_as_video) {
                // Copy the left image to the left side of SBS image
                left_image_ocv.copyTo(svo_image_sbs_rgba(cv::Rect(0, 0, width, height)));

                // Copy the right image to the right side of SBS image
                right_image_ocv.copyTo(svo_image_sbs_rgba(cv::Rect(width, 0, width, height)));

                // Convert SVO image from RGBA to RGB
                cv::cvtColor(svo_image_sbs_rgba, ocv_image_sbs_rgb, CV_RGBA2RGB);

                // Write the RGB image in the video
                video_writer->write(ocv_image_sbs_rgb);
            }
            else {
                // Generate filenames
                ostringstream filename1;
                filename1 << output_path << "/left" << setfill('0') << setw(6) << svo_position << ".png";
                ostringstream filename2;
                filename2 << output_path << (app_type==LEFT_AND_RIGHT?"/right":"/depth") << setfill('0') << setw(6) << svo_position << ".png";
                
                // Save Left images
                cv::imwrite(filename1.str(), left_image_ocv);

                // Save depth
                if(app_type != LEFT_AND_DEPTH_16)
                    cv::imwrite(filename2.str(), right_image_ocv);
                else {
                    // Convert to 16Bit
                    cv::Mat depth16;
                    depth_image_ocv.convertTo(depth16, CV_16UC1);
                    cv::imwrite(filename2.str(), depth16);
                }
            }

            // Display progress
            ProgressBar((float) (svo_position / (float) nb_frames), 30);

            // Check if we have reached the end of the video
            if (svo_position >= (nb_frames - 1)) { // End of SVO
                cout << "\nSVO end has been reached. Exiting now.\n";
                exit_app = true;
            }
        }
    }
    if (output_as_video) {
        // Close the video writer
        video_writer->release();
        delete video_writer;
    }

    zed.close();
    return 0;
}

