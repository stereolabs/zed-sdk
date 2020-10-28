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

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");

int main(int argc, char **argv) {

    if (argc != 4) {
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
    if (!strcmp(argv[3], "1") || !strcmp(argv[3], "3"))
        app_type = LEFT_AND_DEPTH;
    if (!strcmp(argv[3], "4"))
        app_type = LEFT_AND_DEPTH_16;

    // Check if exporting to AVI or SEQUENCE
    if (strcmp(argv[3], "0") && strcmp(argv[3], "1"))
        output_as_video = false;

    if (!output_as_video && !directoryExists(output_path)) {
        print("Input directory doesn't exist. Check permissions or create it." + output_path);
        return EXIT_FAILURE;
    }

    if (!output_as_video && output_path.back() != '/' && output_path.back() != '\\') {
        print("Error: output folder needs to end with '/' or '\\'."+output_path);
        return EXIT_FAILURE;
    }

    // Create ZED objects
    Camera zed;

    // Specify SVO path parameter
    InitParameters init_parameters;
    init_parameters.input.setFromSVOFile(svo_input_path.c_str());
    init_parameters.coordinate_units = UNIT::MILLIMETER;

    // Open the camera
    ERROR_CODE zed_open_state = zed.open(init_parameters);
    if (zed_open_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", zed_open_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Get image size
    Resolution image_size = zed.getCameraInformation().camera_configuration.resolution;

    Mat left_image(image_size, MAT_TYPE::U8_C4);
    cv::Mat left_image_ocv = slMat2cvMat(left_image);

    Mat right_image(image_size, MAT_TYPE::U8_C4);
    cv::Mat right_image_ocv = slMat2cvMat(right_image);

    Mat depth_image(image_size, MAT_TYPE::F32_C1);
    cv::Mat depth_image_ocv = slMat2cvMat(depth_image);

    cv::Mat image_sideByside;
    if(output_as_video)
        image_sideByside = cv::Mat(image_size.height, image_size.width *2, CV_8UC3);

    // Create video writer
    cv::VideoWriter video_writer;
    if (output_as_video) {
#if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
        int fourcc = CV_FOURCC('M','J','P','G');
#else
        int fourcc = cv::VideoWriter::fourcc('M', '4', 'S', '2'); // MPEG-4 part 2 codec
#endif
        int frame_rate = fmax(zed.getInitParameters().camera_fps, 25); // Minimum write rate in OpenCV is 25
        video_writer.open(output_path, fourcc, frame_rate, cv::Size(image_size.width*2, image_size.height));
        if (!video_writer.isOpened()) {
            print("Error: OpenCV video writer cannot be opened. Please check the .avi file path and write permissions.");
            zed.close();
            return EXIT_FAILURE;
        }
    }

    RuntimeParameters rt_param;
    rt_param.sensing_mode = SENSING_MODE::FILL;

    // Start SVO conversion to AVI/SEQUENCE
    print("Converting SVO... Use Ctrl-C to interrupt conversion.");

    int nb_frames = zed.getSVONumberOfFrames();
    int svo_position = 0;
    zed.setSVOPosition(svo_position);

    SetCtrlHandler();

    while (!exit_app) {
        sl::ERROR_CODE err = zed.grab(rt_param);
        if (err == ERROR_CODE::SUCCESS) {
            svo_position = zed.getSVOPosition();

            // Retrieve SVO images
            zed.retrieveImage(left_image, VIEW::LEFT);

            switch (app_type) {
                case LEFT_AND_RIGHT:
                    zed.retrieveImage(right_image, VIEW::RIGHT);
                    break;
                case LEFT_AND_DEPTH:
                    zed.retrieveImage(right_image, VIEW::DEPTH);
                    break;
                case LEFT_AND_DEPTH_16:
                    zed.retrieveMeasure(depth_image, MEASURE::DEPTH);
                    break;
                default:
                    break;
            }

            if (output_as_video) {
                // Convert SVO image from RGBA to RGB
                cv::cvtColor(left_image_ocv, image_sideByside(cv::Rect(0,0,image_size.width,image_size.height)), cv::COLOR_BGRA2BGR);
                cv::cvtColor(right_image_ocv, image_sideByside(cv::Rect(image_size.width,0,image_size.width,image_size.height)), cv::COLOR_BGRA2BGR);
                // Write the RGB image in the video
                video_writer.write(image_sideByside);
            } else {
                // Generate filenames
                ostringstream filename1;
                filename1 << output_path << "/left" << setfill('0') << setw(6) << svo_position << ".png";
                ostringstream filename2;
                filename2 << output_path << (app_type == LEFT_AND_RIGHT ? "/right" : "/depth") << setfill('0') << setw(6) << svo_position << ".png";

                // Save Left images
                cv::imwrite(filename1.str(), left_image_ocv);

                // Save depth
                if (app_type != LEFT_AND_DEPTH_16)
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
        } else if (err == sl::ERROR_CODE::END_OF_SVOFILE_REACHED){
            print("SVO end has been reached. Exiting now.");
            exit_app = true;
        }
        else {
            print("Grab Error: ",err);
            exit_app = true;
        }
    }
    if (output_as_video) {
        // Close the video writer
        video_writer.release();
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
