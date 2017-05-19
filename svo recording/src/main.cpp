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




/***************************************************************************************************
 ** This sample demonstrates how to use the recording capabilities of the SDK.                    **
 ** The SVO file is used to simulate a ZED.                                                       **
 ***************************************************************************************************/

#include <sl/Camera.hpp>
#include "utils.hpp"

using namespace std;

int main(int argc, char **argv) {

    // Parse command line arguments and store them in a Option structure
    InfoOption modes;
    parse_args(argc, argv, modes);


    // Create a ZED camera and Configure parameters according to command line options
    sl::Camera zed;
    sl::InitParameters initParameters;
    if (!modes.recordingMode) initParameters.svo_input_filename = modes.svo_path.c_str();
    else initParameters.camera_fps = 30;
	initParameters.svo_real_time_mode = true;

    if (!modes.computeDisparity) initParameters.depth_mode = sl::DEPTH_MODE_NONE;

    // Open the ZED
    sl::ERROR_CODE err = zed.open(initParameters);
    if (err != sl::SUCCESS) {
        cout << sl::errorCode2str(err) << endl;
        zed.close();
        return 1; // Quit if an error occurred
    }


    // If recording mode has been activated, enable the recording module.
    if (modes.recordingMode) {
        sl::ERROR_CODE err = zed.enableRecording(modes.svo_path.c_str(), sl::SVO_COMPRESSION_MODE::SVO_COMPRESSION_MODE_LOSSLESS);

        if (err != sl::SUCCESS) {
            std::cout << "Error while recording. " << errorCode2str(err) << " " << err << std::endl;
            if (err == sl::ERROR_CODE_SVO_RECORDING_ERROR) std::cout << " Note : This error mostly comes from a wrong path or missing writting permissions..." << std::endl;
            zed.close();
            return 1;
        }
    }


    // Setup key, images, times
    char key = 'z';
    sl::Mat view;
    int waitKey_time = 10; //wait for key event for 10ms
    cout << " Press 'q' to exit..." << endl;

    // Defines actions to do, according to options
    initActions(&zed, modes);


    // Enter main loop
    while (key != 'q') {
        if (zed.grab() == sl::SUCCESS) {

            // Get the side by side image
            zed.retrieveImage(view, sl::VIEW_SIDE_BY_SIDE);
            cv::imshow("View",  cv::Mat(view.getHeight(), view.getWidth(), CV_8UC4, view.getPtr<sl::uchar1>(sl::MEM_CPU)));
            key = cv::waitKey(waitKey_time);

            // Record in SVO file if asked
            if (modes.recordingMode) {
                //////////// Only when using Live mode as input /////////////////////
                zed.record(); // record
            } else {
                //////////// Only when using SVO file as input /////////////////////
                // Performs defined action (convert to avi file, store images...)
                manageActions(&zed, key, modes);

                // If space bar has been pressed , put SVO in pause
                if (key == ' ') waitKey_time = (waitKey_time) ? 0 : 15; // pause

                //Check if we are at the end of the svo file, then close the actions (close the avi file for example) and exit the while loop
                if (zed.getSVOPosition() >= (zed.getSVONumberOfFrames() - 2)) { // end of SVO
                    exitActions();
                    std::cout<<"Finished... exiting now"<<std::endl;
                    break;
                }
            }
        } else sl::sleep_ms(1);
    }

    if (modes.recordingMode) zed.disableRecording();
    zed.close();
    return 0;
}
