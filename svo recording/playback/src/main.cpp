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

int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "Only the path of the input SVO file should be passed as argument.\n";
        return 1;
    }

    // Create ZED objects
    Camera zed;
    InitParameters initParameters;
    initParameters.svo_input_filename.set(argv[1]);

    // Open the ZED
    ERROR_CODE err = zed.open(initParameters);
    if (err != SUCCESS) {
        cout << errorCode2str(err) << endl;
        zed.close();
        return 1; // Quit if an error occurred
    }

    cv::Size size(zed.getResolution().width, zed.getResolution().height);
    cv::Size size_sbs(size.width * 2, size.height);

    // Define OpenCV window size
    int width = 720;
    int height = 404;
    Mat svo_image(width * 2, height, MAT_TYPE_8U_C4, MEM_CPU);
    cv::Mat svo_image_ocv = slMat2cvMat(svo_image);

    // Setup key, images, times
    char key = ' ';
    cout << " Press 's' to save SVO image as a PNG" << endl;
    cout << " Press 'f' to jump forward in the video" << endl;
    cout << " Press 'b' to jump backward in the video" << endl;
    cout << " Press 'q' to exit..." << endl;

    int svo_frame_rate = zed.getCameraFPS();
    int nb_frames = zed.getSVONumberOfFrames();

    // Start SVO playback
    while (key != 'q') {
        if (zed.grab() == SUCCESS) {

            // Get the side by side image
            zed.retrieveImage(svo_image, VIEW_SIDE_BY_SIDE, MEM_CPU, width, height);

            // Display the frame
            cv::imshow("View", svo_image_ocv);

            key = cv::waitKey(15);

            int svo_position = zed.getSVOPosition();

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

            // Check if we have reached the end of the video
            if (svo_position >= (nb_frames - 1)) { // End of SVO
                cout << "\nSVO end has been reached. Exiting now.\n";
                break;
            }

            ProgressBar((float) (svo_position / (float) nb_frames), 30);
        }
    }

    zed.close();
    return 0;
}
