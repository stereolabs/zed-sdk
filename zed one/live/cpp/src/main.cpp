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

/**
 * @file main.cpp
 * @brief ZED One Live Capture Sample
 *
 * This sample demonstrates how to grab images from a ZED One camera
 * and display them using OpenCV.
 */

// ZED include
#include <sl/CameraOne.hpp>

// Sample includes
#include <utils.hpp>

// Using std and sl namespaces
using namespace std;
using namespace sl;

// ---- ROI selection state ----
struct RoiSelector {
    cv::Point origin; // in original image coords
    cv::Point end;    // in original image coords
    bool selecting = false;
    bool roi_ready = false;
};

static RoiSelector roi_sel;

static const std::string g_win_name = "ZED One - Live";

static void onMouse(int event, int x, int y, int /*flags*/, void* userdata) {
    auto* zed = static_cast<CameraOne*>(userdata);
    cv::Point pt(x, y);

    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            roi_sel.origin = pt;
            roi_sel.end = pt;
            roi_sel.selecting = true;
            roi_sel.roi_ready = false;
            break;

        case cv::EVENT_MOUSEMOVE:
            if (roi_sel.selecting)
                roi_sel.end = pt;
            break;

        case cv::EVENT_LBUTTONUP:
            if (roi_sel.selecting) {
                roi_sel.end = pt;
                roi_sel.selecting = false;
                roi_sel.roi_ready = true;

                // Build the ROI rectangle (handle any drag direction)
                int rx = std::min(roi_sel.origin.x, roi_sel.end.x);
                int ry = std::min(roi_sel.origin.y, roi_sel.end.y);
                int rw = std::abs(roi_sel.end.x - roi_sel.origin.x);
                int rh = std::abs(roi_sel.end.y - roi_sel.origin.y);

                if (rw > 0 && rh > 0) {
                    sl::Rect roi(rx, ry, rw, rh);
                    auto err = zed->setCameraSettings(VIDEO_SETTINGS::AEC_AGC_ROI, roi);
                    if (err == ERROR_CODE::SUCCESS)
                        std::cout << "[Sample] AEC/AGC ROI set to [" << rx << ", " << ry << ", " << rw << "x" << rh << "]" << std::endl;
                    else
                        print("setCameraSettings AEC_AGC_ROI", err);
                }
            }
            break;
    }
}

int main(int argc, char** argv) {
    // Create a ZED Camera object
    CameraOne zed;

    InitParametersOne init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.camera_resolution = sl::RESOLUTION::AUTO;

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state > ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Create a Mat to store images
    Mat zed_image;

    // Get and display camera information
    auto cam_info = zed.getCameraInformation();
    std::cout << "\n=== Camera Information ===" << std::endl;
    std::cout << "Model:      " << cam_info.camera_model << std::endl;
    std::cout << "Input:      " << cam_info.input_type << std::endl;
    std::cout << "Serial:     " << cam_info.serial_number << std::endl;
    std::cout << "Resolution: " << cam_info.camera_configuration.resolution.width << "x" << cam_info.camera_configuration.resolution.height
              << std::endl;
    std::cout << "FPS:        " << cam_info.camera_configuration.fps << std::endl;
    std::cout << "==========================\n" << std::endl;

    // Setup resizable window and mouse callback for ROI selection
    cv::namedWindow(g_win_name, cv::WINDOW_NORMAL);
    cv::setMouseCallback(g_win_name, onMouse, &zed);

    std::cout << "Draw a rectangle with the mouse to set the AEC/AGC ROI." << std::endl;
    std::cout << "Press 'r' to reset the ROI. Press 'q' to quit." << std::endl;

    int t_min, t_max;
    zed.getCameraSettingsRange(VIDEO_SETTINGS::AUTO_EXPOSURE_TIME_RANGE, t_min, t_max);
    std::cout << "Available Auto exposure time range: [" << t_min << "µs, " << t_max << "µs]" << std::endl;
    zed.getCameraSettings(VIDEO_SETTINGS::AUTO_EXPOSURE_TIME_RANGE, t_min, t_max);
    std::cout << "Current Auto exposure time value: [" << t_min << "µs, " << t_max << "µs]" << std::endl;

    // Capture new images until 'q' is pressed
    char key = ' ';
    while (key != 'q') {
        // Check that a new image is successfully acquired
        returned_state = zed.grab();
        if (returned_state <= ERROR_CODE::SUCCESS) {
            // Retrieve image
            zed.retrieveImage(zed_image);
            cv::Mat display = slMat2cvMat(zed_image).clone();

            // Draw the rectangle while selecting or after selection (in original image coords)
            if (roi_sel.selecting || roi_sel.roi_ready) {
                cv::rectangle(display, roi_sel.origin, roi_sel.end, cv::Scalar(0, 255, 0), 2);
            }

            cv::imshow(g_win_name, display);
        } else {
            print("Grab", returned_state);
            if (returned_state != sl::ERROR_CODE::CAMERA_REBOOTING)
                break;
        }

        key = cv::waitKey(10);

        // Reset ROI on 'r'
        if (key == 'r') {
            sl::Rect empty_roi;
            auto err = zed.setCameraSettings(VIDEO_SETTINGS::AEC_AGC_ROI, empty_roi, true);
            if (err == ERROR_CODE::SUCCESS)
                std::cout << "[Sample] AEC/AGC ROI reset to full image." << std::endl;
            else
                print("Reset AEC_AGC_ROI", err);
            roi_sel.roi_ready = false;
        }
    }

    zed.close();
    return EXIT_SUCCESS;
}
