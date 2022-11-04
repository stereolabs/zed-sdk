///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2022, STEREOLABS.
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

/****************************************************************************************************
 ** This sample demonstrates how to apply an exclusion ROI to all ZED SDK measures                 **
 ** This can be very useful to avoid noise from a vehicle bonnet or drone propellers for instance  **
 ***************************************************************************************************/


// Standard includes
#include <stdio.h>
#include <string.h>

// ZED include
#include <sl/Camera.hpp>

// OpenCV include (for display)
#include <opencv2/opencv.hpp>

// Using std and sl namespaces
using namespace std;
using namespace sl;

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix);
void parseArgs(int argc, char **argv, InitParameters& param);

#define SELECT_RECT 1

#if SELECT_RECT

struct ROIdata {
    // Current ROI, 0: means discard, other value will keep the pixel
    cv::Mat ROI;
    cv::Rect selection_rect;
    cv::Point origin_rect;
    bool selectInProgress = false;
    bool selection = false;

    void reset(bool full = true) {
        selectInProgress = false;
        selection_rect = cv::Rect(0, 0, 0, 0);
        if (full) {
            ROI.setTo(0);
            selection = false;
        }
    }
};

static void onMouse(int event, int x, int y, int, void* data) {
    auto pdata = reinterpret_cast<ROIdata*> (data);
    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
        {
            pdata->origin_rect = cv::Point(x, y);
            pdata->selectInProgress = true;
            break;
        }
        case cv::EVENT_LBUTTONUP:
        {
            pdata->selectInProgress = false;
            // set ROI to valid for the given rectangle
            cv::rectangle(pdata->ROI, pdata->selection_rect, cv::Scalar(250), -1);
            pdata->selection = true;
            break;
        }
        case cv::EVENT_RBUTTONDOWN:
        {
            pdata->reset(false);
            break;
        }
    }

    if (pdata->selectInProgress) {
        pdata->selection_rect.x = MIN(x, pdata->origin_rect.x);
        pdata->selection_rect.y = MIN(y, pdata->origin_rect.y);
        pdata->selection_rect.width = abs(x - pdata->origin_rect.x) + 1;
        pdata->selection_rect.height = abs(y - pdata->origin_rect.y) + 1;
    }
}
#else

struct ROIdata {
    // Current ROI, 0: means discard, other value will keep the pixel
    cv::Mat ROI;
    std::vector<std::vector<cv::Point>> polygons;
    std::vector<cv::Point> current_select;
    bool selection = false;
    bool selectInProgress = false;

    void reset() {
        polygons.clear();
        ROI.setTo(0);
        selection = true;
        selectInProgress = false;
    }
};

static void onMouse(int event, int x, int y, int, void* data) {
    auto pdata = reinterpret_cast<ROIdata*> (data);
    switch (event) {
    case cv::EVENT_LBUTTONDOWN :
        pdata->selectInProgress = true;
        break;
    case cv::EVENT_MOUSEMOVE:
        if (pdata->selectInProgress)
            pdata->current_select.push_back(cv::Point(x, y));
        break;
    case cv::EVENT_LBUTTONUP:
        if (pdata->current_select.size() > 2) {
            pdata->polygons.push_back(pdata->current_select);
            pdata->current_select.clear();
        }
        pdata->selectInProgress = false;
        break;
    case cv::EVENT_RBUTTONDOWN:    
        pdata->reset();
        break;    
    }
}
#endif

void applyMask(cv::Mat& cvImage, ROIdata& data);

int main(int argc, char **argv) {

    // Create a ZED Camera object
    Camera zed;

    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::HD720;
    parseArgs(argc, argv, init_parameters);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    cv::String imWndName = "Image";
    cv::String depthWndName = "Depth";
    cv::String ROIWndName = "ROI";
    cv::namedWindow(imWndName, cv::WINDOW_NORMAL);
    cv::namedWindow(ROIWndName, cv::WINDOW_NORMAL);
    cv::namedWindow(depthWndName, cv::WINDOW_NORMAL);

#if SELECT_RECT
    std::cout << "Draw some rectangles on the left image with a left click\n";
#else
    std::cout << "Draw some shapes on the left image with a left click\n";
#endif
    std::cout << "Press 'a' to apply the ROI\n"
        "Press 'r' to reset the ROI\n"
        "Press 's' to save the ROI as image file to reload it later\n"
        "Press 'l' to load the ROI from an image file" << std::endl;

    auto resolution = getResolution(init_parameters.camera_resolution);
    // Create a Mat to store images
    Mat zed_image(resolution, MAT_TYPE::U8_C4);
    cv::Mat cvImage(resolution.height, resolution.width, CV_8UC4, zed_image.getPtr<sl::uchar1>(MEM::CPU));

    Mat zed_depth_image(resolution, MAT_TYPE::U8_C4);
    cv::Mat cvDepthImage(resolution.height, resolution.width, CV_8UC4, zed_depth_image.getPtr<sl::uchar1>(MEM::CPU));

    ROIdata roi_data;
    roi_data.ROI = cv::Mat(resolution.height, resolution.width, CV_8UC1);
    roi_data.reset();

    // set Mouse Callback to handle User inputs
    cv::setMouseCallback(imWndName, onMouse, &roi_data);

    std::string mask_name = "Mask.png";

    // Capture new images until 'q' is pressed
    char key = ' ';
    while ((key != 'q') && (key != 27)) {
        // Check that a new image is successfully acquired
        returned_state = zed.grab();
        if (returned_state == ERROR_CODE::SUCCESS) {
            // Retrieve left image
            zed.retrieveImage(zed_image, VIEW::LEFT);
            zed.retrieveImage(zed_depth_image, VIEW::DEPTH);

            // Draw rectangle on the image
            if (roi_data.selection)
                applyMask(cvImage, roi_data);

            cv::imshow(imWndName, cvImage);
            //Display the image and the current global ROI
            cv::imshow(depthWndName, cvDepthImage);
            cv::imshow(ROIWndName, roi_data.ROI);
        }

        key = cv::waitKey(15);

        // Apply Current ROI
        if (key == 'a') {
            Mat slROI(resolution, MAT_TYPE::U8_C1, roi_data.ROI.data, roi_data.ROI.step);
            zed.setRegionOfInterest(slROI);
        } else if (key == 'r') { //Reset ROI
            Mat emptyROI;
            zed.setRegionOfInterest(emptyROI);
            // clear user data
            roi_data.reset();
        } else if (key == 's') {
            // Save the current Mask to be loaded in another app
            cv::imwrite(mask_name, roi_data.ROI);
        } else if (key == 'l') {
            // Load the mask from a previously saved file
            cv::Mat tmp = cv::imread(mask_name);
            if (!tmp.empty()) {
                roi_data.ROI = tmp;
                Mat slROI(resolution, MAT_TYPE::U8_C1, roi_data.ROI.data, roi_data.ROI.step);
                zed.setRegionOfInterest(slROI);
            } else std::cout << mask_name << " could not be found" << std::endl;
        }
    }

    // Exit
    zed.close();
    return EXIT_SUCCESS;
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout << " ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}

void parseArgs(int argc, char **argv, InitParameters& param) {
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        param.input.setFromSVOFile(argv[1]);
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


#if SELECT_RECT
void applyMask(cv::Mat& cvImage, ROIdata& data) {
    auto res = cvImage.size();
    const float darker = 0.8f; // make the image darker

    for (int y = 0; y < res.height; y++) {
        // line pointer
        uchar * ptr_mask = (uchar *) ((data.ROI.data) + y * data.ROI.step);
        sl::uchar4* ptr_image = (sl::uchar4*) (cvImage.data + y * cvImage.step);

        for (int x = 0; x < res.width; x++) {
            if (ptr_mask[x] == 0) {
                auto &px = ptr_image[x];
                // make the pixel darker without overflow
                px.x = px.x * darker;
                px.y = px.y * darker;
                px.z = px.z * darker;
            }
        }
    }

    // DIsplay current selection
    cv::rectangle(cvImage, data.selection_rect, cv::Scalar(255, 90, 0, 255), 3);
}
#else

inline bool contains(std::vector<cv::Point>& poly, cv::Point2f test) {
    int i, j;
    bool c = false;
    const int nvert = poly.size();
    for (i = 0, j = nvert - 1; i < nvert; j = i++) {
        if (((poly[i].y > test.y) != (poly[j].y > test.y)) &&
            (test.x < (poly[j].x - poly[i].x) * (test.y - poly[i].y) / (poly[j].y - poly[i].y) + poly[i].x))
            c = !c;
    }
    return c;
}

inline bool contains(std::vector<std::vector<cv::Point>>& polygons, cv::Point2f test) {
    bool c = false;
    for (auto& it : polygons) {
        c = contains(it, test);
        if (c) break;
    }
    return c;
}

void applyMask(cv::Mat& cvImage, ROIdata &data) {
    // left_sl and mask must be at the same size
    auto res = cvImage.size();
    const float darker = 0.8f; // make the image darker
    
    // Convert P�lygons into real Mask
#if 1 // manual check
    for (int y = 0; y < res.height; y++) {
        uchar* ptr_mask = (uchar*)((data.ROI.data) + y * data.ROI.step);
        sl::uchar4* ptr_image = (sl::uchar4*)(cvImage.data+ y * cvImage.step);
        for (int x = 0; x < res.width; x++) {
            if (contains(data.polygons, cv::Point2f(x,y)))
                ptr_mask[x] = 255;
            else {
                auto& px = ptr_image[x];
                // make the pixel darker without overflow
                px.x = px.x * darker;
                px.y = px.y * darker;
                px.z = px.z * darker;
            }
        }
    }
#else // same with open Function
    cv::fillPoly(data.ROI, data.polygons, 255);
#endif

    // Display current selection
    if (data.current_select.size() > 2) {
        auto last = data.current_select.back();
        for (auto& it : data.current_select) {
            cv::line(cvImage, last, it, cv::Scalar(30, 130, 240), 1);
            last = it;
        }
    }
}
#endif