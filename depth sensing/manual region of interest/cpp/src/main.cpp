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

struct ROIdata
{
    const int radius = 50;
    cv::Point2i last_pt;
    cv::Mat mask, seeds, image;
    bool selectInProgress_frgrnd = false;
    bool selectInProgress_backgrnd = false;
    bool isInit = false;
    cv::Mat im_bgr, frgrnd, bckgrnd;

    void init(sl::Resolution resolution){
        mask = cv::Mat(resolution.height, resolution.width, CV_8UC1);
        mask.setTo(0);
        seeds = cv::Mat(resolution.height, resolution.width, CV_8UC1);
        seeds.setTo(cv::GrabCutClasses::GC_PR_BGD);
        image = cv::Mat(resolution.height, resolution.width, CV_8UC4);
        image.setTo(127);
        isInit = false;
        frgrnd.release();
        bckgrnd.release();
    }

    void set(bool background, cv::Point current_pt){
        cv::line(seeds, current_pt, last_pt, cv::Scalar(background ? cv::GrabCutClasses::GC_BGD: cv::GrabCutClasses::GC_PR_FGD), radius);
        cv::line(image, current_pt, last_pt, cv::Scalar(background ? cv::Scalar::all(0) : cv::Scalar::all(255)), radius);
        last_pt = current_pt;
    }

    void updateImage(cv::Mat &im){
        cv::addWeighted(image, 0.5, im, 0.5, 0, im);
    }

    void compute(cv::Mat &cvImage){
        cv::cvtColor(cvImage, im_bgr, cv::COLOR_BGRA2BGR);
        cv::Mat seeds_cpy;
        seeds.copyTo(seeds_cpy);
        cv::grabCut(im_bgr, seeds_cpy, cv::Rect(0,0,im_bgr.cols, im_bgr.rows), frgrnd, bckgrnd, 1,isInit ? cv::GrabCutModes::GC_EVAL : cv::GrabCutModes::GC_INIT_WITH_MASK);

        mask.setTo(255);
        mask.setTo(0, seeds_cpy & 1);
        cv::erode(mask, mask,cv::Mat(5,5,CV_8UC1));

        isInit = true;
    }
};

static void onMouse(int event, int x, int y, int, void* data) {
    auto pdata = reinterpret_cast<ROIdata*> (data);
    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
        {
            pdata->last_pt = cv::Point(x, y); 
            pdata->selectInProgress_frgrnd = true;
            break;
        }
        case cv::EVENT_LBUTTONUP:
        {
            pdata->selectInProgress_frgrnd = false;            
            break;
        }

        case cv::EVENT_RBUTTONDOWN:
        {
            pdata->last_pt = cv::Point(x, y); 
            pdata->selectInProgress_backgrnd = true;
            break;
        }
        case cv::EVENT_RBUTTONUP:
        {
            pdata->selectInProgress_backgrnd = false;            
            break;
        }

        case cv::EVENT_MOUSEMOVE:
        {
            if(pdata->selectInProgress_backgrnd)
                pdata->set(true, cv::Point(x, y));
            
            if(pdata->selectInProgress_frgrnd)
                pdata->set(false, cv::Point(x, y));
        }
    }
}

int main(int argc, char **argv) {

    // Create a ZED Camera object
    Camera zed;

    InitParameters init_parameters;
    init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL;
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

    std::cout << 
        "Press LeftButton (and keep it pressed) to select foreground seeds\n"
        "Press LeftRight (and keep it pressed) to select background seeds\n"
        "Press 'a' to apply the ROI\n"
        "Press 'r' to reset the ROI\n"
        "Press 's' to save the ROI as image file to reload it later\n"
        << std::endl;

    auto resolution = zed.getCameraInformation().camera_configuration.resolution;
    // Create a Mat to store images
    Mat zed_image(resolution, MAT_TYPE::U8_C4);
    cv::Mat cvImage(resolution.height, resolution.width, CV_8UC4, zed_image.getPtr<sl::uchar1>(MEM::CPU));
    Mat zed_depth_image(resolution, MAT_TYPE::U8_C4);
    cv::Mat cvDepthImage(resolution.height, resolution.width, CV_8UC4, zed_depth_image.getPtr<sl::uchar1>(MEM::CPU));

    ROIdata roi_data;
    roi_data.init(resolution);

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

            roi_data.updateImage(cvImage);

            cv::imshow(imWndName, cvImage);
            //Display the image and the current global ROI
            cv::imshow(depthWndName, cvDepthImage);
            cv::imshow(ROIWndName, roi_data.mask);
        }

        key = cv::waitKey(10);

        // Apply Current ROI
        if (key == 'a') {
            zed.retrieveImage(zed_image, VIEW::LEFT);
            roi_data.compute(cvImage);
            
            Mat slROI(resolution, MAT_TYPE::U8_C1, roi_data.mask.data, roi_data.mask.step);
            zed.setRegionOfInterest(slROI);
        } else if (key == 'r') { //Reset ROI
            Mat emptyROI;
            zed.setRegionOfInterest(emptyROI);
            // clear user data
        } else if (key == 's') {
            // Save the current Mask to be loaded in another app
            cv::imwrite(mask_name, roi_data.mask);
        }else if (key == 'r') 
            roi_data.init(resolution);        
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
        }
        else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        }else if (arg.find("HD1200") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1200;
            cout << "[Sample] Using Camera in resolution HD1200" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        }else if (arg.find("SVGA") != string::npos) {
            param.camera_resolution = RESOLUTION::SVGA;
            cout << "[Sample] Using Camera in resolution SVGA" << endl;
        }else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }
    } else {
        // Default
    }
}
