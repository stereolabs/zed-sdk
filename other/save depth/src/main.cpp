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



/**************************************************************************************************
 ** This sample demonstrates how to save depth information provided by the ZED Camera,           **
 ** or by an SVO file, in different image formats (PNG 16-Bits, PFM).                            **
 **                                                                                              **
 **************************************************************************************************/

#define NOMINMAX

#include <iomanip>
#include <signal.h>
#include <iostream>
#include <limits>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sl/Camera.hpp>

#ifndef _SL_JETSON_   // defined in zed/utils/GlobalDefines.hpp --> Detect if we are running under a Jetson TK1 or TX1
#include <opencv2/core/utility.hpp>
#endif

using namespace std;

typedef struct SaveParamStruct {
    sl::POINT_CLOUD_FORMAT PC_Format;
    sl::DEPTH_FORMAT Depth_Format;
    sl::String saveName;
    bool askSavePC;
    bool askSaveDepth;
    bool stop_signal;
} SaveParam;

sl::Camera *zed_ptr;
SaveParam *param;

std::string getFormatNamePC(sl::POINT_CLOUD_FORMAT f) {
    std::string str_;
    switch (f) {
        case sl::POINT_CLOUD_FORMAT_XYZ_ASCII:
            str_ = "XYZ";
            break;
        case sl::POINT_CLOUD_FORMAT_PCD_ASCII:
            str_ = "PCD";
            break;
        case sl::POINT_CLOUD_FORMAT_PLY_ASCII:
            str_ = "PLY";
            break;
        case sl::POINT_CLOUD_FORMAT_VTK_ASCII:
            str_ = "VTK";
            break;
        default:
            break;
    }
    return str_;
}

std::string getFormatNameD(sl::DEPTH_FORMAT f) {
    std::string str_;
    switch (f) {
        case sl::DEPTH_FORMAT_PNG:
            str_ = "PNG";
            break;
        case sl::DEPTH_FORMAT_PFM:
            str_ = "PFM";
            break;
        case sl::DEPTH_FORMAT_PGM:
            str_ = "PGM";
            break;
        default:
            break;
    }
    return str_;
}

// Fonctions to handle CTRL-C event (exit event)

#ifdef _WIN32

BOOL CtrlHandler(DWORD fdwCtrlType) {
    switch (fdwCtrlType) {
            //Handle the CTRL-C signal.
        case CTRL_C_EVENT:
            printf("\nQuitting...\n");
            delete zed_ptr;
            exit(0);
        default:
            return FALSE;
    }
}
#else

void nix_exit_handler(int s) {
    printf("\nQuitting...\n");
    delete zed_ptr;
    exit(1);
}
#endif

// Save function called in a thread

void saveProcess() {
    while (!param->stop_signal) {

        if (param->askSaveDepth) {
            float max_value = std::numeric_limits<unsigned short int>::max();
            float scale_factor = max_value / zed_ptr->getDepthMaxRangeValue();

            std::cout << "Saving Depth Map " << param->saveName << " in " << getFormatNameD(param->Depth_Format) << " ..." << flush;
            sl::saveDepthAs(*zed_ptr, param->Depth_Format, param->saveName, scale_factor);
            std::cout << "done" << endl;
            param->askSaveDepth = false;
        }

        if (param->askSavePC) {
            std::cout << "Saving Point Cloud " << param->saveName << " in " << getFormatNamePC(param->PC_Format) << " ..." << flush;
            sl::savePointCloudAs(*zed_ptr, param->PC_Format, param->saveName, true, false);
            std::cout << "done" << endl;
            param->askSavePC = false;
        }

        sl::sleep_ms(1);
    }
}


// Save function using opencv

void saveSbSimage(std::string filename) {
    sl::Mat sbs_sl;
    zed_ptr->retrieveImage(sbs_sl, sl::VIEW_SIDE_BY_SIDE);
    sbs_sl.write(filename.c_str());
    std::cout << "Image saved !" << std::endl;
}

int main(int argc, char **argv) {

    sl::Camera zed;
    zed_ptr = &zed;

    sl::InitParameters parameters;

    int nbFrames = 0;
    sl::DEPTH_MODE depth_mode = sl::DEPTH_MODE_PERFORMANCE;


    //*
    //*  OpenCV4Tegra (2.4) and OpenCV 3.1 handles parameters in a different ways.
    //*  In the following lines, we show how to handle both ways by checking if we are on the Jetson (_SL_JETSON_) or not to take "OpenCV2.4" or "OpenCV3.1" style
    //*


#ifdef _SL_JETSON_
    const cv::String keys = {
        "{ h | help      |                    | print help message }"
        "{ f | filename  |                    | path to SVO filename}"
        "{ r | resolution|   2                |ZED Camera resolution, ENUM 0: HD2K   1: HD1080   2: HD720   3: VGA}"
        "{ m | mode      |   2                |Disparity Map mode, ENUM 1: PERFORMANCE  2: MEDIUM   3: QUALITY}"
        "{ p | path      |   ./               |Output path (can include output filename prefix)}"
        "{ d | device    |   -1               |CUDA device ID }"
    };

    cv::CommandLineParser parser(argc, argv, keys.c_str());
#else

    const cv::String keys =
            "{help h usage ? || print this message}"
            "{filename f||SVO filename (ex : -f=test.svo  or --filename=test.svo) }"
            "{resolution r|2|ZED Camera resolution, ENUM 0: HD2K   1: HD1080   2: HD720   3: VGA (ex : -r=1  or --resolution=1 for HD1080)}"
            "{mode m|2|Disparity Map mode, ENUM 1: PERFORMANCE  2: MEDIUM   3: QUALITY  (ex : -m=1  or --mode=1)}"
            "{path p|./|Output path (can include output filename prefix) (ex : -p=./../ or --path=./../)}"
            "{device d|-1|CUDA device (ex : -d=0 or --device=0) }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Sample from ZED SDK" + std::string(sl::Camera::getSDKVersion())); //about is not available under OpenCV2.4
#endif

    // return 0;
#ifdef _SL_JETSON_
    if (parser.get<bool>("help")) {
        parser.printParams();
        return 0;
    }
#else
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
#endif

    cv::String filename = parser.get<cv::String>("filename");
    if (filename.empty()) {
        cout << "Saving depth from ZED camera (LIVE)" << endl;
        int resolution = parser.get<int>("resolution");
        switch (resolution) {
            case 0: cout << "Resolution set to HD2K" << endl;
                break;
            case 1: cout << "Resolution set to HD1080" << endl;
                break;
            case 2: cout << "Resolution set to HD720" << endl;
                break;
            case 3: cout << "Resolution set to VGA" << endl;
                break;
            default: cout << "Invalid Resolution " << resolution << endl;
                break;
        }
        parameters.camera_resolution = static_cast<sl::RESOLUTION> (resolution);
    } else {
        cout << "Saving depth from SVO : " << filename << endl;
        parameters.svo_input_filename = filename.c_str();
        nbFrames = zed.getSVONumberOfFrames();
        std::cout << "SVO number of frames : " << nbFrames << std::endl;
    }

    int mode = parser.get<int>("mode");
    switch (mode) {
        case 1:
            cout << "Mode set to PERFORMANCE" << endl;
            depth_mode = sl::DEPTH_MODE_PERFORMANCE;
            break;
        case 2:
            cout << "Mode set to MEDIUM" << endl;
            depth_mode = sl::DEPTH_MODE_MEDIUM;
            break;
        case 3:
            cout << "Mode set to QUALITY" << endl;
            depth_mode = sl::DEPTH_MODE_QUALITY;
            break;
        default:
            cout << "Invalid depth quality " << mode << endl;
            break;
    }
    depth_mode = static_cast<sl::DEPTH_MODE> (mode);
    string path = parser.get<std::string>("path");
    int device = parser.get<int>("device");

#ifndef _SL_JETSON_
    //this check is available on 3.1 but not on OpenCV4Tegra
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }
#else
    // No check can be performed easily with OpenCV2.4
    // Check that parameters are set correctly or it will result to a SegFault
#endif

    string prefixPC = "PC_"; //Default output file prefix
    string prefixDepth = "Depth_"; //Default output file prefix

    parameters.depth_mode = depth_mode;
    parameters.coordinate_units = sl::UNIT_MILLIMETER;
    parameters.sdk_verbose = 1;
    parameters.sdk_gpu_id = device;

    sl::ERROR_CODE err = zed.open(parameters);
    cout << errorCode2str(err) << endl;

    //Quit if an error occurred
    if (err != sl::SUCCESS) {
        return 1;
    }

    //CTRL-C (= kill signal) handler
#ifdef _WIN32
    SetConsoleCtrlHandler((PHANDLER_ROUTINE) CtrlHandler, TRUE);
#else // unix
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = nix_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
#endif

    char key = ' '; // key pressed
    sl::Mat depthDisplay;
    cv::Mat depth_cv;

    bool printHelp = false;
    std::string helpString = "[d] save Depth, [P] Save Point Cloud, [m] change format PC, [n] change format Depth, [q] quit";

    int depth_clamp = 5000;
    zed.setDepthMaxRangeValue(depth_clamp);

    int mode_PC = 0;
    int mode_Depth = 0;

    param = new SaveParam();
    param->askSavePC = false;
    param->askSaveDepth = false;
    param->stop_signal = false;
    param->PC_Format = static_cast<sl::POINT_CLOUD_FORMAT> (mode_PC);
    param->Depth_Format = static_cast<sl::DEPTH_FORMAT> (mode_Depth);

    std::thread grab_thread(saveProcess);

    bool quit_ = false;

    std::cout << " Press 's' to save Side by side images" << std::endl;

    std::cout << " Press 'p' to save Point Cloud" << std::endl;
    std::cout << " Press 'd' to save Depth image" << std::endl;

    std::cout << " Press 'm' to switch Point Cloud format" << std::endl;
    std::cout << " Press 'n' to switch Depth format" << std::endl;

    std::cout << " Press 'q' to exit" << std::endl;
    int count = 0;
    while (!quit_ && (zed.getSVOPosition() <= nbFrames)) {

        zed.grab(sl::SENSING_MODE_STANDARD);
        zed.retrieveImage(depthDisplay, sl::VIEW_DEPTH);

        depth_cv = cv::Mat(depthDisplay.getHeight(), depthDisplay.getWidth(), CV_8UC4, depthDisplay.getPtr<sl::uchar1>(sl::MEM_CPU));

        if (printHelp) // Write help text on the image if needed
            cv::putText(depth_cv, helpString, cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(111, 111, 111, 255), 2);

        cv::imshow("Depth", depth_cv);
        key = cv::waitKey(5);

        switch (key) {
            case 'p':
            case 'P':
                param->saveName = std::string(path + prefixPC + to_string(count)).c_str();
                param->askSavePC = true;
                break;

            case 'd':
            case 'D':
                param->saveName =  std::string(path + prefixDepth + to_string(count)).c_str();
                param->askSaveDepth = true;
                break;

            case 'm': // point cloud format
            case 'M':
            {
                mode_PC++;
                param->PC_Format = static_cast<sl::POINT_CLOUD_FORMAT> (mode_PC % 4);
                std::cout << "Format Point Cloud " << getFormatNamePC(param->PC_Format) << std::endl;
            }
                break;

            case 'n': // depth format
            case 'N':
            {
                mode_Depth++;
                param->Depth_Format = static_cast<sl::DEPTH_FORMAT> (mode_Depth % 3);
                std::cout << "Format Depth " << getFormatNameD(param->Depth_Format) << std::endl;
            }
                break;

            case 'h': // print help
            case 'H':
                printHelp = !printHelp;
                cout << helpString << endl;
                break;

            case 's': // save side by side images
                saveSbSimage(std::string("ZEDImage") + std::to_string(count) + std::string(".png"));
                break;
            case 'q': // quit
            case 'Q':
            case 27:
                quit_ = true;
                break;
        }
        count++;
    }

    param->stop_signal = true;
    grab_thread.join();

    zed.close();
    return 0;
}


