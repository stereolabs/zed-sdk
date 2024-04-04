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

/**********************************************************************
 ** This sample demonstrates how to export both Image and Depth data **
    from the ZED SKD into standard PNG files                         **
 *********************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Opencv header
#include <opencv2/opencv.hpp>

// useful header to create directory
#ifdef WIN32
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif

// Using std and sl namespaces
using namespace std;
using namespace sl;

void parseArgs(int argc, char **argv, sl::InitParameters& param);
void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix);
void createDirectory(string);

int main(int argc, char **argv) {
    Camera zed;
    // Set configuration parameters for the ZED
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::ULTRA;    
    parseArgs(argc, argv, init_parameters);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // create a custom resolution to save ZED data
    auto camera_config = zed.getCameraInformation().camera_configuration;
    float image_aspect_ratio = camera_config.resolution.width / (1.f * camera_config.resolution.height);
    int requested_low_res_w = min(720, (int)camera_config.resolution.width);
    sl::Resolution res(requested_low_res_w, requested_low_res_w / image_aspect_ratio);
    
    // get the camera serial number
    int serial_number = zed.getCameraInformation().serial_number;

    // create directory to save the data
    string directory_name("ZED_"+to_string(serial_number));
    createDirectory(directory_name);
    string image_directory_name(directory_name+"/IMAGE");
    createDirectory(image_directory_name);
    string depth_directory_name(directory_name+"/DEPTH");
    createDirectory(depth_directory_name);

    Mat image, depth_map;
    // Main Loop 
    while (1) {        
        // Check that a new image is successfully acquired
        if (zed.grab() == ERROR_CODE::SUCCESS) {

            // retrieve current Image
            zed.retrieveImage(image, VIEW::LEFT, MEM::CPU, res);

            // save the image in PNG with the image timestamp as name
            auto timestamp = image.timestamp.getMilliseconds();
            image.write((image_directory_name+"/"+to_string(timestamp)+".png").c_str());

            // retrieve current depth map directly in 16bits (millimeter to be saved as png)
            zed.retrieveMeasure(depth_map, MEASURE::DEPTH_U16_MM, MEM::CPU, res);
            depth_map.write((depth_directory_name+"/"+to_string(timestamp)+".png").c_str());

            // display the current image
            cv::imshow("Image", cv::Mat((int) image.getHeight(), (int) image.getWidth(), CV_8UC4, image.getPtr<sl::uchar1>(sl::MEM::CPU)));
            auto k = cv::waitKey(20);
            if(k=='q') break;
        }
    }    

    // close the ZED
    zed.close();

    return EXIT_SUCCESS;
}

void parseArgs(int argc, char **argv, sl::InitParameters& param) {
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        cout << "[Sample] Using SVO File input: " << argv[1] << endl;
    } else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        }else if (arg.find("HD2K") != string::npos) {
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

void createDirectory(std::string name) {
#ifdef _WIN32
    CreateDirectory(name.c_str(), NULL);
#else
    mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
}