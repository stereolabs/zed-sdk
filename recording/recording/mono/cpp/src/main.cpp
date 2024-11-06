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

/****************************************************************************************
** This sample shows how to record video in Stereolabs SVO format.					   **
** SVO video files can be played with the ZED API and used with its different modules  **
*****************************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "utils.hpp"

// Using namespace
using namespace sl;
using namespace std;

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
void parseArgs(int argc, char **argv,sl::InitParameters& param);

int main(int argc, char **argv) {

    if (argc < 2) {
        cout << "Usage : Only the path of the output SVO file should be passed as argument.\n";
        return EXIT_FAILURE;
    }

    // Create a ZED camera
    Camera zed;
    
    // Set configuration parameters for the ZED
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::NONE;
    init_parameters.async_image_retrieval = false; //This parameter can be used to record SVO in camera FPS even if the grab loop is running at a lower FPS (due to compute for ex.)
    parseArgs(argc,argv,init_parameters);

    // Open the camera
    auto returned_state  = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Enable recording with the filename specified in argument
    RecordingParameters recording_parameters;
    recording_parameters.video_filename.set(argv[1]);
    recording_parameters.compression_mode = SVO_COMPRESSION_MODE::H265;
    returned_state = zed.enableRecording(recording_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Recording ZED : ", returned_state);
        zed.close();
        return EXIT_FAILURE;
    }

    // Start recording SVO, stop with Ctrl-C command
    print("SVO is Recording, use Ctrl-C to stop." );
    SetCtrlHandler();
    sl::RecordingStatus rec_status;
    while (!exit_app) {
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            // Each new frame is added to the SVO file
            rec_status = zed.getRecordingStatus();
            printf(" NFrames SVO: %d  / %d\n",rec_status.number_frames_ingested,rec_status.number_frames_encoded);
        }
        else
            break;
    }

    // Stop recording
    zed.disableRecording();
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

void parseArgs(int argc, char **argv,sl::InitParameters& param)
{
    if (argc > 2 && string(argv[2]).find(".svo")!=string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[2]);
        cout << "[Sample] Using SVO File input: " << argv[2] << endl;
    } else if (argc > 2 && string(argv[2]).find(".svo")==string::npos) {
        string arg = string(argv[2]);
        unsigned int a,b,c,d,port;
        if (sscanf(arg.c_str(),"%u.%u.%u.%u:%d", &a, &b, &c, &d,&port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a)+"."+to_string(b)+"."+to_string(c)+"."+to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()),port);
            cout<<"[Sample] Using Stream input, IP : "<<ip_adress<<", port : "<<port<<endl;
        }
        else  if (sscanf(arg.c_str(),"%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[2]));
            cout<<"[Sample] Using Stream input, IP : "<<argv[2]<<endl;
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
