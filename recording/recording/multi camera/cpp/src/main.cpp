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

/****************************************************************************************
** This sample shows how to record video in Stereolabs SVO format.					   **
** SVO video files can be played with the ZED API and used with its different modules  **
*****************************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

#ifndef _WIN32
#include <sl/CameraOne.hpp>
#endif

// Sample includes
#include "utils.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

template<typename T>
void Acquisition(T& zed) {
    sl::RecordingStatus rec_status;
    auto infos = zed.getCameraInformation();
    const int cam_fps = infos.camera_configuration.fps;	
    while (!exit_app) {
        // grab current images
        if (zed.grab() <= ERROR_CODE::SUCCESS) {
            // Sensors Data are already recorded (internally), no need to push them in the SVO file
			rec_status = zed.getRecordingStatus();
			if((rec_status.number_frames_encoded %cam_fps)==0)
				std::cout << "ZED "<<infos.camera_model<<"["<<infos.serial_number << "] FPS: " << zed.getCurrentFPS() <<" "<<rec_status.number_frames_encoded<<" rec \n";
        }
		sl::sleep_ms(5);
    }

    std::cout << "ZED "<<infos.camera_model<<"["<<infos.serial_number << "] QUIT \n";
    zed.disableRecording();
    zed.close();
}

inline
void setDepthMode(sl::InitParameters &ip){
    ip.depth_mode = DEPTH_MODE::NONE;
}

inline
void setDepthMode(sl::InitParametersOne &ip){
    // NA
}

template<typename T, typename IP>
bool OpenCamera(T& zed, int sn, uint64_t ts) {

    IP init_parameters;
    init_parameters.camera_resolution = RESOLUTION::AUTO;
    setDepthMode(init_parameters);

    RecordingParameters recording_parameters;
    recording_parameters.compression_mode = SVO_COMPRESSION_MODE::H265;

    init_parameters.input.setFromSerialNumber(sn);
    init_parameters.camera_fps = 30;
    ERROR_CODE err = zed.open(init_parameters);
    if (err <= ERROR_CODE::SUCCESS) {
        auto cam_info = zed.getCameraInformation();
        cout << cam_info.camera_model << ", SN: " << sn << " Opened" << endl;

        std::string svo_name = "SN" + std::to_string(sn) + "_" +  std::to_string(ts) + ".svo";
        recording_parameters.video_filename.set(svo_name.c_str()); 
        // Enable recording with the filename specified in argument
        
        auto returned_state = zed.enableRecording(recording_parameters); //W: Here is the recording of the Images in the SVO
        if (returned_state != ERROR_CODE::SUCCESS) {
            cout<<"Recording ZED : "<< returned_state<<"\n";
            zed.close();
            return false;
        } 
    } else {
        cout << "ZED SN:" << sn << " Error: " << err << endl;
        zed.close();
        return false;
    }
    return true;
}

void printDeviceInfo(const vector< DeviceProperties> & devs) {
	for (auto &dev : devs)
		std::cout << "ID : " << dev.id << ", model : " << dev.camera_model
				  << " , S/N : " << dev.serial_number
				  << " , state : " << dev.camera_state
				  << std::endl;
}

 int main(int argc, char** argv) {
     
	vector< DeviceProperties> zedList = Camera::getDeviceList();
	printDeviceInfo(zedList);

    vector< DeviceProperties> devOneList;

#ifndef _WIN32
    devOneList = CameraOne::getDeviceList();
	printDeviceInfo(devOneList);
#endif

    const int nb_One = devOneList.size() ;
    const int nb_Stereo = zedList.size();
	
	if (nb_One + nb_Stereo == 0) {
		cout << "No ZED Detected, exit program" << endl;
		return EXIT_FAILURE;
	}
	
	cout << nb_One << " ZED One Detected" << endl;
	cout << nb_Stereo << " ZED Stereo Detected" << endl;

	vector<Camera> zedsStereo(nb_Stereo);
#ifndef _WIN32
    vector<CameraOne> zedsOne(nb_One);
#endif
	
	auto init_ts = sl::getCurrentTimeStamp();

	zedList.insert( zedList.end(), devOneList.begin(), devOneList.end() );

	const int nb_detected_zed = zedList.size();
	
	bool zed_open = false;
    for (int z = 0; z < nb_detected_zed; z++){
        if(z < nb_Stereo){
            bool open = OpenCamera<sl::Camera, sl::InitParameters>(zedsStereo[z], zedList[z].serial_number, init_ts);
                zed_open = true;
        }
#ifndef _WIN32
        else{
            int z_one = z-nb_Stereo;
            bool open = OpenCamera<sl::CameraOne, sl::InitParametersOne>(zedsOne[z_one], zedList[z].serial_number, init_ts);
            if(open)
                zed_open = true;            
        }
#endif
    }

    if(!zed_open) {        
        cout << "No ZED opened, exit program" << endl;
        return EXIT_FAILURE;
    }
	
	SetCtrlHandler(); 

	// Create a grab thread for each opened camera
	vector<thread> thread_pool(nb_detected_zed); // compute threads
    for (int z = 0; z < nb_detected_zed; z++){
        if(z < nb_Stereo){
            if(zedsStereo[z].isOpened())
                thread_pool[z] = std::thread(Acquisition<sl::Camera>, ref(zedsStereo[z]));
        }
#ifndef _WIN32
        else{
            int z_one = z-nb_Stereo;
            //std::cout<<"z "<<z<<" zone "<<z_one<<" open "<<zedsOne[z_one].isOpened()<<"\n";
            if(zedsOne[z_one].isOpened())
                thread_pool[z] = std::thread(Acquisition<sl::CameraOne>, ref(zedsOne[z_one]));       
        }
#endif
    }

    // Ctrl +C to close
    while (!exit_app) // main loop
        sl::sleep_ms(20);
      
	// stop all running threads
	std::cout << "Exit signal, closing ZEDs and SVO files" << std::endl;
	sl::sleep_ms(100);

	// Wait for every thread to be stopped
	for (int z = 0; z < nb_detected_zed; z++)
		if(thread_pool[z].joinable())
				thread_pool[z].join();
  
    std::cout << "Exit" << std::endl;

     return EXIT_SUCCESS;
 }