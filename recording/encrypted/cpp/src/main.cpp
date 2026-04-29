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

/*************************************************************************************
** This sample shows how to record and play back encrypted SVO2 files.              **
**                                                                                  **
** If the SVO file does not exist, it records from a live camera.                   **
** If the SVO file already exists, it plays it back.                                **
**                                                                                  **
** The encryption key can be:                                                       **
**   - A passphrase (any string, key derived via PBKDF2-SHA256)                     **
**   - A path to a 32-byte binary key file                                          **
**   - A 64-character hex string representing a raw 256-bit key                     **
**                                                                                  **
** Encrypted SVO2 files use AES-256-CTR and require OpenSSL to be installed.        **
*************************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include <opencv2/opencv.hpp>
#include "utils.hpp"

#include <filesystem>

// Using namespace
using namespace sl;
using namespace std;

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");

int main(int argc, char** argv) {

    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <file.svo2> <encryption_key>\n";
        cout << "\n";
        cout << "If the file does not exist, records an encrypted SVO from a live camera.\n";
        cout << "If the file already exists, plays it back.\n";
        cout << "\n";
        cout << "The encryption key can be a passphrase, a path to a 32-byte key file,\n";
        cout << "or a 64-char hex string.\n";
        return EXIT_FAILURE;
    }

    string svo_path = argv[1];
    string key = argv[2];
    bool file_exists = std::filesystem::exists(svo_path);

    if (!file_exists) {
        // ---- RECORDING MODE ----
        Camera zed;

        InitParameters init_parameters;
        init_parameters.depth_mode = DEPTH_MODE::NONE;
        init_parameters.async_image_retrieval = false;

        auto returned_state = zed.open(init_parameters);
        if (returned_state > ERROR_CODE::SUCCESS) {
            print("Camera Open", returned_state, "Exit program.");
            return EXIT_FAILURE;
        }

        // Set up encrypted recording
        RecordingParameters recording_parameters;
        recording_parameters.video_filename.set(svo_path.c_str());
        recording_parameters.compression_mode = SVO_COMPRESSION_MODE::H265;
        recording_parameters.encryption_key.set(key.c_str());

        returned_state = zed.enableRecording(recording_parameters);
        if (returned_state > ERROR_CODE::SUCCESS) {
            print("Enable Recording", returned_state, "Is OpenSSL installed?");
            zed.close();
            return EXIT_FAILURE;
        }

        print("Encrypted SVO is recording, use Ctrl-C to stop.");
        SetCtrlHandler();
        RecordingStatus rec_status;
        while (!exit_app) {
            if (zed.grab() <= ERROR_CODE::SUCCESS) {
                rec_status = zed.getRecordingStatus();
                printf("  Frames: %d ingested / %d encoded\r", rec_status.number_frames_ingested, rec_status.number_frames_encoded);
            } else
                break;
        }
        cout << endl;

        zed.disableRecording();
        zed.close();
        print("Recording stopped. Encrypted SVO saved to " + svo_path);

    } else {
        // ---- PLAYBACK MODE ----
        print("File exists, playing back: " + svo_path);
        Camera zed;

        InitParameters init_parameters;
        init_parameters.input.setFromSVOFile(svo_path.c_str());
        init_parameters.depth_mode = DEPTH_MODE::NONE;
        init_parameters.svo_decryption_key.set(key.c_str());

        auto returned_state = zed.open(init_parameters);
        if (returned_state > ERROR_CODE::SUCCESS) {
            print("Camera Open", returned_state, "Wrong key or corrupted file?");
            return EXIT_FAILURE;
        }

        auto resolution = zed.getCameraInformation().camera_configuration.resolution;
        sl::Resolution low_resolution(min(720, (int)resolution.width) * 2, min(404, (int)resolution.height));
        Mat svo_image(low_resolution, MAT_TYPE::U8_C4, MEM::CPU);
        cv::Mat svo_image_ocv(svo_image.getHeight(), svo_image.getWidth(), CV_8UC4, svo_image.getPtr<sl::uchar1>(MEM::CPU));

        int nb_frames = zed.getSVONumberOfFrames();
        int svo_frame_rate = zed.getInitParameters().camera_fps;
        print("[Info] Encrypted SVO contains " + to_string(nb_frames) + " frames");

        cout << " Press 's' to save a frame as PNG" << endl;
        cout << " Press 'f' to jump forward" << endl;
        cout << " Press 'b' to jump backward" << endl;
        cout << " Press 'q' to exit..." << endl;

        char key_pressed = ' ';
        while (key_pressed != 'q') {
            returned_state = zed.grab();
            if (returned_state <= ERROR_CODE::SUCCESS) {
                zed.retrieveImage(svo_image, VIEW::SIDE_BY_SIDE, MEM::CPU, low_resolution);
                int svo_position = zed.getSVOPosition();

                cv::imshow("Encrypted SVO Playback", svo_image_ocv);
                key_pressed = cv::waitKey(10);

                switch (key_pressed) {
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

                // Progress bar
                int pct = (nb_frames > 0) ? static_cast<int>(100.f * svo_position / nb_frames) : 0;
                int done = pct * 30 / 100;
                printf("\r[%.*s%.*s] %3d%%", done, "==============================", 30 - done, "------------------------------", pct);
            } else if (returned_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
                print("SVO end reached. Looping back to 0\n");
                zed.setSVOPosition(0);
            } else {
                print("Grab ZED", returned_state);
                break;
            }
        }
        zed.close();
    }

    return EXIT_SUCCESS;
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "[Sample]";
    if (err_code > ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else if (err_code < ERROR_CODE::SUCCESS)
        cout << "[Warning] ";
    else
        cout << " ";
    cout << msg_prefix << " ";
    if (err_code > ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}
