///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
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

int streamer(int argc, char **argv)
{
    // Create a ZED camera
    Camera zed;

    // Set configuration parameters for the ZED
    InitParameters initParameters;
    initParameters.camera_resolution = RESOLUTION_HD2K;
    initParameters.depth_mode = DEPTH_MODE_PERFORMANCE;

    // Open the camera
    ERROR_CODE err = zed.open(initParameters);
    if (err != ERROR_CODE::SUCCESS) {
        std::cout << toString(err) << std::endl;
        zed.close();
        return -1; // Quit if an error occurred
    }

   err = zed.enableStreaming(2,42000,2000);
    if (err != ERROR_CODE::SUCCESS) {
        std::cout << "Streaming initialization error. " << toString(err) << std::endl;
        zed.close();
        return -2;
    }


    SetCtrlHandler();

    while (1) {
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            sl::sleep_ms(1);
        }
    }

    // Stop recording
    zed.disableStreaming();
    zed.close();
    return 0;
}

int batch_recorder(int argc, char **argv)
{
    // Create a ZED camera
    Camera zed;

    // Set configuration parameters for the ZED
    InitParameters initParameters;
    initParameters.camera_resolution = RESOLUTION_HD2K;
    initParameters.depth_mode = DEPTH_MODE_PERFORMANCE;


    // Open the camera
    ERROR_CODE err = zed.open(initParameters);
    if (err != ERROR_CODE::SUCCESS) {
        std::cout << toString(err) << std::endl;
        zed.close();
        return -1; // Quit if an error occurred
    }

    char fileName[256];
    int nbsvomax = 5;
    int nbsvo = 0;

    while(nbsvo<nbsvomax)
    {
         sprintf(fileName,"./test2_%d.svo",nbsvo);
         INIT_TIMER
         ERROR_CODE err = zed.enableRecording(sl::String(fileName),sl::SVO_COMPRESSION_MODE_HEVC);
         STOP_TIMER("enableRecording")
         if (err == 0)
         {
         int i=0;
         while(i<100)
         {
             if (zed.grab()==0)
             {
                 std::cout<<" Nb SVO : "<<nbsvo<<" Ct Image : "<<i<<std::endl;
                 zed.record();
                 i++;
             }
         }
         INIT_TIMER
         zed.disableRecording();
         STOP_TIMER("disableRecodring")
         }
         else
             std::cout<<" Failed to enable recording pass"<<nbsvo<<std::endl;

         nbsvo++;
    }

    zed.close();

}

int comp_recorder(int argc, char **argv)
{
    // Create a ZED camera
    Camera zed;

    // Set configuration parameters for the ZED
    InitParameters initParameters;
    initParameters.camera_resolution = RESOLUTION_HD2K;
    initParameters.depth_mode = DEPTH_MODE_PERFORMANCE;


    // Open the camera
    ERROR_CODE err = zed.open(initParameters);
    if (err != ERROR_CODE::SUCCESS) {
        std::cout << toString(err) << std::endl;
        zed.close();
        return -1; // Quit if an error occurred
    }

    char fileName[256];
    sprintf(fileName,"./record_gop_2fps.svo");
    err = zed.enableRecording(sl::String(fileName),sl::SVO_COMPRESSION_MODE_HEVC);

     if (err == 0)
     {
     int i=0;
     //5 min  = 15*60*5;
     int fMax =15*60*5;
     while(i<fMax)
     {
         if (zed.grab()==0)
         {
             zed.record();
             i++;
         }
     }
     zed.disableRecording();
     }

     sprintf(fileName,"./record_gop_256.svo");
     err = zed.enableRecording(sl::String(fileName),static_cast<sl::SVO_COMPRESSION_MODE>(sl::SVO_COMPRESSION_MODE_LAST+2));

      if (err == 0)
      {
      int i=0;
      //5 min  = 15*60*5;
      int fMax =15*60*5;
      while(i<fMax)
      {
          if (zed.grab()==0)
          {
              zed.record();
              i++;
          }
      }
      zed.disableRecording();
      }


    zed.close();

}


int main(int argc, char **argv) {

    //return comp_recorder(argc,argv);
    // return batch_recorder(argc,argv);
     return streamer(argc, argv);

}
