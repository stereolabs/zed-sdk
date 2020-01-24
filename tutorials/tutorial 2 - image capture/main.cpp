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


#include <sl/Camera.hpp>

using namespace sl;

int main(int argc, char **argv) {

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD1080; // Use HD1080 video mode
    init_params.camera_fps = 30; // Set fps at 30

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        std::cout << "Error " << err << ", exit program.\n";
        return -1;
    }

    // Capture 50 frames and stop
    int i = 0;
    sl::Mat image;
    while (i < 50) {
        // Grab an image
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            // A new image is available if grab() returns ERROR_CODE::SUCCESS
            zed.retrieveImage(image, VIEW::LEFT); // Get the left image
            auto timestamp = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE); // Get the timestamp at the time the image was captured
            printf("Image resolution: %d x %d  || Image timestamp: %llu\n", (int) image.getWidth(), (int) image.getHeight(), (unsigned long long int) timestamp.getNanoseconds());
            i++;
        }
    }

    // Close the camera
    zed.close();
    return 0;
}
