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

#include <sl/Camera.hpp>

using namespace std;
using namespace sl;

int main(int argc, char **argv) {

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::NEURAL; // Use NEURAL depth mode
    init_parameters.coordinate_units = UNIT::MILLIMETER; // Use millimeter units (for depth measurements)

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error " << returned_state << ", exit program." << endl;
        return EXIT_FAILURE;
    }

    // Capture 150 images then stop
    int frame_count = 0;
    int depth_count = 0;
    sl::Mat image, depth, point_cloud;

    const sl::Timestamp start_ts = sl::getCurrentTimeStamp();
    const int depth_every_n_frames = 6;
    
    while (frame_count < 150) {
        // A new image is available if read() returns ERROR_CODE::SUCCESS
        if (zed.read() == ERROR_CODE::SUCCESS) {
            // Retrieve left image
            zed.retrieveImage(image, VIEW::LEFT);
            frame_count++;
        }

        // Measurement are available if grab() returns ERROR_CODE::SUCCESS
        if(((frame_count % depth_every_n_frames) == 0) && (zed.grab() == ERROR_CODE::SUCCESS)) {
            // Retrieve depth map. Depth is aligned on the left image
            zed.retrieveMeasure(depth, MEASURE::DEPTH);
            // Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA);

            // Get and print distance value in mm at the center of the image
            // We measure the distance camera - object using Euclidean distance
            const int x = point_cloud.getWidth() / 2;
            const int y = point_cloud.getHeight() / 2;
            sl::float4 point_cloud_value;
            point_cloud.getValue(x, y, &point_cloud_value);

            if(std::isfinite(point_cloud_value.z)) // convert to float3 to use norm(), the 4th component is used to store the color
                cout<<"Distance to Camera at {"<<x<<";"<<y<<"}: "<<sl::float3(point_cloud_value).norm()<<"mm"<<endl;
            else
                cout<<"The Distance can not be computed at {"<<x<<";"<<y<<"}"<<endl;
            depth_count++;            
        }
    }

    auto diff = (sl::getCurrentTimeStamp() - start_ts).getSeconds();
    // Print the FPS
    if(diff > 0)
        std::cout << "Image: " << (frame_count / diff) << "FPS / Depth: " << (depth_count / diff) <<"FPS" << std::endl;

    // Close the camera
    zed.close();
    return EXIT_SUCCESS;
}