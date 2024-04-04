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

// ZED include
#include <sl/Camera.hpp>
#include "GLViewer.hpp"
#include "SK_Serializer.hpp"

// Main loop
int main(int argc, char **argv)
{
    sl::Camera zed;
    // Open the camera
    sl::InitParameters init_parameters;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // Opengl Coordinate system
    auto state = zed.open(init_parameters);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error open ZED " << state << std::endl;
        return -1;
    }

    // enable positional tracking (with default parameters)
    state = zed.enablePositionalTracking();
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error enable Positional Tracking" << state << std::endl;
        return -1;
    }

    // define body tracking parameters
    sl::BodyTrackingParameters body_tracking_parameters;
    body_tracking_parameters.detection_model = BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
    body_tracking_parameters.body_format = sl::BODY_FORMAT::BODY_38;
    body_tracking_parameters.enable_tracking = true;
    body_tracking_parameters.enable_body_fitting = false;
    state = zed.enableBodyTracking(body_tracking_parameters);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error enable Body Tracking" << state << std::endl;
        return -1;
    }

    GLViewer viewer;
    viewer.init(argc, argv);

    bool run = true;
    nlohmann::json bodies_json_file;
    sl::Bodies bodies;
    while (viewer.isAvailable())
    {
        if (zed.grab() == ERROR_CODE::SUCCESS)
        {
            // Retrieve Detected Human Bodies
            zed.retrieveBodies(bodies);

            // display detected bodies
            viewer.updateData(bodies);

            // Serialize dected bodies into a json container
            bodies_json_file[std::to_string(bodies.timestamp.getMilliseconds())] = sk::serialize(bodies);
        }
    }
    zed.close();

    std::string outfileName("detected_bodies.json");
    if (bodies_json_file.size())
    {
        std::ofstream file_out(outfileName);
        file_out << std::setw(4) << bodies_json_file << std::endl;
        file_out.close();
        std::cout << "Successfully saved the body data to bodies.json" << std::endl;

        // playback the recorded data
        std::ifstream in(outfileName);
        nlohmann::json skeletons_file;
        in >> skeletons_file;
        in.close();

        // restart viewer
        viewer.restart();
        // iterate over detected bodies
        for (auto& it : skeletons_file.get<nlohmann::json::object_t>()) {
            // deserialize current bodies
            auto objs = sk::deserialize(it.second);
            // display
            viewer.updateData(objs);
            viewer.isAvailable();
        }
    }
    else
        std::cout << "No body data to save." << std::endl;

    return EXIT_SUCCESS;
}
