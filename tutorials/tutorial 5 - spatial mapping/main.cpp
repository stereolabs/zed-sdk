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


#include <sl/Camera.hpp>

using namespace sl;

int main(int argc, char **argv) {

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION_HD720; // Use HD720 video mode (default fps: 60)
    init_params.coordinate_system = COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP; // Use a right-handed Y-up coordinate system
    init_params.coordinate_units = UNIT_METER; // Set units in meters

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != SUCCESS)
        exit(-1);

    // Enable positional tracking with default parameters. Positional tracking needs to be enabled before using spatial mapping
    sl::TrackingParameters tracking_parameters;
    err = zed.enableTracking(tracking_parameters);
    if (err != SUCCESS)
        exit(-1);

    // Enable spatial mapping
    sl::SpatialMappingParameters mapping_parameters;
    err = zed.enableSpatialMapping(mapping_parameters);
    if (err != SUCCESS)
        exit(-1);

    // Grab data during 500 frames
    int i = 0;
    sl::Mesh mesh; // Create a mesh object
    while (i < 500) {
		// For each new grab, mesh data is updated 
        if (zed.grab() == SUCCESS) {
            // In the background, spatial mapping will use newly retrieved images, depth and pose to update the mesh
            sl::SPATIAL_MAPPING_STATE mapping_state = zed.getSpatialMappingState();

            // Print spatial mapping state
            std::cout << "\rImages captured: " << i << " / 500  ||  Spatial mapping state: " << spatialMappingState2str(mapping_state) << "                     " << std::flush;
            i++;
        }
    }
    printf("\n");

    // Extract, filter and save the mesh in a obj file
    printf("Extracting Mesh...\n");
    zed.extractWholeMesh(mesh); // Extract the whole mesh
    printf("Filtering Mesh...\n");
    mesh.filter(sl::MeshFilterParameters::FILTER_LOW); // Filter the mesh (remove unnecessary vertices and faces)
    printf("Saving Mesh...\n");
    mesh.save("mesh.obj"); // Save the mesh in an obj file


    // Disable tracking and mapping and close the camera
    zed.disableSpatialMapping();
    zed.disableTracking();
    zed.close();
    return 0;
}
