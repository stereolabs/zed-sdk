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


/*************************************************************************
** This sample demonstrates how to use the ZED for positional tracking  **
** and display camera motion in an OpenGL window. 		                **
**************************************************************************/

// Standard includes
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "TrackingViewer.hpp"

// Using std namespace
using namespace std;
using namespace sl;

// Create ZED objects
sl::Camera zed;
sl::Pose camera_pose;
std::thread zed_callback;
bool quit = false;
std::string csvName; // CSV file to log camera motion and timestamp

// OpenGL window to display camera motion
TrackingViewer viewer;

const int MAX_CHAR = 128;

// Sample functions
void startZED();
void run();
void close();
void transformPose(sl::Transform &pose, float tx);
void parse_args(int argc, char **argv, sl::InitParameters &initParameters);

int main(int argc, char **argv) {

    // Set configuration parameters for the ZED
    InitParameters initParameters;
    initParameters.camera_resolution = RESOLUTION_HD720;
    initParameters.depth_mode = DEPTH_MODE_PERFORMANCE;
    initParameters.coordinate_units = UNIT_METER;
    initParameters.coordinate_system = COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP;
    initParameters.sdk_verbose = true;

    parse_args(argc, argv, initParameters);

    // Open the camera
    ERROR_CODE err = zed.open(initParameters);
    if (err != sl::SUCCESS) {
        std::cout << sl::errorCode2str(err) << std::endl;
        zed.close();
        return 1; // Quit if an error occurred
    }

    // Set positional tracking parameters
    TrackingParameters trackingParameters;
    trackingParameters.initial_world_transform = sl::Transform::identity();
    trackingParameters.enable_spatial_memory = true;     // Enable Spatial memory

    // Start motion tracking
    zed.enableTracking(trackingParameters);

    // Initialize OpenGL viewer
    viewer.init();

    // Start ZED callback
    startZED();

    // Set the display callback
    glutCloseFunc(close);
    glutMainLoop();

    return 0;
}


/**
 *   Launch ZED thread. Using a thread here allows to retrieve camera motion and display it in a GL window concurrently.
 **/
void startZED() {
    quit = false;
    zed_callback = std::thread(run);
}

/**
 *  This function loops to get image and motion data from the ZED. It is similar to a callback.
 *  Add your own code here.
 **/
void run() {

    float tx = 0, ty = 0, tz = 0;
    float rx = 0, ry = 0, rz = 0;

    // Get the distance between the center of the camera and the left eye
    float translation_left_to_center = zed.getCameraInformation().calibration_parameters.T.x * 0.5f;

    // Create text for GUI
    char text_rotation[MAX_CHAR];
    char text_translation[MAX_CHAR];

    // If activated, create a CSV file to log motion tracking data
    std::ofstream outputFile;
    if (!csvName.empty()) {
        outputFile.open(csvName + ".csv");
        if (!outputFile.is_open())
            cout << "WARNING: Can't create CSV file. Run the application with administrator rights." << endl;
        else
            outputFile << "Timestamp(ns);Rotation_X(rad);Rotation_Y(rad);Rotation_Z(rad);Position_X(m);Position_Y(m);Position_Z(m);" << endl;
    }

    while (!quit) {
        if (zed.grab() == SUCCESS) {
            // Get the position of the camera in a fixed reference frame (the World Frame)
            TRACKING_STATE tracking_state = zed.getPosition(camera_pose, sl::REFERENCE_FRAME_WORLD);

            if (tracking_state == TRACKING_STATE_OK) {
                // getPosition() outputs the position of the Camera Frame, which is located on the left eye of the camera.
                // To get the position of the center of the camera, we transform the pose data into a new frame located at the center of the camera.
                // The generic formula used here is: Pose(new reference frame) = M.inverse() * Pose (camera frame) * M, where M is the transform between two frames.
                transformPose(camera_pose.pose_data, translation_left_to_center); // Get the pose at the center of the camera (baseline/2 on X axis)

                // Update camera position in the viewing window
                viewer.updateZEDPosition(camera_pose.pose_data);

                // Get quaternion, rotation and translation
                sl::float4 quaternion = camera_pose.getOrientation();
                sl::float3 rotation = camera_pose.getEulerAngles(); // Only use Euler angles to display absolute angle values. Use quaternions for transforms.
                sl::float3 translation = camera_pose.getTranslation();

                // Display translation and rotation (pitch, yaw, roll in OpenGL coordinate system)
                snprintf(text_rotation, MAX_CHAR, "%3.2f; %3.2f; %3.2f", rotation.x, rotation.y, rotation.z);
                snprintf(text_translation, MAX_CHAR, "%3.2f; %3.2f; %3.2f", translation.x, translation.y, translation.z);

                // Save the pose data in a csv file
                if (outputFile.is_open())
                    outputFile << zed.getCameraTimestamp() << "; " << text_rotation << "; " << text_translation << ";" << endl;
            }

            // Update rotation, translation and tracking state values in the OpenGL window
            viewer.updateText(string(text_translation), string(text_rotation), tracking_state);
        } else sl::sleep_ms(1);
    }
}

/**
 *  Trasnform pose to create a Tracking Frame located in a separate location from the Camera Frame
 **/
void transformPose(sl::Transform &pose, float tx) {
    sl::Transform transform_;
    transform_.setIdentity();
    // Move the tracking frame by tx along the X axis
    transform_.tx = tx;
    // Apply the transformation
    pose = Transform::inverse(transform_) * pose * transform_;
}

/**
 *  This function parses and checks command line arguments
 **/
void parse_args(int argc, char **argv, sl::InitParameters &initParameters) {
    // Check number of arguments. Cannot be higher than 3
    if (argc > 3) {
        cout << "Only an SVO path or a CSV name can be passed as an argument." << endl;
        exit(0);
    }
}

/**
 * This function closes the ZED camera, its callback (thread) and the GL viewer
 **/
void close() {
    quit = true;
    zed_callback.join();
    zed.disableTracking("./ZED_spatial_memory"); // Record an area file

    zed.close();
    viewer.exit();
}
