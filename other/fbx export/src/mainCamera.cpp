///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2022, STEREOLABS.
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
#include <fbxsdk.h>

#include "Common.h"
#include "utils.hpp"

using namespace std;
using namespace sl;

void parseArgs(int argc, char** argv, InitParameters& param);
FbxNode* CreateMyZEDCamera(FbxScene* pScene, Camera& zed);

int main(int argc, char **argv) {

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::HD1080;
    init_parameters.camera_fps = 30;
    // The FBX Scene axis and unit are right handed, Y-Up and Centimeter. Do not change these parameters.
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_parameters.coordinate_units = UNIT::CENTIMETER;

    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;

    parseArgs(argc, argv, init_parameters);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error " << returned_state << ", exit program.\n";
        return EXIT_FAILURE;
    }

    // Enable positional tracking
    PositionalTrackingParameters tracking_parameters;
    tracking_parameters.enable_area_memory = false;
    tracking_parameters.enable_pose_smoothing = true;

    // Enable this parameter to align the tracking with the floor plane position. 
    // It is recommended not to disable it unless you know what you are doing.
    tracking_parameters.set_floor_as_origin = true;
    returned_state = zed.enablePositionalTracking(tracking_parameters);

    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error " << returned_state << ", exit program.\n";
        return EXIT_FAILURE;
    }

    FbxManager* fbx_manager = nullptr;
    FbxScene* fbx_scene = nullptr;
    // Prepare the FBX SDK.
    InitializeSdkObjects(fbx_manager, fbx_scene);

    //Create a fbx node for camera
    FbxNode* zed_camera_node = CreateMyZEDCamera(fbx_scene, zed);
    // Create an animation stack
    FbxAnimStack* anim_stack = FbxAnimStack::Create(fbx_scene, "Anim Stack");
    // Create the base layer (this is mandatory)
    FbxAnimLayer* anim_base_layer = FbxAnimLayer::Create(fbx_scene, "Base Layer");
    anim_stack->AddMember(anim_base_layer);
    // Get the cameras curve node for local translation.
    // The second parameter to GetCurveNode() is "true" to ensure
    // that the curve node is automatically created, if it does not exist.
    FbxAnimCurveNode* translation_curve_node = zed_camera_node->LclTranslation.GetCurveNode(anim_base_layer, true);
    FbxAnimCurveNode* rotation_curve_node = zed_camera_node->LclRotation.GetCurveNode(anim_base_layer, true);
    FbxTime time;
    int key_index = 0;                // Index for the keys that define the curve

    // Get the animation curve for local translation of the camera.
    // true: If the curve does not exist yet, create it.
    FbxAnimCurve *translation_[3];
    translation_[0] = zed_camera_node->LclTranslation.GetCurve(anim_base_layer, FBXSDK_CURVENODE_COMPONENT_X, true);
    translation_[1] = zed_camera_node->LclTranslation.GetCurve(anim_base_layer, FBXSDK_CURVENODE_COMPONENT_Y, true);
    translation_[2] = zed_camera_node->LclTranslation.GetCurve(anim_base_layer, FBXSDK_CURVENODE_COMPONENT_Z, true);

    // Get the animation curve for local rotation of the camera.
    // true: If the curve does not exist yet, create it.
    FbxAnimCurve *rotation_[3];
    rotation_[0] = zed_camera_node->LclRotation.GetCurve(anim_base_layer, FBXSDK_CURVENODE_COMPONENT_X, true);
    rotation_[1] = zed_camera_node->LclRotation.GetCurve(anim_base_layer, FBXSDK_CURVENODE_COMPONENT_Y, true);
    rotation_[2] = zed_camera_node->LclRotation.GetCurve(anim_base_layer, FBXSDK_CURVENODE_COMPONENT_Z, true);

    bool has_started = false;
    sl::Pose zed_pose;

    sl::Timestamp ts_start = sl::Timestamp(0);
    sl::RuntimeParameters rt_params;
    rt_params.measure3D_reference_frame = sl::REFERENCE_FRAME::WORLD;

    // Create countdown
    int counter = 3; //duration of the countdown
    // Countdown
    std::cout << "FBX Export tool ... " << std::endl;
    while (counter >= 1)
    {
        std::cout << "\r Starting recording in : " << counter - 1 << " sec" << std::flush;
        sl::sleep_ms(1000);
        counter--;
    }
    std::cout << std::endl;

    // Stop recording with Ctrl+C
    std::cout << "Recording ... use Ctrl-C to stop." << std::endl;
    SetCtrlHandler();

    // Main loop
    while (!exit_app) {
        if (zed.grab(rt_params) == ERROR_CODE::SUCCESS) {
            // Get the pose of the left eye of the camera with reference to the world frame
            if (zed.getPosition(zed_pose, sl::REFERENCE_FRAME::WORLD) != sl::POSITIONAL_TRACKING_STATE::OK) {
                continue;
            }

            // Apply a 90 degres rotation as a FBXCamera is always oriented toward the X axis by default
            sl::Orientation lcl_rotation = zed_pose.getOrientation() * sl::Orientation(sl::float4(0.0f, 0.707f, 0.0f, 0.707f)); // 90degres offset along the Y axis
            // Convert rotation to euler angles
            FbxQuaternion quat(lcl_rotation.x, lcl_rotation.y, lcl_rotation.z, lcl_rotation.w);
            FbxVector4 rota_euler;
            rota_euler.SetXYZ(quat);

            if (!has_started) {
                ts_start = zed_pose.timestamp;
                has_started = true;
            }

            // Compute animation timestamp
            sl::Timestamp ts_ms = (zed_pose.timestamp - ts_start).getMilliseconds();
            time.SetMilliSeconds(ts_ms);

            auto transaltion_value = zed_pose.getTranslation();
            // Set local translation of the camera
            for(int t =0; t < 3; t++ ){                
                translation_[t]->KeyModifyBegin();
                key_index = translation_[t]->KeyAdd(time);
                translation_[t]->KeySet(key_index, time, transaltion_value.v[t], FbxAnimCurveDef::eInterpolationLinear);
                translation_[t]->KeyModifyEnd();
            }

            // Set local rotation of the camera
            for(int r =0; r < 3; r++ ){                
                rotation_[r]->KeyModifyBegin();
                key_index = rotation_[r]->KeyAdd(time);
                rotation_[r]->KeySet(key_index, time, rota_euler[r], FbxAnimCurveDef::eInterpolationLinear);
                rotation_[r]->KeyModifyEnd();
            }
        }
        else 
            exit_app = true;
    }

    // Disable positional tracking and close the camera
    zed.disablePositionalTracking();
    zed.close();

    // Save the scene.
    std::string sampleFileName = "ZedCameraPath.fbx";
    auto result = SaveScene(fbx_manager, fbx_scene, sampleFileName.c_str(), 0 /*save as binary*/);
    DestroySdkObjects(fbx_manager, result);

    if (result == false) {
        FBXSDK_printf("\n\nAn error occurred while saving the scene...\n");
        return EXIT_FAILURE;
    }else
        return EXIT_SUCCESS;
}

//This function illustrates how to create and connect camera.
FbxNode* CreateMyZEDCamera(FbxScene* pScene, Camera& zed)
{
    if (!pScene)
        return NULL;

    pScene->GetGlobalSettings().SetCustomFrameRate(zed.getCameraInformation().camera_fps);
    //create a fbx node for camera
    FbxNode* cameraNode = FbxNode::Create(pScene, "zedCameraNode");
    //connect camera node to root node
    FbxNode* rootNode = pScene->GetRootNode();
    rootNode->AddChild(cameraNode);
    //create a camera, it's a node attribute of  camera node.
    fbxsdk::FbxCamera* camera = fbxsdk::FbxCamera::Create(pScene, "zedCamera");
    pScene->GetGlobalSettings().SetDefaultCamera((char*)camera->GetName());
    //set Camera as a node attribute of the FBX node.
    cameraNode->SetNodeAttribute(camera);
    camera->ProjectionType.Set(fbxsdk::FbxCamera::EProjectionType::ePerspective);

    //set camera format
    camera->SetFormat(FbxCamera::eHD);
    //set camera aperture format
    camera->SetApertureFormat(FbxCamera::eCustomAperture);
    //set camera aperture mode
    camera->SetApertureMode(FbxCamera::eVertical);
    //set camera FOV
    double lFOV = zed.getCameraInformation().calibration_parameters.left_cam.v_fov;
    camera->FieldOfView.Set(lFOV);
    //set camera focal length
    double lFocalLength = camera->ComputeFocalLength(lFOV);
    camera->FocalLength.Set(lFocalLength);
    cameraNode->LclTranslation.Set(FbxDouble3(0, 0, 0));
    cameraNode->LclRotation.Set(FbxDouble3(0, 0, 0));
    cameraNode->LclScaling.Set(FbxDouble3(1, 1, 1));

    return cameraNode;
}

void parseArgs(int argc, char** argv, InitParameters& param) {
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        cout << "[Sample] Using SVO File input: " << argv[1] << endl;
    }
    else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        }
        else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        }
        else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        }
        else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        }
        else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        }
        else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }
    }
}