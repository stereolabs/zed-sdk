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

using namespace sl;

// Struct that holds skeleton data
struct SkeletonHandler {
    FbxNode* root;
    std::vector<FbxNode*> joints;
};

void parseArgs(int argc, char** argv, sl::InitParameters& param);
SkeletonHandler CreateSkeleton(FbxScene* pScene);

int main(int argc, char **argv) {
    // Create a ZED camera object
    sl::Camera zed;

    // Set configuration parameters
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION::HD1080; 
    init_parameters.camera_fps = 30;
    // The FBX Scene axis and unit are right handed, Y-Up and Centimeter. Do not change these parameters.
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_parameters.coordinate_units = UNIT::CENTIMETER;

    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;

    parseArgs(argc, argv, init_parameters);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Error " << returned_state << ", exit program.\n";
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

    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Error " << returned_state << ", exit program.\n";
        return EXIT_FAILURE;
    }

    // Enable the Objects detection module
    sl::ObjectDetectionParameters obj_det_params;
    // Tracking is mandatory, do not disable it.
    obj_det_params.enable_tracking = true; 
    obj_det_params.enable_body_fitting = true;
    // Do not change this parameter, the FBX export expects 34 bones, it is not compatible with 18
    obj_det_params.body_format = sl::BODY_FORMAT::POSE_34;
   
    // Only HUMAN_BODY_X models are compatible.
    obj_det_params.detection_model = sl::DETECTION_MODEL::HUMAN_BODY_FAST;
    returned_state = zed.enableObjectDetection(obj_det_params);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Error enable OD" << returned_state << ", exit program.\n";
        zed.close();
        return EXIT_FAILURE;
    }

    FbxManager* fbx_manager = nullptr;
    FbxScene* fbx_scene = nullptr;
    // Prepare the FBX SDK.
    InitializeSdkObjects(fbx_manager, fbx_scene);

    bool has_started = false;
    sl::Timestamp ts_start = sl::Timestamp(0);
    sl::RuntimeParameters rt_params;
    rt_params.measure3D_reference_frame = sl::REFERENCE_FRAME::WORLD;

    // Configure object detection runtime parameters
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    objectTracker_parameters_rt.detection_confidence_threshold = 70;

    // Create ZED Objects filled in the main loop
    sl::Objects bodies;

    // Create countdown
    int counter = 3; //duration of the countdown
    std::cout << "FBX Export tool ... " << std::endl;
    while (counter >= 1)
    {
        std::cout << "\r Start recording in : " << counter - 1 << " sec" << std::flush;
        sl::sleep_ms(1000);
        counter--;
    }
    std::cout << std::endl;
    std::cout << "Recording ... use Ctrl-C to stop." << std::endl;
    SetCtrlHandler();

    FbxTime time;
    int key_index = 0;

    sl::ERROR_CODE err = sl::ERROR_CODE::FAILURE;

    // Create skeleton hierarchy
    SkeletonHandler skeleton = CreateSkeleton(fbx_scene);
    FbxNode* root_node = fbx_scene->GetRootNode();
    // Add skeleton in the Scene
    root_node->AddChild(skeleton.root);

    // List of all anim layers, in case there are multiple skeletons in the scene
    std::map<int, FbxAnimLayer*> animLayers;
    
    std::vector<std::string> vec_str_component = {FBXSDK_CURVENODE_COMPONENT_X, FBXSDK_CURVENODE_COMPONENT_Y, FBXSDK_CURVENODE_COMPONENT_Z};

    // Main loop
    while (!exit_app) {
        err = zed.grab(rt_params);
        if (err == sl::ERROR_CODE::SUCCESS) {
            // Retrieve Detected Human Bodies
            zed.retrieveObjects(bodies, objectTracker_parameters_rt);

            if (!has_started) {
                ts_start = bodies.timestamp;
                has_started = true;
            }

            // Compute animation timestamp
            sl::Timestamp ts_ms = (bodies.timestamp - ts_start).getMilliseconds();
            time.SetMilliSeconds(ts_ms);

            // For each detection
            for (int i = 0; i < bodies.object_list.size(); i++) {
                sl::ObjectData obj = bodies.object_list[i];
                // Create a new animLayer if it is a new detection
               if (animLayers.find(obj.id) == animLayers.end())
                {
                    FbxAnimStack* anim_stack = FbxAnimStack::Create(fbx_scene, ("Anim Stack  ID " + std::to_string(obj.id)).c_str());
                    // Create the base layer (this is mandatory)
                    FbxAnimLayer* anim_base_layer = FbxAnimLayer::Create(fbx_scene, ("Base Layer " + std::to_string(obj.id)).c_str());
                    anim_stack->AddMember(anim_base_layer);
                    animLayers[obj.id] = anim_base_layer;
                }

                auto anim_id = animLayers[obj.id];

                // For each keypoint
                for (int j = 0; j < skeleton.joints.size(); j++) {
                    auto joint = skeleton.joints[j];
                    sl::Orientation lcl_rotation = sl::Orientation(obj.local_orientation_per_joint[j]);

                    // Set translation of the root (first joint)
                    if (j == 0) {
                        sl::float3 rootPosition = obj.keypoint[0];
                        joint->LclTranslation.GetCurveNode(anim_id, true);
                        
                        for(int t = 0; t<3; t++){
                            FbxAnimCurve* lCurve = joint->LclTranslation.GetCurve(anim_id, vec_str_component[t].c_str(), true);
                            if (lCurve) {
                                lCurve->KeyModifyBegin();
                                key_index = lCurve->KeyAdd(time);
                                lCurve->KeySet(key_index,
                                    time,
                                    rootPosition.v[t],
                                    FbxAnimCurveDef::eInterpolationConstant);
                                lCurve->KeyModifyEnd();
                            }
                        }
                        // Use global rotation for the root
                        lcl_rotation = sl::Orientation(obj.global_root_orientation);
                    }

                    // Convert rotation to euler angles
                    FbxQuaternion quat = FbxQuaternion(lcl_rotation.x, lcl_rotation.y, lcl_rotation.z, lcl_rotation.w);
                    FbxVector4 rota_euler;
                    rota_euler.SetXYZ(quat);

                    // Set local rotation of the joint
                    for(int r = 0; r<3; r++){
                        FbxAnimCurve* lCurve = joint->LclRotation.GetCurve(anim_id, vec_str_component[r].c_str(), true);
                        if (lCurve) {
                            lCurve->KeyModifyBegin();
                            key_index = lCurve->KeyAdd(time);
                            lCurve->KeySet(key_index,
                                time,
                                rota_euler[r],
                                FbxAnimCurveDef::eInterpolationConstant);
                            lCurve->KeyModifyEnd();
                        }
                    }
                }
            }
        }
        else {
            exit_app = true;
        }
    }

    bodies.object_list.clear();
    // Disable modules
    zed.disableObjectDetection();
    // Disable positional tracking and close the camera
    zed.disablePositionalTracking();
    zed.close();

    // Save the scene.    
    std::string sampleFileName = "ZedSkeletons.fbx";
    auto result = SaveScene(fbx_manager, fbx_scene, sampleFileName.c_str(), 0 /*save as binary*/);
    DestroySdkObjects(fbx_manager, result);
    
    if (result == false) {
        FBXSDK_printf("\n\nAn error occurred while saving the scene...\n");
        return EXIT_FAILURE;
    }else
        return EXIT_SUCCESS;
}


// Create Skeleton node hierarchy based on sl::BODY_FORMAT::POSE_34 body format.
SkeletonHandler CreateSkeleton(FbxScene* pScene) {
    FbxNode* reference_node = FbxNode::Create(pScene, ("Skeleton"));
    SkeletonHandler skeleton;
    for (int i = 0; i < static_cast<int>(sl::BODY_PARTS_POSE_34::LAST); i++) {
        FbxString joint_name;
        joint_name =  sl::toString(static_cast<sl::BODY_PARTS_POSE_34>(i)).c_str();
        FbxSkeleton* skeleton_node_attribute = FbxSkeleton::Create(pScene, joint_name);
        skeleton_node_attribute->SetSkeletonType(FbxSkeleton::eLimbNode);
        skeleton_node_attribute->Size.Set(1.0);
        FbxNode* skeleton_node = FbxNode::Create(pScene, joint_name.Buffer());
        skeleton_node->SetNodeAttribute(skeleton_node_attribute);

        FbxDouble3 tr(local_joints_translations[i].x, local_joints_translations[i].y, local_joints_translations[i].z);
        skeleton_node->LclTranslation.Set(tr);

        skeleton.joints.push_back(skeleton_node);
    }

    reference_node->AddChild(skeleton.joints[0]);

    // Build skeleton node hierarchy. 
    for (int i = 0; i < skeleton.joints.size(); i++) {
        for (int j = 0; j < childenIdx[i].size(); j++)        
            skeleton.joints[i]->AddChild(skeleton.joints[childenIdx[i][j]]);        
    }

    skeleton.root = reference_node;

    return skeleton;
}

void parseArgs(int argc, char** argv, sl::InitParameters& param) {
    if (argc > 1 && std::string(argv[1]).find(".svo") != std::string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        std::cout << "[Sample] Using SVO File input: " << argv[1] << std::endl;
    }
    else if (argc > 1 && std::string(argv[1]).find(".svo") == std::string::npos) {
        std::string arg = std::string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            // Stream input mode - IP + port
            std::string ip_adress = std::to_string(a) + "." + std::to_string(b) + "." + std::to_string(c) + "." + std::to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()), port);
            std::cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << std::endl;
        }
        else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            std::cout << "[Sample] Using Stream input, IP : " << argv[1] << std::endl;
        }
        else if (arg.find("HD2K") != std::string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD2K;
            std::cout << "[Sample] Using Camera in resolution HD2K" << std::endl;
        }
        else if (arg.find("HD1080") != std::string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD1080;
            std::cout << "[Sample] Using Camera in resolution HD1080" << std::endl;
        }
        else if (arg.find("HD720") != std::string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD720;
            std::cout << "[Sample] Using Camera in resolution HD720" << std::endl;
        }
        else if (arg.find("VGA") != std::string::npos) {
            param.camera_resolution = sl::RESOLUTION::VGA;
            std::cout << "[Sample] Using Camera in resolution VGA" << std::endl;
        }
    }
}
