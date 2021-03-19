///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, STEREOLABS.
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

/*********************************************************************************
 ** This sample demonstrates how to capture 3D point cloud and detected objects **
 **      with the ZED SDK and display the result in an OpenGL window. 	        **
 *********************************************************************************/

// Standard includes
#include <iostream>
#include <fstream>

// Flag to disable the GUI to increase detection performances
// On low-end hardware such as Jetson Nano, the GUI significantly slows
// down the detection and increase the memory consumption
#define ENABLE_GUI 1

// Flag to enable/disable the tracklet merger module.
// Tracklet merger allows to reconstruct trajectories from objects from object detection module by using Re-ID between objects.
// For example, if an object is not seen during some time, it can be re-ID to a previous ID if the matching score is high enough
#define TRACKLET_MERGER 1

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#if ENABLE_GUI
#include "GLViewer.hpp"
#include "TrackingViewer.hpp"
#endif

// Using std and sl namespaces
using namespace std;
using namespace sl;
bool is_playback = false;
void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
void parseArgs(int argc, char **argv, InitParameters& param);

Camera zed;
InitParameters init_parameters;
ObjectDetectionParameters detection_parameters;
PositionalTrackingParameters positional_tracking_parameters;
RuntimeParameters runtime_parameters;
Objects objects;
Pose cam_pose;
ObjectDetectionRuntimeParameters detection_parameters_rt;
bool quit = false;
bool newFrame = false;
std::mutex guard;
int detection_confidence = 35;
TrackingViewer track_view_generator;
cv::Mat image_track_ocv;

#if TRACKLET_MERGER
std::deque<sl::Objects> objects_tracked_queue;
std::map<unsigned long long,Pose> camPoseMap_ms;


///
/// \brief ingestPoseInMap
/// \param ts: timestamp of the pose
/// \param pose : sl::Pose of the camera
/// \param batch_duration_sc: duration in seconds in order to remove past elements.
///
void ingestPoseInMap(sl::Timestamp ts, sl::Pose pose, int batch_duration_sc)
{
    std::map<unsigned long long,Pose>::iterator it = camPoseMap_ms.begin();
    for(auto it = camPoseMap_ms.begin(); it != camPoseMap_ms.end(); ) {
        if(it->first<ts.getMilliseconds() - (unsigned long long)batch_duration_sc*1000)
            it = camPoseMap_ms.erase(it);
        else
            ++it;
    }

    camPoseMap_ms[ts.getMilliseconds()]=pose;
}

///
/// \brief findClosestPoseFromTS : find closest sl::Pose according to timestamp. Use when resampling is used in tracklet merger, since generated objects can have a different
/// timestamp than the camera timestamp. If resampling==0, then std::map::find() will be enough.
/// \param timestamp in milliseconds. ( at least in the same unit than camPoseMap_ms)
/// \return sl::Pose found.
///
sl::Pose findClosestPoseFromTS(unsigned long long timestamp)
{
    sl::Pose pose = sl::Pose();
    unsigned long long ts_found = 0;
    if (camPoseMap_ms.find(timestamp)!=camPoseMap_ms.end()) {
        ts_found = timestamp;
        pose = camPoseMap_ms[timestamp];
    }
    else
    {
        std::map<unsigned long long,Pose>::iterator it = camPoseMap_ms.begin();
        unsigned long long diff_max_time = ULONG_LONG_MAX;
        while(it!=camPoseMap_ms.end())
        {
            long long diff = abs((long long)timestamp - (long long)it->first);
            if (diff<diff_max_time)
            {
                pose = it->second;
                diff_max_time = diff;
                ts_found = it->first;
            }
            it++;
        }
    }
    return pose;
}

///
/// \brief ingestInObjectsQueue : convert a list of trajectory from SDK retreiveBatchTrajectories to a sorted list of sl::Objects
/// \n Use this function to fill a std::deque<sl::Objects> that can be considered and used as a stream of objects with a delay.
/// \param trajs from retreiveBatchTrajectories
///
void ingestInObjectsQueue(std::vector<sl::Trajectory> trajs)
{
    // If list is empty, do nothing.
    if (trajs.empty())
        return;

    // add objects in map with timestamp as a key.
    // This ensure
    std::map<uint64_t,sl::Objects> list_of_newobjects;
    for (int i=0;i<trajs.size();i++)
    {
        sl::Trajectory current_traj = trajs.at(i);

        // Impossible but still better to check...
        if (current_traj.timestamp.size()!=current_traj.position.size())
            continue;


        //For each sample, construct a objetdata and put it in the corresponding sl::Objects
        for (int j=0;j<current_traj.timestamp.size();j++)
        {
            sl::Timestamp ts = current_traj.timestamp.at(j);
            sl::ObjectData newObjectData;
            newObjectData.id = current_traj.ID;
            newObjectData.tracking_state = current_traj.tracking_state;
            newObjectData.position = current_traj.position.at(j);
            newObjectData.label = current_traj.label;
            newObjectData.sublabel = current_traj.sublabel;


            if (list_of_newobjects.find(ts.getMilliseconds())!=list_of_newobjects.end())
                list_of_newobjects[ts.getMilliseconds()].object_list.push_back(newObjectData);
            else
            {
                sl::Objects current_obj;
                current_obj.timestamp.setMilliseconds(ts.getMilliseconds());
                current_obj.is_new = true;
                current_obj.is_tracked = true;
                current_obj.object_list.push_back(newObjectData);
               list_of_newobjects[ts.getMilliseconds()] = current_obj;
            }
        }
    }


    // Ingest in Queue of objects that will be empty by the main loop
    // Since std::map is sorted by key, we are sure that timestamp are continous.
    for (auto &elem : list_of_newobjects)
       objects_tracked_queue.push_back(elem.second);

    return;
}
#endif

///
/// \brief run : thread function that capture ZED
///
void run()
{
    while(!quit)
    {
        if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS)
        {
            // update confidence threshold based on TrackBar
            if(detection_parameters_rt.object_class_filter.empty())
                detection_parameters_rt.detection_confidence_threshold = detection_confidence;
            else // if using class filter, set confidence for each class
                for (auto& it : detection_parameters_rt.object_class_filter)
                    detection_parameters_rt.object_class_detection_confidence_threshold[it] = detection_confidence;

            guard.lock();
            zed.retrieveObjects(objects, detection_parameters_rt);
            guard.unlock();
            newFrame = true;
        }
        else
            quit=true;


    }
}


///
/// \brief main function
///
int main(int argc, char **argv) {
    
#ifdef _SL_JETSON_
    const bool isJetson = true;
#else
    const bool isJetson = false;
#endif

    // Create ZED objects
    init_parameters.camera_resolution = RESOLUTION::HD2K;
    init_parameters.sdk_verbose = true;
    // On Jetson (Nano, TX1/2) the object detection combined with an heavy depth mode could reduce the frame rate too much
    init_parameters.depth_mode = isJetson ? DEPTH_MODE::PERFORMANCE : DEPTH_MODE::ULTRA;
    init_parameters.depth_maximum_distance = 10.0f * 1000.0f;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    parseArgs(argc, argv, init_parameters);
    
    // Open the camera
    auto returned_state  = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Define the Objects detection module parameters
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_mask_output = false; // designed to give person pixel mask
    detection_parameters.detection_model = isJetson ? DETECTION_MODEL::MULTI_CLASS_BOX : DETECTION_MODEL::MULTI_CLASS_BOX_ACCURATE;

    auto camera_config = zed.getCameraInformation().camera_configuration;

    // If the camera is static in space, enabling this settings below provides better depth quality and faster computation
    positional_tracking_parameters.set_as_static = true;
    zed.enablePositionalTracking(positional_tracking_parameters);

    print("Object Detection: Loading Module...");
    returned_state = zed.enableObjectDetection(detection_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enableObjectDetection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

#if TRACKLET_MERGER
    sl::BatchTrajectoryParameters trajectory_parameters;
    trajectory_parameters.resampling_rate = 0;
    trajectory_parameters.batch_duration = 2.f;
    returned_state = zed.enableBatchTrajectories(trajectory_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enableTrajectoryPostProcess", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }
#endif

    // Detection runtime parameters
    // default detection threshold, apply to all object class
    detection_parameters_rt.detection_confidence_threshold = detection_confidence;
    // To select a set of specific object classes:
    detection_parameters_rt.object_class_filter = {OBJECT_CLASS::VEHICLE, OBJECT_CLASS::PERSON};
    // To set a specific threshold
    detection_parameters_rt.object_class_detection_confidence_threshold[OBJECT_CLASS::PERSON] = detection_confidence;
    detection_parameters_rt.object_class_detection_confidence_threshold[OBJECT_CLASS::VEHICLE] = detection_confidence;


#if ENABLE_GUI

    Resolution display_resolution(min((int)camera_config.resolution.width, 1280) , min((int)camera_config.resolution.height, 720));
    Resolution tracks_resolution(400, display_resolution.height);
    // create a global image to store both image and tracks view
    cv::Mat global_image(display_resolution.height, display_resolution.width + tracks_resolution.width, CV_8UC4);
    // retrieve ref on image part
    auto image_left_ocv = global_image(cv::Rect(0, 0, display_resolution.width, display_resolution.height));
    // retrieve ref on tracks view part
    image_track_ocv = global_image(cv::Rect(display_resolution.width, 0, tracks_resolution.width, tracks_resolution.height));
    image_track_ocv.setTo(0);
    // init an sl::Mat from the ocv image ref (which is in fact the memory of global_image)
    Mat image_left(display_resolution, MAT_TYPE::U8_C4, image_left_ocv.data, image_left_ocv.step);
    sl::float2 img_scale(display_resolution.width / (float)camera_config.resolution.width, display_resolution.height / (float)camera_config.resolution.height);

    // 2D tracks
    track_view_generator=TrackingViewer(tracks_resolution, camera_config.fps, init_parameters.depth_maximum_distance,3);
    track_view_generator.setCameraCalibration(camera_config.calibration_parameters);

    string window_name = "ZED| 2D View and Birds view";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL); // Create Window
    cv::createTrackbar("Confidence", window_name, &detection_confidence, 100);

    char key = ' ';
    Resolution pc_resolution(min((int)camera_config.resolution.width, 720) , min((int)camera_config.resolution.height, 404));
    auto camera_parameters = zed.getCameraInformation(pc_resolution).camera_configuration.calibration_parameters.left_cam;
    Mat point_cloud(pc_resolution, MAT_TYPE::F32_C4, MEM::GPU);
    GLViewer viewer;
    viewer.init(argc, argv, camera_parameters);
#endif

    runtime_parameters.confidence_threshold = 50;
    cam_pose.pose_data.setIdentity();
    bool gl_viewer_available=true;
    std::thread runner(run);
    sl::Timestamp init_app_ts = 0ULL;
    sl::Timestamp init_queue_ts = 0ULL;
    while (
       #if ENABLE_GUI
           gl_viewer_available &&
       #endif
           !quit) {

        if (newFrame && objects.is_new) {
            newFrame = false;
#if ENABLE_GUI
            zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA, MEM::GPU, pc_resolution);
            zed.getPosition(cam_pose, REFERENCE_FRAME::WORLD);
            zed.retrieveImage(image_left, VIEW::LEFT, MEM::CPU, display_resolution);
            zed.getPosition(cam_pose, REFERENCE_FRAME::CAMERA);

#if TRACKLET_MERGER
            //Save the pose at the current timestamp. Since the tracklet merger version outputs objects in the past,
            //we need to save the pose at that timestamp to make the camera -> world transformation, with the pose at that time.
            unsigned long long ts = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE).data_ns;
            ingestPoseInMap(zed.getTimestamp(sl::TIME_REFERENCE::IMAGE),cam_pose,trajectory_parameters.batch_duration*2);
            if (init_app_ts.data_ns==0ULL)
                init_app_ts =  zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);

#endif

            // Get a protected copy of the vector of objects
            guard.lock();
            std::vector<sl::ObjectData> object_data_list = objects.object_list;
            guard.unlock();

            // Update the 3D window with point cloud and 3D boxes.
            viewer.updateData(point_cloud,object_data_list, cam_pose.pose_data);
            // as image_left_ocv is a ref of image_left, it contains directly the new grabbed image
            render_2D(image_left_ocv, img_scale, object_data_list, true);

            // update birds view of tracks based on camera position and detected objects
#if !TRACKLET_MERGER
            track_view_generator.generate_view(objects, cam_pose, image_track_ocv, objects.is_tracked);
#endif
#else
            cout << "Detected " << objects.object_list.size() << " Object(s)" << endl;
#endif
        }

#if TRACKLET_MERGER
        std::vector<sl::Trajectory> trajectories;
        zed.retrieveBatchTrajectories(trajectories);
        ingestInObjectsQueue(trajectories);
        if (objects_tracked_queue.size()>0)
        {
            sl::Objects tracked_merged_obj = objects_tracked_queue.front();
            do
            {
                if (init_queue_ts.data_ns==0ULL)
                {
                    init_queue_ts = tracked_merged_obj.timestamp;
                    break;
                }
                else
                {
                    //Compare delay between live. Keep it close to live with batch_duration.
                    // The queue contains data between [live - 2*batch_duration , live - batch_duration], therefore make sure the delay is not higher than live - 2*batch_duration
                    unsigned long long delay_queue_ms = tracked_merged_obj.timestamp.getMilliseconds() - init_queue_ts.getMilliseconds();
                    unsigned long long delay_app_ms = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE).getMilliseconds() - init_app_ts.getMilliseconds();
                    if (delay_app_ms-trajectory_parameters.batch_duration*2*1000>0 && delay_queue_ms<delay_app_ms-trajectory_parameters.batch_duration*2.0*1000 && objects_tracked_queue.size()>1){
                        objects_tracked_queue.pop_front();
                        tracked_merged_obj = objects_tracked_queue.front();
                    }
                    else
                        break;
                }
            } while(trajectory_parameters.resampling_rate!=0);
            track_view_generator.generate_view(tracked_merged_obj, findClosestPoseFromTS(tracked_merged_obj.timestamp.getMilliseconds()), image_track_ocv, tracked_merged_obj.is_tracked);
            objects_tracked_queue.pop_front();
        }
#endif

#if ENABLE_GUI
        gl_viewer_available = viewer.isAvailable();
        // as image_left_ocv and image_track_ocv are both ref of global_image, no need to update it
        cv::imshow(window_name, global_image);
        key = cv::waitKey(5);
        if (key == 'i') {
            track_view_generator.zoomIn();
        } else if (key == 'o') {
            track_view_generator.zoomOut();
        } else if (key == 'q') {
            quit = true;
        } else if (key == 'p') {
            detection_parameters_rt.object_class_filter.clear();
            detection_parameters_rt.object_class_filter.push_back(OBJECT_CLASS::PERSON);
            detection_parameters_rt.object_class_detection_confidence_threshold[OBJECT_CLASS::PERSON] = detection_confidence;
            cout << "Person only" << endl;
        } else if (key == 'v') {
            detection_parameters_rt.object_class_filter.clear();
            detection_parameters_rt.object_class_filter.push_back(OBJECT_CLASS::VEHICLE);
            detection_parameters_rt.object_class_detection_confidence_threshold[OBJECT_CLASS::VEHICLE] = detection_confidence;
            cout << "Vehicle only" << endl;
        } else if (key == 'c') {
            detection_parameters_rt.object_class_filter.clear();
            detection_parameters_rt.object_class_detection_confidence_threshold.clear();
            cout << "Clear Filters" << endl;
        }
#endif
    }

    if (runner.joinable())
        runner.join();

#if ENABLE_GUI
    viewer.exit();
    point_cloud.free();
    image_left.free();
#endif
    zed.disableObjectDetection();
    zed.close();
    return EXIT_SUCCESS;
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "[Sample] ";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}

void parseArgs(int argc, char **argv, InitParameters& param) {
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        is_playback = true;
        cout << "[Sample] Using SVO File input: " << argv[1] << endl;
    } else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        } else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        } else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }
    }
}
