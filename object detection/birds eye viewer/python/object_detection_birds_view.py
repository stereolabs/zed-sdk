########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample demonstrates how to capture 3D point cloud and detected objects
    with the ZED SDK and display the result in an OpenGL window.
"""

import sys
import numpy as np
import cv2
import pyzed.sl as sl
import argparse
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import platform
from collections import deque


is_jetson = False

if platform.uname().machine.startswith('aarch64'):
    is_jetson = True

def main():
    zed = sl.Camera()
    
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.depth_maximum_distance = 10.0
    parse_args(init_params)
    
    is_playback = len(opt.input_svo_file)>0 # Defines if an SVO is used
        
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()
     # Enable positional tracking module
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static in space, enabling this setting below provides better depth quality and faster computation
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    # Enable object detection module
    batch_parameters = sl.BatchParameters()
    if opt.enable_batching_reid:
        batch_parameters.enable = True
        batch_parameters.latency = 3.0
    else:
        batch_parameters.enable = False
    obj_param = sl.ObjectDetectionParameters(batch_trajectories_parameters=batch_parameters)
        
    if is_jetson:
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
    else : 
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE
    # Defines if the object detection will track objects across images flow.
    obj_param.enable_tracking = True
    returned_state = zed.enable_object_detection(obj_param)
    camera_configuration = zed.get_camera_information().camera_configuration

    
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print("enable_object_detection", returned_state, "\nExit program.")
        zed.close()
        exit() 
    # Detection runtime parameters
    # default detection threshold, apply to all object class
    detection_confidence = 60
    detection_parameters_rt = sl.ObjectDetectionRuntimeParameters(detection_confidence)
    # To select a set of specific object classes:
    detection_parameters_rt.object_class_filter = [sl.OBJECT_CLASS.VEHICLE, sl.OBJECT_CLASS.PERSON]
    # To set a specific threshold
    detection_parameters_rt.object_class_detection_confidence_threshold[sl.OBJECT_CLASS.PERSON] = detection_confidence
    detection_parameters_rt.object_class_detection_confidence_threshold[sl.OBJECT_CLASS.VEHICLE] = detection_confidence


    quit_bool = False
    if not opt.disable_gui:
        
        image_aspect_ratio = camera_configuration.resolution.width/camera_configuration.resolution.height
        requested_low_res_w = min(1280, camera_configuration.resolution.width)
        
        display_resolution = sl.Resolution(requested_low_res_w, requested_low_res_w/image_aspect_ratio)
        image_scale = [display_resolution.width / camera_configuration.resolution.width
                    , display_resolution.height / camera_configuration.resolution.height]
        image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239,255], np.uint8)

        # Utilities for tracks view
        camera_config = zed.get_camera_information().camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)
        track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance*1000, batch_parameters.latency)
        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

        # Will store the 2D image and tracklet views 
        global_image = np.full((display_resolution.height, display_resolution.width+tracks_resolution.width, 4), [245, 239, 239,255], np.uint8)
        viewer = gl.GLViewer()
        pc_resolution = sl.Resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio) 
        viewer.init(zed.get_camera_information().camera_model, pc_resolution, obj_param.enable_tracking)
        point_cloud = sl.Mat(pc_resolution.width, pc_resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        objects = sl.Objects()
        image_left = sl.Mat()
        cam_w_pose = sl.Pose()
        image_scale = (display_resolution.width/camera_config.resolution.width,display_resolution.height/camera_config.resolution.height)
                
    id_counter = {}
    objects = sl.Objects() 
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    window_name = "ZED| 3D View tracking"
    gl_viewer_available = True
    printHelp()
    while 1:
        if not opt.disable_gui and ( zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS or quit_bool):
            break 
        if opt.disable_gui and (zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS or quit_bool or not gl_viewer_available):
            break 
        if len(detection_parameters_rt.object_class_filter) == 0:
            detection_parameters_rt.detection_confidence_threshold = detection_confidence
        else :  # if using class filter, set confidence for each class
            for parameter in detection_parameters_rt.object_class_filter:
                detection_parameters_rt.object_class_detection_confidence_threshold[parameter] = detection_confidence
        
        returned_state = zed.retrieve_objects(objects, detection_parameters_rt)
        if returned_state == sl.ERROR_CODE.SUCCESS:
            if opt.enable_batching_reid:
                for object in objects.object_list : 
                    id_counter[str(object.id)] = 1
                        
                #check if batched trajectories are available 
                objects_batch = [] 
                if zed.get_objects_batch(objects_batch) == sl.ERROR_CODE.SUCCESS:
                    if len(objects_batch)>0:
                        print("During last batch processing: ",len(id_counter)," Objets were detected: ", end=" ")
                        for it in id_counter:
                            print(it, end=" ")
                        print("\nWhile", len(objects_batch), "different only after reID:", end=" ")
                        for it in objects_batch:
                            print(it.id, end=" ")
                        print()
                        id_counter.clear()
            if not opt.disable_gui:
                
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, pc_resolution)
                zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                image_render_left = image_left.get_data()
                np.copyto(image_left_ocv,image_render_left)
                track_view_generator.generate_view(objects, image_left_ocv,image_scale ,cam_w_pose, image_track_ocv, objects.is_tracked)
                global_image = cv2.hconcat([image_left_ocv,image_track_ocv])
                viewer.updateData(point_cloud, objects)
                
                gl_viewer_available = viewer.is_available()
                
                cv2.imshow(window_name, global_image)
                key = cv2.waitKey(10)
                if key == 113: #for 'q' key 
                    quit_bool = True
                if key == 105: #for 'i' key 
                    track_view_generator.zoomIn()
                if key == 111 : #for 'o' key
                    track_view_generator.zoomOut() 
                elif key == 112: #for 'p' key
                    detection_parameters_rt.object_class_filter = [sl.OBJECT_CLASS.PERSON]
                    detection_parameters_rt.object_class_detection_confidence_threshold[sl.OBJECT_CLASS.PERSON] = detection_confidence
                    print("Person only")
                elif key == 118: #for 'v' key
                    detection_parameters_rt.object_class_filter = [sl.OBJECT_CLASS.VEHICLE]
                    detection_parameters_rt.object_class_filter.append(sl.OBJECT_CLASS.VEHICLE)
                    detection_parameters_rt.object_class_detection_confidence_threshold[sl.OBJECT_CLASS.VEHICLE] = detection_confidence
                    print("Vehicle only")
                elif key == 99: #for 'c' key
                    detection_parameters_rt.object_class_filter = []
                    detection_parameters_rt.object_class_detection_confidence_threshold.clear()
                    print("Clear Filters")
        if is_playback and zed.get_svo_position() == zed.get_svo_number_of_frames()-1:
            quit_bool = True        
    if not opt.disable_gui:
        viewer.exit()
        point_cloud.free()
        image_left.free()
    zed.disable_object_detection()
    zed.close()

    
    
    
def parse_args(init):
    if len(opt.input_svo_file)>0 and (opt.input_svo_file.endswith(".svo") or opt.input_svo_file.endswith(".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("resolution" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")
        
def printHelp():
    print("\n\n Birds eye view hotkeys:")
    print("* Zoom in tracking view            'i'")
    print("* Zoom out tracking view           'o'")
    print("* Filter Vehicule Only:            'v'")
    print("* Filter Person Only:              'p'")
    print("* Filter Vehicule Only:            'v'")
    print("* Clear Filters:                   'c'")
    print("* Exit:                            'q'")

       
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d.', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    parser.add_argument('--disable_gui', action = 'store_true', help='Flag to disable the GUI to increase detection performances. On low-end hardware such as Jetson Nano, the GUI significantly slows down the detection and increase the memory consumption')
    parser.add_argument('--enable_batching_reid', action = 'store_true', help='Flag to enable the batching re identification. Difference on ID tracking will be displayed in the console')
    opt = parser.parse_args()
    if (len(opt.input_svo_file)>0 and len(opt.ip_address)>0):
        print("Specify only input_svo_file or ip_address, not both. Exit program")
        exit()
    main() 
    
