########################################################################
#
# Copyright (c) 2024, STEREOLABS.
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

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer


##
# Variable to enable/disable the batch option in Object Detection module
# Batching system allows to reconstruct trajectories from the object detection module by adding Re-Identification / Appareance matching.
# For example, if an object is not seen during some time, it can be re-ID to a previous ID if the matching score is high enough
# Use with caution if image retention is activated (See batch_system_handler.py) :
#   --> Images will only appear if an object is detected since the batching system is based on OD detection.
USE_BATCHING = False

if __name__ == "__main__":
    print("Running object detection ... Press 'Esc' to quit")
    zed = sl.Camera()
    
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.depth_maximum_distance = 20
    is_playback = False                             # Defines if an SVO is used
        
    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.svo_real_time_mode = True
        init_params.set_from_svo_file(filepath)
        is_playback = True

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()


    # Enable positional tracking module
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static in space, enabling this setting below provides better depth quality and faster computation
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    # Enable object detection module
    obj_param = sl.ObjectDetectionParameters()
    obj_param.instance_module_id = 0
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
    # Defines if the object detection will track objects across images flow.
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)



    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                # Track people across images flow
    body_param.enable_body_fitting = False            # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST 
    body_param.body_format = sl.BODY_FORMAT.BODY_18  # Choose the BODY_FORMAT you wish to use
    body_param.instance_module_id = 1
    zed.enable_body_tracking(body_param)


    camera_infos = zed.get_camera_information()
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_infos.camera_configuration.resolution.width, 720), min(camera_infos.camera_configuration.resolution.height, 404)) 
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    
    # Configure object detection runtime parameters
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    detection_confidence = 60
    obj_runtime_param.detection_confidence_threshold = detection_confidence
    # To select a set of specific object classes
    obj_runtime_param.object_class_filter = [sl.OBJECT_CLASS.PERSON]
    # To set a specific threshold
    obj_runtime_param.object_class_detection_confidence_threshold = {sl.OBJECT_CLASS.PERSON: detection_confidence} 

    # Runtime parameters
    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 50

    # Create objects that will store SDK outputs
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    objects = sl.Objects()

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40
    bodies = sl.Bodies()

    image_left = sl.Mat()

    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_infos.camera_configuration.resolution.width, 1280), min(camera_infos.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_infos.camera_configuration.resolution.width
                 , display_resolution.height / camera_infos.camera_configuration.resolution.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239,255], np.uint8)

    # Camera pose
    cam_w_pose = sl.Pose()
    cam_c_pose = sl.Pose()

    quit_app = False

    while(viewer.is_available() and (quit_app == False)):
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve objects
            returned_state = zed.retrieve_objects(objects, obj_runtime_param, obj_param.instance_module_id)
            returned_state2 = zed.retrieve_bodies(bodies, body_runtime_param, body_param.instance_module_id)
            
            # Retrieve image
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            image_render_left = image_left.get_data()            
            np.copyto(image_left_ocv,image_render_left)
            
            if (returned_state == sl.ERROR_CODE.SUCCESS and objects.is_new):
                # Retrieve point cloud
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, point_cloud_res)
                point_cloud.copy_to(point_cloud_render)
                # Retrieve image
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                image_render_left = image_left.get_data()
                # Get camera pose
                zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

                update_render_view = True
                update_3d_view = True

                # 3D rendering
                if update_3d_view:
                    viewer.updateData(point_cloud_render, objects)

                # 2D rendering
                if update_render_view:
                    cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)

            if (returned_state2 == sl.ERROR_CODE.SUCCESS and bodies.is_new):
                cv_viewer.render_2D_SK(image_left_ocv, image_scale, bodies.body_list, obj_param.enable_tracking, sl.BODY_FORMAT.BODY_18)
                
            cv2.imshow("ZED | Body tracking and Object detection", image_left_ocv)
            cv2.waitKey(10)

        if (is_playback and (zed.get_svo_position() == zed.get_svo_number_of_frames()-1)):
            print("End of SVO")
            quit_app = True


    cv2.destroyAllWindows()
    viewer.exit()
    image_left.free(sl.MEM.CPU)
    point_cloud.free(sl.MEM.CPU)
    point_cloud_render.free(sl.MEM.CPU)

    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()

    zed.close()