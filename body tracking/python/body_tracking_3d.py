########################################################################
#
# Copyright (c) 2020, STEREOLABS.
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
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import sys
import pyzed.sl as sl
import body_tracking.gl_viewer as gl
# import cv2
import numpy as np

# id_colors = [(59, 232, 176),
#              (25,175,208),
#              (105,102,205),
#              (255,185,0),
#              (252,99,107)]

# def get_color_id_gr(idx):
#     color_idx = idx % 5
#     arr = id_colors[color_idx]
#     return arr

def main():
    print("Running Body Tracking sample ... Press 'q' to quit")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.coordinate_units = sl.UNIT.METER         # Set coordinate units
    init_params.camera_fps = 15                          # Set fps at 15

    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.set_from_svo_file(filepath)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Set runtime parameters
    runtime_parameters = sl.RuntimeParameters()

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = true;
    # zed.enable_positional_tracking(positional_tracking_parameters)
    zed.enable_positional_tracking()
    
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True
    obj_param.enable_body_fitting = True        # Smooth skeleton moves
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST # sl.DETECTION.HUMAN_BODY_ACCURATE

    # Enable Object Detection module
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 50

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.calibration_parameters.left_cam)

    # Create ZED objects filled in the main loop
    bodies = sl.Objects()
    image = sl.Mat()

    # Floor plane handle
    floor_plane = sl.Plane()
    # Camera transform once floor plane is detected
    reset_from_floor_plane = sl.Transform()

    # Main Loop
    need_floor_plane = positional_tracking_parameters.set_as_static


    while viewer.is_available():
        # TODO if need_floor_plane
        # 
        # 

        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve objects
            zed.retrieve_objects(bodies, obj_runtime_param)
            # Update GL view
            viewer.update_view(image, bodies)        

    viewer.exit()

    image.free(memory_type=sl.MEM.CPU)
    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()

################################################################################# orig code
    # while key != 113: # for 'q' key
    #     # Grab an image, a RuntimeParameters object must be given to grab()
    #     if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    #         # A new image is available if grab() returns SUCCESS
    #         zed.retrieve_image(mat, sl.VIEW.LEFT)
    #         zed.retrieve_objects(objects, obj_runtime_param)
    #         obj_array = objects.object_list
    #         image_data = mat.get_data()
    #         for i in range(len(obj_array)) :
    #             obj_data = obj_array[i]
    #             bounding_box = obj_data.bounding_box_2d
    #             cv2.rectangle(image_data, (int(bounding_box[0,0]),int(bounding_box[0,1])),
    #                         (int(bounding_box[2,0]),int(bounding_box[2,1])),
    #                           get_color_id_gr(int(obj_data.id)), 3)

    #             keypoint = obj_data.keypoint_2d
    #             for kp in keypoint:
    #                 if kp[0] > 0 and kp[1] > 0:
    #                     cv2.circle(image_data, (int(kp[0]), int(kp[1])), 3, get_color_id_gr(int(obj_data.id)), -1)
                
    #             for bone in sl.BODY_BONES:
    #                 kp1 = keypoint[bone[0].value]
    #                 kp2 = keypoint[bone[1].value]
    #                 if kp1[0] > 0 and kp1[1] > 0 and kp2[0] > 0 and kp2[1] > 0 :
    #                     cv2.line(image_data, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), get_color_id_gr(int(obj_data.id)), 2)

    #         cv2.imshow("ZED", image_data)
    #     key = cv2.waitKey(5)

    # cv2.destroyAllWindows()

    # # Close the camera
    # zed.close()

if __name__ == "__main__":
    main()
