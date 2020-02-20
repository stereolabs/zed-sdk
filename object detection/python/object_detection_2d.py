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

import pyzed.sl as sl
import cv2
import numpy as np

id_colors = [(59, 232, 176),
             (25,175,208),
             (105,102,205),
             (255,185,0),
             (252,99,107)]

def get_color_id_gr(idx):
    color_idx = idx % 5
    arr = id_colors[color_idx]
    return arr

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.camera_fps = 15  # Set fps at 15

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Capture 50 frames and stop
    mat = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    key = ''

    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = False

    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    while key != 113: # for 'q' key
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            zed.retrieve_objects(objects, obj_runtime_param)
            obj_array = objects.object_list
            image_data = mat.get_data()
            for i in range(len(obj_array)) :
                obj_data = obj_array[i]
                bounding_box = obj_data.bounding_box_2d
                cv2.rectangle(image_data, (int(bounding_box[0,0]),int(bounding_box[0,1])),
                            (int(bounding_box[2,0]),int(bounding_box[2,1])),
                              get_color_id_gr(int(obj_data.id)), 3)

            cv2.imshow("ZED", image_data)
        key = cv2.waitKey(5)

    cv2.destroyAllWindows()

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
