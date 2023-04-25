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
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import cv2
import sys
import pyzed.sl as sl
import time
import ogl_viewer.viewer as gl
import numpy as np
import json

def addIntoOutput(out, identifier, tab):
    out[identifier] = []
    for element in tab:
        out[identifier].append(element)
    return out

def serializeBodyData(body_data):
    """Serialize BodyData into a JSON like structure"""
    out = {}
    out["id"] = body_data.id
    out["unique_object_id"] = str(body_data.unique_object_id)
    out["tracking_state"] = str(body_data.tracking_state)
    out["action_state"] = str(body_data.action_state)
    addIntoOutput(out, "position", body_data.position)
    addIntoOutput(out, "velocity", body_data.velocity)
    addIntoOutput(out, "bounding_box_2d", body_data.bounding_box_2d)
    out["confidence"] = body_data.confidence
    addIntoOutput(out, "bounding_box", body_data.bounding_box)
    addIntoOutput(out, "dimensions", body_data.dimensions)
    addIntoOutput(out, "keypoint_2d", body_data.keypoint_2d)
    addIntoOutput(out, "keypoint", body_data.keypoint)
    addIntoOutput(out, "keypoint_cov", body_data.keypoints_covariance)
    addIntoOutput(out, "head_bounding_box_2d", body_data.head_bounding_box_2d)
    addIntoOutput(out, "head_bounding_box", body_data.head_bounding_box)
    addIntoOutput(out, "head_position", body_data.head_position)
    addIntoOutput(out, "keypoint_confidence", body_data.keypoint_confidence)
    addIntoOutput(out, "local_position_per_joint", body_data.local_position_per_joint)
    addIntoOutput(out, "local_orientation_per_joint", body_data.local_orientation_per_joint)
    addIntoOutput(out, "global_root_orientation", body_data.global_root_orientation)
    return out

def serializeBodies(bodies):
    """Serialize Bodies objects into a JSON like structure"""
    out = {}
    out["is_new"] = bodies.is_new
    out["is_tracked"] = bodies.is_tracked
    out["timestamp"] = bodies.timestamp.data_ns
    out["body_list"] = []
    for sk in bodies.body_list:
        out["body_list"].append(serializeBodyData(sk))
    return out

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":

    # common parameters
    init_params = sl.InitParameters()
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    zed = sl.Camera()
    error_code = zed.open(init_params)
    if(error_code != sl.ERROR_CODE.SUCCESS):
        print("Can't open camera: ", error_code)

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    error_code = zed.enable_positional_tracking(positional_tracking_parameters)
    if(error_code != sl.ERROR_CODE.SUCCESS):
        print("Can't enable positionnal tracking: ", error_code)

    body_tracking_parameters = sl.BodyTrackingParameters()
    body_tracking_parameters.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_18
    body_tracking_parameters.enable_body_fitting = False
    body_tracking_parameters.enable_tracking = True

    error_code = zed.enable_body_tracking(body_tracking_parameters)
    if(error_code != sl.ERROR_CODE.SUCCESS):
        print("Can't enable positionnal tracking: ", error_code)


    
    # Get ZED camera information
    camera_info = zed.get_camera_information()
    viewer = gl.GLViewer()
    viewer.init()

    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    # single_bodies = [sl.Bodies]

    skeleton_file_data = {}
    while (viewer.is_available()):
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_bodies(bodies)
            skeleton_file_data[str(bodies.timestamp.get_milliseconds())] = serializeBodies(bodies)
            viewer.update_bodies(bodies)

    # Save data into JSON file:
    file_sk = open("bodies.json", 'w')
    file_sk.write(json.dumps(skeleton_file_data, cls=NumpyEncoder, indent=4))
    file_sk.close()
        
    viewer.exit()