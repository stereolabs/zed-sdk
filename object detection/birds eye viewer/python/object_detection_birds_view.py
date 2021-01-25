########################################################################
#
# Copyright (c) 2021, STEREOLABS.
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
import ogl_viewer.viewer as gl
import pyzed.sl as sl

if __name__ == "__main__":
    print("Running object detection ... Press 'Esc' to quit")
    zed = sl.Camera()
    
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode    
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        
    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.set_from_svo_file(filepath)

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # Enable object detection module
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX
    # Defines if the object detection will track objects across images flow.
    obj_param.enable_tracking = False       # if True, enable positional tracking

    if obj_param.enable_tracking:
        zed.enable_positional_tracking()
        
    zed.enable_object_detection(obj_param)

    res = sl.Resolution()
    res.width = 720
    res.height = 404

    camera_model = zed.get_camera_information().camera_model
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_model, res)
    
    # Configure object detection runtime parameters
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 50

    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    objects = sl.Objects()

    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
            # Retrieve objects
            zed.retrieve_objects(objects, obj_runtime_param)
            viewer.updateData(point_cloud, objects)

    viewer.exit()
    
    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()

    zed.close()
