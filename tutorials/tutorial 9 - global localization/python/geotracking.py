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
    This sample shows how to fuse the position of the ZED camera with an external GNSS Sensor
"""

import sys
import pyzed.sl as sl


if __name__ == "__main__":

    # some variables
    camera_pose = sl.Pose()    
    odometry_pose = sl.Pose()    
    py_translation = sl.Translation()
    pose_data = sl.Transform()
    text_translation = ""
    text_rotation = ""   

    # Create a ZED camera object
    
    init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.AUTO,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
                                 
    # step 1
    # create the camera that will input the position from its odometry
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open: " + repr(status) + ". Exit program.")
        exit()
    
    # set up communication parameters and start publishing
    communication_parameters = sl.CommunicationParameters()
    communication_parameters.set_for_shared_memory()
    zed.start_publishing(communication_parameters)

    # warmup for camera 
    if zed.grab() != sl.ERROR_CODE.SUCCESS:
        print("Camera grab: " + repr(status) + ". Exit program.")
        exit()
    else:
        zed.get_position(odometry_pose, sl.REFERENCE_FRAME.WORLD)

    tracking_params = sl.PositionalTrackingParameters()
    # These parameters are mandatory to initialize the transformation between GNSS and ZED reference frames.
    tracking_params.enable_imu_fusion = True
    tracking_params.set_gravity_as_origin = True
    err = zed.enable_positional_tracking(tracking_params)
    if (err != sl.ERROR_CODE.SUCCESS):
        print("Camera positional tracking: " + repr(status) + ". Exit program.")
        exit()
    camera_info = zed.get_camera_information()

    # step 2
    # init the fusion module that will input both the camera and the GPS
    fusion = sl.Fusion()
    init_fusion_parameters = sl.InitFusionParameters()
    init_fusion_parameters.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_fusion_parameters.coordinate_units = sl.UNIT.METER

    fusion.init(init_fusion_parameters)
    positional_tracking_fusion_parameters = sl.PositionalTrackingFusionParameters()
    fusion.enable_positionnal_tracking(positional_tracking_fusion_parameters)
    
    uuid = sl.CameraIdentifier(camera_info.serial_number)
    print("Subscribing to", uuid.serial_number, communication_parameters.comm_type) #Subscribe fusion to camera
    status = fusion.subscribe(uuid, communication_parameters, sl.Transform(0,0,0))
    if status != sl.FUSION_ERROR_CODE.SUCCESS:
        print("Failed to subscribe to", uuid.serial_number, status)
        exit(1)

    x = 0
    
    i = 0
    while i < 200:
        # get the odometry information
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.get_position(odometry_pose, sl.REFERENCE_FRAME.WORLD)

        elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break

        # dummy GPS value
        x = x + 0.000000001
        # get the GPS information
        gnss_data = sl.GNSSData()
        gnss_data.ts = sl.get_current_timestamp()

        # put your GPS coordinates here : latitude, longitude, altitude
        gnss_data.set_coordinates(x,0,0)

        # put your covariance here if you know it, as an matrix 3x3 in a line
        # This is the default value
        covariance = [  
                        1,0.1,0.1,
                        0.1,1,0.1,
                        0.1,0.1,1
                    ]

        gnss_data.position_covariances = covariance
        fusion.ingest_gnss_data(gnss_data)

        # get the fused position
        if fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            fused_tracking_state = fusion.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
            if fused_tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                
                rotation = camera_pose.get_rotation_vector()
                translation = camera_pose.get_translation(py_translation)
                text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))
                pose_data = camera_pose.pose_data(sl.Transform())
                print("get position translation  = ",text_translation,", rotation_message = ",text_rotation)
        i = i + 1

    zed.close()

