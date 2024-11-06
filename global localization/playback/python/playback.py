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
    This sample shows how to track the position of the ZED camera 
    and displays it in a OpenGL window.
"""

import sys
from display.generic_display import GenericDisplay
from gnss_replay import GNSSReplay
import pyzed.sl as sl
import json
import exporter.KMLExporter as export
import argparse
import cv2


def main():
    zed_pose = sl.Pose()
    py_translation = sl.Translation()
    text_translation = ""
    text_rotation = ""

    init_params = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                                    coordinate_units=sl.UNIT.METER,
                                    coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

    init_params.set_from_svo_file(opt.input_svo_file)

    # create the camera that will input the position from its odometry
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("[ZED][ERROR] Camera Open : " + repr(status) + ". Exit program.")
        exit()
    # Enable positional tracking:
    positional_init = zed.enable_positional_tracking()
    if positional_init != sl.ERROR_CODE.SUCCESS:
        print("[ZED][ERROR] Can't start tracking of camera : " + repr(status) + ". Exit program.")
        exit()

    # Display
    display_resolution = sl.Resolution(1280, 720)
    left_img = sl.Mat()

    # Create Fusion object:

    fusion = sl.Fusion()
    init_fusion_param = sl.InitFusionParameters()
    init_fusion_param.coordinate_units = sl.UNIT.METER
    init_fusion_param.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_fusion_param.verbose = True

    fusion_init_code = fusion.init(init_fusion_param)
    if fusion_init_code != sl.FUSION_ERROR_CODE.SUCCESS:
        print("[ZED][ERROR] Failed to initialize fusion :" + repr(fusion_init_code) + ". Exit program")
        exit()

    # Enable odometry publishing:
    configuration = sl.CommunicationParameters()
    zed.start_publishing(configuration)

    uuid = sl.CameraIdentifier(zed.get_camera_information().serial_number)
    fusion.subscribe(uuid, configuration, sl.Transform(0, 0, 0))

    # Enable positional tracking for Fusion object
    positional_tracking_fusion_parameters = sl.PositionalTrackingFusionParameters()
    positional_tracking_fusion_parameters.enable_GNSS_fusion = True
    gnss_calibration_parameters = sl.GNSSCalibrationParameters()
    gnss_calibration_parameters.target_yaw_uncertainty = 7e-3
    gnss_calibration_parameters.enable_translation_uncertainty_target = False
    gnss_calibration_parameters.target_translation_uncertainty = 15e-2
    gnss_calibration_parameters.enable_reinitialization = False
    gnss_calibration_parameters.gnss_vio_reinit_threshold = 5
    positional_tracking_fusion_parameters.gnss_calibration_parameters = gnss_calibration_parameters

    fusion.enable_positionnal_tracking(positional_tracking_fusion_parameters)

    # Setup viewer:
    py_translation = sl.Translation()
    # Setup viewer:
    viewer = GenericDisplay()
    viewer.init(zed.get_camera_information().camera_model)
    print("Start grabbing data ...")

    gnss_replay = GNSSReplay(opt.input_json_gps_file, zed)
    input_gnss_sync = sl.GNSSData()

    while viewer.isAvailable():
        # get the odometry information
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
            zed.retrieve_image(left_img, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            cv2.imshow("left", left_img.numpy())
            cv2.waitKey(10)

        elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("End of SVO file.")
            fusion.close()
            zed.close()
            exit()
        status, input_gnss = gnss_replay.grab(zed_pose.timestamp.get_nanoseconds())
        if status == sl.FUSION_ERROR_CODE.SUCCESS:
            ingest_error = fusion.ingest_gnss_data(input_gnss)
            latitude, longitude, altitude = input_gnss.get_coordinates(False)
            coordinates = {
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
            }
            export.saveKMLData("raw_gnss.kml", coordinates)
            # Fusion is asynchronous and needs synchronization. Sometime GNSSData comes before camera data raising "NO_NEW_DATA_AVAILABLE" error
            # This does not necessary means that fusion doesn't work but that no camera data were presents for the gnss timestamp when you ingested the data.
            if ingest_error != sl.FUSION_ERROR_CODE.SUCCESS and ingest_error != sl.FUSION_ERROR_CODE.NO_NEW_DATA_AVAILABLE:
                print("Ingest error occurred when ingesting GNSSData: ", ingest_error)

        # get the fused position
        if fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            fused_position = sl.Pose()
            # Get position into the ZED CAMERA coordinate system:
            current_state = fusion.get_position(fused_position)
            if current_state == sl.POSITIONAL_TRACKING_STATE.OK:
                current_state = fusion.get_fused_positional_tracking_status().tracking_fusion_status
                rotation = fused_position.get_rotation_vector()
                translation = fused_position.get_translation(py_translation)
                text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                text_translation = str(
                    (round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))
                viewer.updatePoseData(fused_position.pose_data(), text_translation, text_rotation, current_state)

            fusion.get_current_gnss_data(input_gnss_sync)
            # Display it on the Live Server
            viewer.updateRawGeoPoseData(input_gnss_sync)

            # Get position into the GNSS coordinate system - this needs a initialization between CAMERA 
            # and GNSS. When the initialization is finish the getGeoPose will return sl::POSITIONAL_TRACKING_STATE::OK
            current_geopose = sl.GeoPose()
            current_geopose_satus = fusion.get_geo_pose(current_geopose)
            if current_geopose_satus == sl.GNSS_FUSION_STATUS.OK:
                # Display it on the Live Server
                viewer.updateGeoPoseData(current_geopose)
                _, yaw_std, position_std = fusion.get_current_gnss_calibration_std()
                if yaw_std != -1:
                    print("GNSS State : ", current_geopose_satus, " : calibration uncertainty yaw_std ", yaw_std,
                          " rd position_std", position_std[0], " m,", position_std[1], " m,", position_std[2],
                          end=' m\r')

            """
            else : 
                GNSS coordinate system to ZED coordinate system is not initialized yet
                The initialization between the coordinates system is basically an optimization problem that
                Try to fit the ZED computed path with the GNSS computed path. In order to do it just move
                your system and wait that uncertainty come bellow uncertainty threshold you set up in your 
                initialization parameters.
            """
    fusion.close()
    zed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to a .svo file or .svo2 file containing gps data',
                        required=True)
    parser.add_argument('--input_json_gps_file', type=str, help='Path to a .json file, containing gps data',
                        required=False)
    opt = parser.parse_args()
    main()
