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
    This sample shows how to record the position of the ZED camera 
    and displays it in a OpenGL window.
"""

from display.generic_display import GenericDisplay
import pyzed.sl as sl
from gnss_reader.gpsd_reader import GPSDReader
from exporter.gnss_saver import GNSSSaver, get_current_datetime
import exporter.KMLExporter as export
from signal import signal, SIGINT
import sys 



is_running = True 
#Handler to deal with CTRL+C properly
def handler(signal_received, frame):
    global is_running
    is_running = False 
    
signal(SIGINT, handler)

def main():
    global is_running
    # Open the camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 1 
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("[ZED][ERROR] Camera Open : "+repr(status)+". Exit program.")
        exit()
    # Enable positional tracking:
    positional_init = zed.enable_positional_tracking()
    if positional_init != sl.ERROR_CODE.SUCCESS:
        zed.close()
        print("[ZED][ERROR] Can't start tracking of camera : "+repr(status)+". Exit program.")
        exit()
    
    #Enable SVO recording :
    svo_path = "ZED_SN"+str(zed.get_camera_information().serial_number)+"_"+get_current_datetime()+".svo"
    returned_state = zed.enable_recording(sl.RecordingParameters(svo_path,sl.SVO_COMPRESSION_MODE.H264_LOSSLESS))
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print("Recording ZED : "+repr(status)+". Exit program.")
        zed.close()
        exit()
    
    # Create Fusion object:
    fusion = sl.Fusion()
    init_fusion_param = sl.InitFusionParameters()
    init_fusion_param.coordinate_units = sl.UNIT.METER
    fusion_init_code = fusion.init(init_fusion_param)
    if fusion_init_code != sl.FUSION_ERROR_CODE.SUCCESS:
        print("[ZED][ERROR] Failed to initialize fusion :"+repr(fusion_init_code)+". Exit program")
        exit()
    # Enable odometry publishing:
    configuration = sl.CommunicationParameters()
    configuration.set_for_shared_memory()
    zed.start_publishing(configuration)
    # Enable GNSS data producing:
    gnss_reader = GPSDReader()
    status_initialize = gnss_reader.initialize()
    if status_initialize==-1:
        gnss_reader.stop_thread()
        zed.close()
        exit()
    # Subscribe to Odometry
    uuid = sl.CameraIdentifier(zed.get_camera_information().serial_number)
    fusion.subscribe(uuid,configuration,sl.Transform(0,0,0))
    # Enable positional tracking for Fusion object
    positional_tracking_fusion_parameters = sl.PositionalTrackingFusionParameters()
    positional_tracking_fusion_parameters.enable_GNSS_fusion = True 
    gnss_calibration_parameters = {
        "target_yaw_uncertainty" : 0.1,
        "enable_translation_uncertainty_target" : False,
        "target_translation_uncertainty" : 10e-2,
        "enable_reinitialization" : True,
        "gnss_vio_reinit_threshold" : 5,
        "enable_rolling_calibration" : True
    }
    fusion.enable_positionnal_tracking(positional_tracking_fusion_parameters)
    # Setup viewer:
    viewer = GenericDisplay()
    viewer.init(zed.get_camera_information().camera_model)
    print("Start grabbing data ...")
    print("SVO is Recording, use Ctrl-C to stop.")
    py_translation = sl.Translation()
    gnss_data_saver = GNSSSaver(zed)
    while is_running and viewer.isAvailable():
        # Grab camera:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed_pose = sl.Pose()
            # You can still use the classical getPosition for your application, just not that the position returned by this method
            # is the position without any GNSS/cameras fusion
            zed.get_position(zed_pose, sl.REFERENCE_FRAME.CAMERA)
        # Get GNSS data:
        status, input_gnss = gnss_reader.grab()
        if status == sl.ERROR_CODE.SUCCESS:
            # Display it on the Live Server
            viewer.updateRawGeoPoseData(input_gnss)

            # Publish GNSS data to Fusion
            ingest_error = fusion.ingest_gnss_data(input_gnss)
            if ingest_error == sl.FUSION_ERROR_CODE.SUCCESS:
                # Save current GNSS data to KML file:
                latitude, longitude, altitude = input_gnss.get_coordinates(False)
                coordinates = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude": altitude,
                }
                export.saveKMLData("raw_gnss.kml", coordinates)
                # Save GNSS data into JSON:
                gnss_data_saver.addGNSSData(input_gnss)
            else:
                print("Ingest error occurred when ingesting GNSSData: ", ingest_error)

        # Process data and compute positions:
        if fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            fused_position = sl.Pose()
            # Get position into the ZED CAMERA coordinate system:
            current_state = fusion.get_position(fused_position)
            # Display it on OpenGL:
            rotation = fused_position.get_rotation_vector()
            translation = fused_position.get_translation(py_translation)
            text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
            text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))
            viewer.updatePoseData(fused_position.pose_data(),text_translation,text_rotation, current_state) 
            # Get position into the GNSS coordinate system - this needs a initialization between CAMERA 
            # and GNSS. When the initialization is finish the getGeoPose will return sl.POSITIONAL_TRACKING_STATE.OK
            current_geopose = sl.GeoPose()
            current_geopose_satus = fusion.get_geo_pose(current_geopose)
            if current_geopose_satus == sl.GNSS_FUSION_STATUS.OK:
                viewer.updateGeoPoseData(current_geopose)
            """
            else:
                GNSS coordinate system to ZED coordinate system is not initialized yet
                The initialization between the coordinates system is an optimization problem that
                tries to fit the ZED computed path with the GNSS computed path. In order to do it just move
                your system by the distance specified in positional_tracking_fusion_parameters.gnss_initialization_distance
            """
    gnss_reader.stop_thread()
    zed.disable_recording()
    gnss_data_saver.saveAllData()
    fusion.close()
    zed.close()
    
    
if __name__ == '__main__':
    main() 