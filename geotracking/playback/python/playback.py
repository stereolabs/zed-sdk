########################################################################
#
# Copyright (c) 2023, STEREOLABS.
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
import ogl_viewer.tracking_viewer as gl
import pyzed.sl as sl
import json
import exporter.KMLExporter as export

# optional libraries : ZED HUB and GPSD
try:
    import pyzed.sl_iot as sliot
    with_zed_hub = True
except ImportError:
    with_zed_hub = False
    print("ZED Hub not detected.")

if __name__ == "__main__":
    if not sys.argv or len(sys.argv) != 2:
        print("This sample plays back a SVO file associated with a GNSS recording.")
        print("Only the path of the input SVO file should be passed as argument.")
        print("The JSON with the GPS data must have the same name, with the json extension.")
        print("You can generate it with the recording sample.")
        exit(1)

    filepath = sys.argv[1]

    # some variables
    camera_pose = sl.Pose()    
    odometry_pose = sl.Pose()    
    py_translation = sl.Translation()
    pose_data = sl.Transform()
    text_translation = ""
    text_rotation = ""   

    # connect to ZED Hub
    if with_zed_hub:                    
        sliot.HubClient.connect("Geotracking sample")

    init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
                                 

    gps_filepath = filepath.replace('.svo', '.json')           
    init_params.set_from_svo_file(filepath)

    # create the camera that will input the position from its odometry
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(status)
        exit()
    if with_zed_hub:                    
        status = sliot.HubClient.register_camera(zed)
        print("STATUS", status)

    communication_parameters = sl.CommunicationParameters()
    communication_parameters.set_for_shared_memory()
    zed.start_publishing(communication_parameters)

    # warmup
    if zed.grab() != sl.ERROR_CODE.SUCCESS:
        print("Unable to initialize the camera.")
        exit(1)
    else:
        zed.get_position(odometry_pose, sl.REFERENCE_FRAME.WORLD)

    tracking_params = sl.PositionalTrackingParameters()
    # These parameters are mandatory to initialize the transformation between GNSS and ZED reference frames.
    tracking_params.enable_imu_fusion = True
    tracking_params.set_gravity_as_origin = True
    zed.enable_positional_tracking(tracking_params)

    camera_info = zed.get_camera_information()
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_model)

    # init the fusion module that will input both the camera and the GPS
    fusion = sl.Fusion()
    init_fusion_parameters = sl.InitFusionParameters()
    init_fusion_parameters.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_fusion_parameters.coordinate_units = sl.UNIT.METER

    fusion.init(init_fusion_parameters)
    fusion.enable_positionnal_tracking()

    uuid = sl.CameraIdentifier(camera_info.serial_number)
    print("Subscribing to", uuid.serial_number, communication_parameters.comm_type)

    status = fusion.subscribe(uuid, communication_parameters, sl.Transform(0,0,0))
    if status != sl.FUSION_ERROR_CODE.SUCCESS:
        print("Failed to subscribe to", uuid.serial_number, status)
        exit(1)

    # read gps data
    with open(gps_filepath, 'r') as f:
        gps_data = json.load(f)
        gps_measures = gps_data["GNSS"]

    # we can ingest all the gps data at once
    # the fusion will use the timestamp to use them at the right time.
    for data in gps_measures:
        gnss_data = sl.GNSSData()
        ts = sl.Timestamp()
        ts.set_microseconds(data["ts"])
        gnss_data.ts = ts

        # put your GPS corrdinates here : latitude, longitude, altitude
        latitude = data["coordinates"]["latitude"]
        longitude = data["coordinates"]["longitude"]
        altitude = data["coordinates"]["altitude"]
        gnss_data.set_coordinates(latitude,longitude,altitude)

        eph = data["longitude_std"]
        epv = data["altitude_std"]
        
        covariance = [  
                        eph * eph,      0,  0,
                        0,      eph * eph,  0,
                        0,      0,      epv * epv
                    ]

        gnss_data.position_covariances = covariance
        fusion.ingest_gnss_data(gnss_data)

    while (viewer.is_available()):
        # get the odometry information
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.get_position(odometry_pose, sl.REFERENCE_FRAME.WORLD)

        elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("End of SVO file.")
            fusion.close()
            viewer.exit()
            zed.close()
            exit(0)

        # get the fused position
        if fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            fused_tracking_state = fusion.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
            if fused_tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                
                rotation = camera_pose.get_rotation_vector()
                translation = camera_pose.get_translation(py_translation)
                text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))
                pose_data = camera_pose.pose_data(sl.Transform())

                viewer.updateData(pose_data, text_translation, text_rotation, fused_tracking_state)
                # send data to zed hub
                # visualize it on https://hub.stereolabs.com/workspaces/<workspace_id>/maps
                geopose = sl.GeoPose()
                status = fusion.camera_to_geo(camera_pose, geopose)
                
                # the fusion will stay in SEARCHING mode until the GNSS has walked at least 5 meters.
                if status != sl.POSITIONAL_TRACKING_STATE.OK:
                    print(status)
                else:
                    print("OK")
                    latitude, longitude, altitude = geopose.latlng_coordinates.get_coordinates(False)
                    if with_zed_hub:                    
                        gps = {}
                        gps["layer_type"] = "geolocation"
                        gps["label"] = "GPS_data"
                        gps["position"] = {}
                        gps["position"]["latitude"] = latitude
                        gps["position"]["longitude"] = longitude
                        gps["position"]["altitude"] = altitude
                        status = sliot.HubClient.send_data_to_peers("geolocation", json.dumps(gps))
                    else:
                        # Save computed path into KML
                        latitude, longitude, altitude = geopose.latlng_coordinates.get_coordinates(False)
                        coordinates = {
                            "latitude": latitude,
                            "longitude": longitude,
                            "altitude": altitude,
                        }
                        export.saveKMLData("computed_geoposition.kml", coordinates)
                    
                    print(latitude, longitude, altitude)

            if with_zed_hub:
                sliot.HubClient.update()


    viewer.exit()
    zed.close()

