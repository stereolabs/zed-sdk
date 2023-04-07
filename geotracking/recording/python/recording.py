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
import time
import datetime
import multiprocessing
import threading
import json
import exporter.KMLExporter as export

# optional library : GPSD
try:
    from gpsdclient import GPSDClient
    with_gpsd = True
except ImportError:
    with_gpsd = False
    print ("GPSD not detected. GNSS data will be dummy instead.")

global current_gps_value
global gps_stream
global new_gps_data_available
global running

# this loop will run and either retrieve GNSS data from GPSD or no data.
def gps_loop(output_path : str):
    global current_gps_value
    global gps_stream
    global running

    gps_data = []

    while running:
        current_gps_value = next(gps_stream)
        gnss_measure = {}
        gnss_measure["coordinates"] = {}
        eph = 1
        epv = 1
        if(with_gpsd):
            gnss_measure["coordinates"]["latitude"] = current_gps_value.get("lat", "n/a")
            gnss_measure["coordinates"]["longitude"] = current_gps_value.get("lon", "n/a")
            gnss_measure["coordinates"]["altitude"] = current_gps_value.get("alt", "n/a")
            eph = current_gps_value.get("eph", "n/a")
            epv = current_gps_value.get("epv", "n/a")
        else:
            gnss_measure["coordinates"]["latitude"] = 0
            gnss_measure["coordinates"]["longitude"] = 0
            gnss_measure["coordinates"]["altitude"] = 0
        
        
        gnss_measure["longitude_std"] = eph
        gnss_measure["altitude_std"] = epv
        gnss_measure["latitude_std"] = eph
        gnss_measure["position_covariance"] = [eph * eph, 0, 0, 0, eph * eph, 0, 0, 0, epv * epv]
        ts = int(datetime.datetime.timestamp(current_gps_value.get("time", 0)) * 1000000) if with_gpsd else int(time.time() * 1e9)
        gnss_measure["ts"] = ts
        gps_data.append(gnss_measure)

        # Save raw GNSS data to KML
        latitude = current_gps_value.get("lat", "n/a") if with_gpsd else 0
        longitude = current_gps_value.get("lon", "n/a") if with_gpsd else 0
        altitude = current_gps_value.get("alt", "n/a") if with_gpsd else 0
        coordinates = {
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
        }
        export.saveKMLData("raw_gnss.kml", coordinates)
    
    gps_to_save = {}
    gps_to_save["GNSS"] = gps_data

    with open(output_path, 'w') as outfile:
        json.dump(gps_to_save, outfile)
    print("GPS data saved.")

if __name__ == "__main__":

    global current_gps_value
    global gps_stream
    global new_gps_data_available 

    new_gps_data_available = False

    # some variables
    camera_pose = sl.Pose()    
    odometry_pose = sl.Pose()    
    py_translation = sl.Translation()
    pose_data = sl.Transform()
    text_translation = ""
    text_rotation = ""   
    current_gps_value = None

    init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
                                 
    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.set_from_svo_file(filepath)

    # step 1
    # create the camera that will input the position from its odometry
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(status)
        exit()
    
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

    # step 2
    # init the fusion module that will input both the camera and the GPS
    fusion = sl.Fusion()
    init_fusion_parameters = sl.InitFusionParameters()
    init_fusion_parameters.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_fusion_parameters.coordinate_units = sl.UNIT.METER
    init_fusion_parameters.output_performance_metrics = False
    init_fusion_parameters.verbose = True
    fusion.init(init_fusion_parameters)
    fusion.enable_positionnal_tracking()
    uuid = sl.CameraIdentifier(camera_info.serial_number)
    print("Subscribing to", uuid.serial_number, communication_parameters.comm_type)
    status = fusion.subscribe(uuid, communication_parameters, sl.Transform(0,0,0))
    if status != sl.FUSION_ERROR_CODE.SUCCESS:
        print("Failed to subscribe to", uuid.serial_number, status)
        exit(1)

    # initialize the GNSS - gpsd https://gpsd.gitlab.io/gpsd/installation.html
    if with_gpsd:
        client = GPSDClient(host="127.0.0.1")
        gps_stream = client.dict_stream(convert_datetime=True, filter=["TPV"])
    else:
        dummy_longitude = 0

    dummy_longitude = 0

    gps_thread = threading.Thread(target=gps_loop)
    gps_thread.start()

    try:
        while (viewer.is_available()):
            # get the odometry information
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.get_position(odometry_pose, sl.REFERENCE_FRAME.WORLD)

            elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break

            dummy_longitude = dummy_longitude + 0.000001

            # GPS ingest
            if current_gps_value is not None and new_gps_data_available:
                time.sleep(0.01)
                gnss_data = sl.GNSSData()

                if with_gpsd:
                # retrieve the latest value from the GPS
                    longitude = current_gps_value.get("lon", "n/a")
                    # If you want the GNSS Fusion to work without moving your GNSS, uncomment this line.
                    # longitude = current_gps_value.get("lon", "n/a") + 20* dummy_longitude

                    latitude = current_gps_value.get("lat", "n/a")
                    altitude = current_gps_value.get("alt", "n/a")

                    # retrieve the timestamp and convert it to ZED SDK format
                    date = current_gps_value.get("time")
                    timestamp = sl.Timestamp()
                    timestamp.set_seconds(datetime.datetime.timestamp(date))

                    eph = current_gps_value.get("eph", "n/a")
                    epv = current_gps_value.get("epv", "n/a")                
                else:
                    # dummy_longitude = dummy_longitude + 0.000000001
                    longitude = dummy_longitude
                    latitude = 0
                    altitude = 0
                    timestamp = sl.get_current_timestamp()
                    eph = 1
                    epv = 1

                # put your GPS corrdinates here : latitude, longitude, altitude
                new_gps_data_available = False
                gnss_data.set_coordinates(latitude, longitude, altitude, False)
                gnss_data.ts = timestamp

                print(latitude, longitude, altitude)

                # put your covariance here if you know it, as an matrix 3x3 in a line
                # in this case
                # [eph * eph   0   0]
                # [0   eph * eph   0]
                # [0   0   epv * epv]
                covariance = [  
                                eph * eph,      0.1,  0.1,
                                0.1,      eph * eph,  0.1,
                                0.1,      0.1,      epv * epv
                            ]

                gnss_data.position_covariances = covariance
                fusion.ingest_gnss_data(gnss_data)

            # get the fused position
            if fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
                fused_tracking_state = fusion.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
                # you can also retrieve the un-fused position with
                # tracking_state = fusion.get_current_gnss_data(...)

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
                        
                        # Save computed path into KML
                        latitude, longitude, altitude = geopose.latlng_coordinates.get_coordinates(False)
                        coordinates = {
                            "latitude": latitude,
                            "longitude": longitude,
                            "altitude": altitude,
                        }
                        export.saveKMLData("computed_geoposition.kml", coordinates)
                                                
                        print(latitude, longitude, altitude)


    except KeyboardInterrupt:
        # got a ^C.  Say bye, bye
        print('Bye !')
        gps_thread.join()

    viewer.exit()
    zed.close()

