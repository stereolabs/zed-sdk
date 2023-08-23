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
    This sample shows how to track the position of the ZED camera 
    and displays it in a OpenGL window.
"""

import sys
import ogl_viewer.tracking_viewer as gl
import pyzed.sl as sl
import argparse
import time 


def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("resolution" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")
        
def main():
    init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    parse_args(init_params)                              
   
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open", status, "Exit program.")
        exit(1)

    tracking_params = sl.PositionalTrackingParameters() #set parameters for Positional Tracking
    tracking_params.enable_imu_fusion = True 
    status = zed.enable_positional_tracking(tracking_params) #enable Positional Tracking
    if status != sl.ERROR_CODE.SUCCESS:
        print("Enable Positional Tracking : "+repr(status)+". Exit program.")
        zed.close()
        exit()

    runtime = sl.RuntimeParameters()
    camera_pose = sl.Pose()

    camera_info = zed.get_camera_information()
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_model)
    if opt.imu_only:
        sensors_data = sl.SensorsData()
    py_translation = sl.Translation()
    pose_data = sl.Transform()

    text_translation = ""
    text_rotation = ""
    file = open('output_trajectory.csv', 'w')
    file.write('tx, ty, tz \n')
    while viewer.is_available():
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            tracking_state = zed.get_position(camera_pose,sl.REFERENCE_FRAME.WORLD) #Get the position of the camera in a fixed reference frame (the World Frame)
            if opt.imu_only :
                if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
                    rotation = sensors_data.get_imu_data().get_pose().get_euler_angles()
                    text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                    viewer.updateData(sensors_data.get_imu_data().get_pose(), text_translation, text_rotation, tracking_state)
            else : 
                if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    #Get rotation and translation and displays it
                    rotation = camera_pose.get_rotation_vector()
                    translation = camera_pose.get_translation(py_translation)
                    text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                    text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))
                    pose_data = camera_pose.pose_data(sl.Transform())
                    file.write(str(translation.get()[0])+", "+str(translation.get()[1])+", "+str(translation.get()[2])+"\n")
                # Update rotation, translation and tracking state values in the OpenGL window
                viewer.updateData(pose_data, text_translation, text_rotation, tracking_state)
        else : 
            time.sleep(0.001)
    viewer.exit()
    zed.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    parser.add_argument('--imu_only', action = 'store_true', help = 'Either the tracking should be done with imu data only (that will remove translation estimation)' )
    opt = parser.parse_args()
    if (len(opt.input_svo_file)>0 and len(opt.ip_address)>0):
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 
    