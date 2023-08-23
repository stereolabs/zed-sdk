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
    This sample shows how to detect planes in a 3D scene and
    displays it on an OpenGL window
"""
import sys
import pyzed.sl as sl
import time
import ogl_viewer.viewer as gl
import argparse



def main():
    init = sl.InitParameters()
    parse_args(init)
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # OpenGL coordinate system
    # Create a camera object
    zed = sl.Camera()
    # Open the camera
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()
    # Get camera info and check if IMU data is available
    camera_infos = zed.get_camera_information()
    has_imu =  camera_infos.sensors_configuration.gyroscope_parameters.is_available

    # Initialize OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_infos.camera_configuration.calibration_parameters.left_cam, has_imu)

    image = sl.Mat()    # current left image
    pose = sl.Pose()    # positional tracking data
    plane = sl.Plane()  # detected plane 
    mesh = sl.Mesh()    # plane mesh

    find_plane_status = sl.ERROR_CODE.SUCCESS
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF

    # Timestamp of the last mesh request
    last_call = time.time()

    user_action = gl.UserAction()
    user_action.clear()

    # Enable positional tracking before starting spatial mapping
    zed.enable_positional_tracking()

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

    # the plane detection parameters can be change
    plane_parameters = sl.PlaneDetectionParameters()

    while viewer.is_available():
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Update pose data (used for projection of the mesh over the current image)
            tracking_state = zed.get_position(pose)

            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                # Compute elapse time since the last call of plane detection
                duration = time.time() - last_call  
                # Ask for a mesh update on mouse click
                if user_action.hit:
                    image_click = [user_action.hit_coord[0] * camera_infos.camera_configuration.resolution.width
                                , user_action.hit_coord[1] * camera_infos.camera_configuration.resolution.height]
                    find_plane_status = zed.find_plane_at_hit(image_click, plane, plane_parameters)

                # Check if 500 ms have elapsed since last mesh request
                if duration > .5 and user_action.press_space:
                    # Update pose data (used for projection of the mesh over the current image)
                    reset_tracking_floor_frame = sl.Transform()
                    find_plane_status = zed.find_floor_plane(plane, reset_tracking_floor_frame)
                    last_call = time.time()

                if find_plane_status == sl.ERROR_CODE.SUCCESS:
                    mesh = plane.extract_mesh()
                    viewer.update_mesh(mesh, plane.type)

            user_action = viewer.update_view(image, pose.pose_data(), tracking_state)
    
    viewer.exit()
    image.free(sl.MEM.CPU)
    mesh.clear()

    # Disable modules and close camera
    zed.disable_positional_tracking()
    zed.close()

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
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d.', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if (len(opt.input_svo_file)>0 and len(opt.ip_address)>0):
        print("Specify only input_svo_file or ip_address, not both. Exit program")
        exit()
    main() 
    
