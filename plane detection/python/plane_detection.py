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
    This sample shows how to detect planes in a 3D scene and
    displays it on an OpenGL window
"""
import sys
import pyzed.sl as sl
import time
import ogl_viewer.viewer as gl

def main():
    print("Running Plane Detection sample ... Press 'q' to quit")

    # Create a camera object
    zed = sl.Camera()

    # Set configuration parameters
    init = sl.InitParameters()
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP     # OpenGL coordinate system
    
    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream    
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Reading SVO file: {0}".format(filepath))
        init.set_from_svo_file(filepath)

    # Open the camera
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

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
                    image_click = [user_action.hit_coord[0] * camera_infos.camera_configuration.camera_resolution.width
                                , user_action.hit_coord[1] * camera_infos.camera_configuration.camera_resolution.height]
                    find_plane_status = zed.find_plane_at_hit(image_click, plane)

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

if __name__ == "__main__":
    main()
