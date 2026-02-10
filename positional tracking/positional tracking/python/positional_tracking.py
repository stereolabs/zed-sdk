###########################################################################
#
# Copyright (c) 2025, STEREOLABS.
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
###########################################################################

# ZED imports
import pyzed.sl as sl

# Sample imports
import sys
import time
from typing import Optional
import cv2
from utilities.utilities import *
from slam_view.slam_view import SLAMView

#
# Arguments
#
class Arguments:
    def __init__(self):
        self.resolution: Optional[sl.RESOLUTION] = None
        self.svo_file: Optional[str] = None
        self.stream_ip: Optional[str] = None
        self.stream_port: Optional[int] = None
        self.map: bool = False
        self.input_area_file: Optional[str] = None
        self.output_area_file: Optional[str] = None
        self.roi_file: Optional[str] = None
        self.custom_initial_pose: bool = False
        self.enable_2d_ground_mode: bool = False
        self.export_tum_file: bool = False

#
# Main
#
def main(args: Arguments) -> int:
    #
    # Configure a camera
    #
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.camera_disable_self_calib = True

    if args.resolution:
        init_params.camera_resolution = args.resolution

    if args.svo_file:
        init_params.set_from_svo_file(args.svo_file)
        init_params.svo_real_time_mode = True
    elif args.stream_ip and args.stream_port:
        init_params.set_from_stream(args.stream_ip, args.stream_port)
    elif args.stream_ip:
        init_params.set_from_stream(args.stream_ip)

    #
    # Open the camera
    #
    status = zed.open(init_params)
    if status > sl.ERROR_CODE.SUCCESS:
        sample_print("Failed to open the camera", status)
        zed.close()
        return 1

    #
    # Configure positional tracking
    #
    tracking_parameters = sl.PositionalTrackingParameters()

    if args.input_area_file:
        tracking_parameters.area_file_path = args.input_area_file
    
    tracking_parameters.set_floor_as_origin = False
    tracking_parameters.set_gravity_as_origin = True
    tracking_parameters.enable_area_memory = True
    tracking_parameters.enable_pose_smoothing = False
    tracking_parameters.enable_imu_fusion = True
    tracking_parameters.set_as_static = False
    tracking_parameters.depth_min_range = -1
    tracking_parameters.enable_2d_ground_mode = args.enable_2d_ground_mode
    tracking_parameters.enable_localization_only = False
    tracking_parameters.mode = sl.POSITIONAL_TRACKING_MODE.GEN_3

    if args.roi_file is not None:
        roi = sl.Mat()
        roi.read(args.roi_file)
        zed.set_region_of_interest(roi)

    #
    # Optionally define an initial pose for the tracking to start from.
    # This is useful if you know the initial pose of the camera in a previously mapped area.
    #
    if args.custom_initial_pose:
        # Homogeneous transformation matrix for the initial pose:
        # For example, we have a translation to (2.0, 4.0, 1.0) in the right-most column,
        # and a rotation of 45° about the Z-axis in the upper-left 3x3 portion of the matrix.
        initial_pose = [0.7071, -0.7071, 0.0, 2.0,
                        0.7071,  0.7071, 0.0, 4.0,
                        0.0,     0.0,    1.0, 1.0,
                        0.0,     0.0,    0.0, 1.0]
        
        tracking_parameters.set_initial_world_transform(sl.Transform(initial_pose))

    #
    # Enable positional tracking
    #
    status = zed.enable_positional_tracking(tracking_parameters)

    if status > sl.ERROR_CODE.SUCCESS:
        print(f"Failed to enable positional tracking: {status}")
        zed.close()
        return 1

    print_tracking_parameters(tracking_parameters)

    #
    # Set runtime parameters: for example, a low depth confidence to avoid introducing noise
    #
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 30

    #
    # Configure display parameters
    #
    landmark_map = {}
    landmarks_2d = []

    last_landmark_update = sl.get_current_timestamp().get_seconds()
    display_resolution = zed.get_retrieve_measure_resolution()

    #
    # Prepare out file for TUM trajectory.
    # TUM format useful for benchmarking SLAM trajectories.
    # Tools like evo (https://github.com/sbesson/evo) used it for evaluation and visualization.
    #
    out_tum = None
    if args.export_tum_file:
        out_tum = open("out.tum", mode="w", encoding="utf-8", buffering=1)

    #
    # Main loop
    #
    left_image = sl.Mat()
    point_cloud = sl.Mat(display_resolution.width, display_resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    pose = sl.Pose()
    view = SLAMView(left_image, point_cloud)

    def render_loop():
        nonlocal last_landmark_update

        #
        # Grab the next image frame
        #
        status = zed.grab(runtime_parameters)

        if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            view.stop()
            return
        elif status > sl.ERROR_CODE.SUCCESS:
            print("Failed to grab image frame", status)
            view.stop()
            return
       
        # Retrieve the left image
        if(tracking_parameters.mode == sl.POSITIONAL_TRACKING_MODE.GEN_3):
            zed.retrieve_image(left_image, sl.VIEW.LEFT_UNRECTIFIED, sl.MEM.CPU, display_resolution)
        else:
            zed.retrieve_image(left_image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)

        # Retrieve the calculated point cloud
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZBGRA, sl.MEM.CPU, display_resolution)

        # Retrieve the calculated camera pose
        zed.get_position(pose)

        # Export the pose in TUM format
        if out_tum is not None:
            out_tum.write(f"{pose.timestamp.get_milliseconds()} ")
            out_tum.write(f"{pose.get_translation().get()[0]:.9f} ")
            out_tum.write(f"{pose.get_translation().get()[1]:.9f} ")
            out_tum.write(f"{pose.get_translation().get()[2]:.9f} ")
            out_tum.write(f"{pose.get_orientation().get()[0]:.9f} ")
            out_tum.write(f"{pose.get_orientation().get()[1]:.9f} ")
            out_tum.write(f"{pose.get_orientation().get()[2]:.9f} ")
            out_tum.write(f"{pose.get_orientation().get()[3]:.9f}\n")
            out_tum.flush()

        #
        # Update display
        #
        view.update_pose_transform(pose.pose_data())
        view.update_positional_tracking_status(zed.get_positional_tracking_status())

        if zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_seconds() - last_landmark_update > 1:
            zed.get_positional_tracking_landmarks(landmark_map)
            view.update_landmarks(landmark_map)
            last_landmark_update = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_seconds()

        if view.is_landmark_mode_enabled and len(landmark_map) > 0:
            zed.get_positional_tracking_landmarks2d(landmarks_2d)

            inlier_color = (63, 255, 67, 255)
            outlier_color = (100, 100, 255, 255)

            resolution = zed.get_camera_information().camera_configuration.resolution
            width_ratio = display_resolution.width / resolution.width
            height_ratio = display_resolution.height / resolution.height

            left_image_cv_mat = left_image.get_data()

            for landmark_2d in landmarks_2d:
                color = interpolate_color(inlier_color, outlier_color, 1 - landmark_2d.dynamic_confidence)
                
                cv2.circle(left_image_cv_mat, (
                    int(landmark_2d.position[0] * width_ratio),
                    int(landmark_2d.position[1] * height_ratio)
                ), 2, color, -1)

    view.run(render_loop)

    if out_tum is not None:
        out_tum.close()

    #
    # Saving area map
    #
    if args.map:
        sample_print(f"Saving area map to {args.output_area_file} ...")
        assert args.output_area_file is not None

        status = zed.save_area_map(args.output_area_file)

        if status == sl.ERROR_CODE.SUCCESS:
            export_state = zed.get_area_export_state()

            while export_state == sl.AREA_EXPORTING_STATE.RUNNING:
                time.sleep(0.01)
                export_state = zed.get_area_export_state()

            if export_state == sl.AREA_EXPORTING_STATE.SUCCESS:
                sample_print(f"Successfully saved area map to {args.output_area_file}")
            else:
                sample_print(f"Failed to save area map: {str(export_state)}", sl.ERROR_CODE.FAILURE, show_error_detail=False)
        else:
            sample_print("Failed to save area map", status)

    # Close the camera
    zed.close()

    return 0

#
# Main entry
#
if __name__ == "__main__":
    print_header("ZED Positional Tracking")

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print_usage(sys.argv[0])
        sys.exit()

    args = Arguments()
    if not parse_args(sys.argv, args):
        print()
        print_usage(sys.argv[0])
        sys.exit()

    print_args(args)

    exit(main(args))
