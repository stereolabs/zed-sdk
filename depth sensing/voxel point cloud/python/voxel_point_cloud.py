########################################################################
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
########################################################################

"""
Voxel Point Cloud sample — demonstrates voxel-decimated 3D point cloud
capture with interactive controls for voxel size, resolution mode, and
scale factor.

Controls:
  +/-     Increase/decrease voxel size
  1/2/3   Mode: FIXED / STEREO_UNCERTAINTY / LINEAR
  ,/.     Decrease/increase resolution scale
  c       Toggle centroid / grid center
  p/P     Decrease/increase point size
  d/D     Decrease/increase depth confidence
  v       Toggle voxel / full point cloud
  r       Reset all parameters
  Space   Pause / resume (SVO)
  Left/Right  Rewind / fast-forward 200 frames (SVO)
  s       Save point cloud to .ply
  q/Esc   Quit
"""

import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import argparse


def parse_args(init, opt):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if ip_str.replace(':', '').replace('.', '').isdigit() and len(ip_str.split('.')) == 4 and len(ip_str.split(':')) == 2:
            init.set_from_stream(ip_str.split(':')[0], int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ", ip_str)
        elif ip_str.replace(':', '').replace('.', '').isdigit() and len(ip_str.split('.')) == 4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ", ip_str)
        else:
            print("Invalid IP format. Using live stream")
    if "HD2K" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD2K
    elif "HD1200" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1200
    elif "HD1080" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1080
    elif "HD720" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD720
    elif "SVGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.SVGA
    elif "VGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.VGA


def main(opt):
    print("=== Voxel Point Cloud Viewer ===")
    print("  +/-     Increase/decrease voxel size")
    print("  1/2/3   Mode: FIXED / STEREO_UNCERTAINTY / LINEAR")
    print("  ,/.     Decrease/increase resolution scale")
    print("  c       Toggle centroid / grid center")
    print("  p/P     Decrease/increase point size")
    print("  d/D     Decrease/increase depth confidence")
    print("  v       Toggle voxel / full point cloud")
    print("  r       Reset all parameters")
    print("  Space   Pause / resume (SVO)")
    print("  Left/Right  Rewind / fast-forward 200 frames (SVO)")
    print("  s       Save point cloud to .ply")
    print("  q/Esc   Quit")
    print("================================\n")

    use_gpu = gl.GPU_ACCELERATION_AVAILABLE and not opt.disable_gpu_data_transfer
    mem_type = sl.MEM.GPU if use_gpu else sl.MEM.CPU
    if use_gpu:
        print("Using GPU data transfer with CuPy")

    init = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        coordinate_units=sl.UNIT.MILLIMETER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
        svo_real_time_mode=False,
    )
    parse_args(init, opt)

    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # Voxel parameters
    voxel_params = sl.VoxelMeasureParameters()
    voxel_params.voxel_size = 50.0  # 50 mm
    voxel_params.centroid = True
    voxel_params.resolution_mode = sl.VOXELIZATION_MODE.STEREO_UNCERTAINTY
    voxel_params.resolution_scale = 0.2

    # Get full measure resolution for viewer buffer (must be large enough for both voxel and full PC)
    point_cloud = sl.Mat()
    res = sl.Resolution()
    res.width = -1
    res.height = -1
    err = zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, res)
    res = point_cloud.get_resolution()
    print(f"[Info] Point cloud buffer: {res.width}x{res.height}, mem={mem_type}")

    MODE_MAP = [sl.VOXELIZATION_MODE.FIXED, sl.VOXELIZATION_MODE.STEREO_UNCERTAINTY, sl.VOXELIZATION_MODE.LINEAR]

    viewer = gl.GLViewer()
    viewer.init(1, sys.argv, res)

    left_image = sl.Mat()
    preview_res = sl.Resolution(640, 360)

    run_params = sl.RuntimeParameters()
    svo_position = 0

    while viewer.is_available():
        run_params.confidence_threshold = viewer.confidenceThreshold

        # SVO seek (arrow keys: +/- 200 frames)
        if viewer.seekOffset != 0:
            svo_position = max(0, svo_position + viewer.seekOffset)
            zed.set_svo_position(svo_position)
            viewer.seekOffset = 0

        # SVO pause: re-seek to the same frame before grabbing
        if viewer.paused:
            zed.set_svo_position(svo_position)

        grab_status = zed.grab(run_params)
        # Loop SVO when end is reached
        if grab_status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            zed.set_svo_position(0)
            svo_position = 0
            continue
        if grab_status <= sl.ERROR_CODE.SUCCESS:
            if not viewer.paused:
                svo_position = zed.get_svo_position()

            # Sync viewer HUD state → VoxelMeasureParameters (buttons/clicks modify viewer attrs)
            voxel_params.voxel_size = viewer.voxel_size
            voxel_params.resolution_scale = viewer.resolution_scale
            voxel_params.resolution_mode = MODE_MAP[viewer.resolution_mode]
            voxel_params.centroid = viewer.centroid

            # Handle keyboard input (for keys not handled by buttons)
            key = viewer.key_pressed
            if key:
                if key == ord('+') or key == ord('='):
                    viewer.voxel_size *= 1.25
                elif key == ord('-'):
                    viewer.voxel_size = max(5.0, viewer.voxel_size * 0.8)
                elif key == ord('1'):
                    viewer.resolution_mode = 0
                elif key == ord('2'):
                    viewer.resolution_mode = 1
                elif key == ord('3'):
                    viewer.resolution_mode = 2
                elif key == ord('.') or key == ord('>'):
                    viewer.resolution_scale = min(3.0, viewer.resolution_scale * 1.25)
                elif key == ord(',') or key == ord('<'):
                    viewer.resolution_scale = max(0.01, viewer.resolution_scale * 0.8)
                elif key == ord('c') or key == ord('C'):
                    viewer.centroid = not viewer.centroid
                elif key == ord('p'):
                    viewer.pointSize = max(0.5, viewer.pointSize - 0.2)
                elif key == ord('P'):
                    viewer.pointSize = min(20.0, viewer.pointSize + 0.2)
                elif key == ord('d'):
                    viewer.confidenceThreshold = max(1, viewer.confidenceThreshold - 5)
                elif key == ord('D'):
                    viewer.confidenceThreshold = min(100, viewer.confidenceThreshold + 5)
                elif key == ord('r') or key == ord('R'):
                    viewer.voxel_size = 50.0
                    viewer.resolution_scale = 0.2
                    viewer.resolution_mode = 1
                    viewer.centroid = True
                    viewer.pointSize = 3.0
                    viewer.confidenceThreshold = 50
                    viewer.use_voxels = True
                viewer.key_pressed = 0
                # Re-sync after keyboard
                voxel_params.voxel_size = viewer.voxel_size
                voxel_params.resolution_scale = viewer.resolution_scale
                voxel_params.resolution_mode = MODE_MAP[viewer.resolution_mode]
                voxel_params.centroid = viewer.centroid

            if viewer.use_voxels:
                zed.retrieve_voxel_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, voxel_params)
            else:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, res)
            zed.retrieve_image(left_image, sl.VIEW.LEFT, sl.MEM.CPU, preview_res)
            viewer.updateData(point_cloud)
            viewer.updateImage(left_image)

            pt_count = point_cloud.get_width()
            pause_tag = " PAUSED |" if viewer.paused else ""
            if viewer.use_voxels:
                mode_name = viewer.mode_names[viewer.resolution_mode]
                print(f"\rFPS: {zed.get_current_fps():.0f}"
                      f" |{pause_tag} VOXEL | pts: {pt_count}"
                      f" | voxel: {viewer.voxel_size:.1f} mm"
                      f" | mode: {mode_name}"
                      f" | scale: {viewer.resolution_scale:.2f}"
                      f" | conf: {viewer.confidenceThreshold}"
                      f" | {'centroid' if viewer.centroid else 'grid center'}"
                      f"        ", end='', flush=True)
            else:
                print(f"\rFPS: {zed.get_current_fps():.0f}"
                      f" |{pause_tag} FULL PC | pts: {pt_count}"
                      f" | conf: {viewer.confidenceThreshold}"
                      f"        ", end='', flush=True)

            if viewer.save_data:
                pc_save = sl.Mat()
                if viewer.use_voxels:
                    zed.retrieve_voxel_measure(pc_save, sl.MEASURE.XYZRGBA, sl.MEM.CPU, voxel_params)
                else:
                    zed.retrieve_measure(pc_save, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                err = pc_save.write('VoxelPointcloud.ply')
                if err <= sl.ERROR_CODE.SUCCESS:
                    print(f"\nSaved VoxelPointcloud.ply ({pc_save.get_width()} points)")
                else:
                    print("\nFailed to save .ply file")
                viewer.save_data = False

    viewer.exit()
    zed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, default='',
                        help='Path to an .svo file for replay')
    parser.add_argument('--ip_address', type=str, default='',
                        help='IP Address for streaming (a.b.c.d:port or a.b.c.d)')
    parser.add_argument('--resolution', type=str, default='',
                        help='Resolution: HD2K, HD1200, HD1080, HD720, SVGA, VGA')
    parser.add_argument('--disable-gpu-data-transfer', action='store_true',
                        help='Disable GPU data transfer with CuPy')
    opt = parser.parse_args()
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, not both. Exit.")
        exit()
    main(opt)
