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
ZED One SVO Playback Sample

This sample demonstrates how to read and play back SVO files
recorded with a ZED One camera.
"""

import sys
import pyzed.sl as sl
import cv2
import argparse
import os


def progress_bar(percent_done, bar_length=50):
    """Display a progress bar in the console."""
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %i%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def print_controls():
    """Print available keyboard controls."""
    print("\nControls:")
    print("  's' - Save current frame as PNG")
    print("  'f' - Jump forward 1 second")
    print("  'b' - Jump backward 1 second")
    print("  'q' - Quit")
    print()


def main(opt):
    # Set input from SVO file
    input_type = sl.InputType()
    input_type.set_from_svo_file(opt.input_svo_file)
    
    # Initialize camera parameters
    init = sl.InitParametersOne()
    init.input = input_type
    
    # Open the camera
    cam = sl.CameraOne()
    status = cam.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open: {status}. Exit program.")
        exit(1)

    # Get SVO info
    resolution = cam.get_camera_information().camera_configuration.resolution
    svo_frame_rate = cam.get_init_parameters().camera_fps
    nb_frames = cam.get_svo_number_of_frames()
    
    print("\n=== SVO Information ===")
    print(f"File:         {opt.input_svo_file}")
    print(f"Resolution:   {resolution.width}x{resolution.height}")
    print(f"Frame rate:   {svo_frame_rate} fps")
    print(f"Total frames: {nb_frames}")
    print("=======================")
    
    print_controls()

    # Create display resolution
    low_resolution = sl.Resolution(min(1280, resolution.width), min(720, resolution.height))
    svo_image = sl.Mat(low_resolution.width, low_resolution.height, sl.MAT_TYPE.U8_C4, sl.MEM.CPU)
    mat = sl.Mat()

    key = ''
    while key != ord('q'):
        err = cam.grab()
        if err <= sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(svo_image, sl.VIEW.LEFT, sl.MEM.CPU, low_resolution)
            svo_position = cam.get_svo_position()
            
            # Display frame
            cv2.imshow("ZED One - SVO Playback", svo_image.get_data())
            key = cv2.waitKey(10)
            
            if key == ord('s'):
                # Save current frame as PNG
                cam.retrieve_image(mat)
                filepath = f"capture_{svo_position}.png"
                result = mat.write(filepath)
                if result == sl.ERROR_CODE.SUCCESS:
                    print(f"Saved: {filepath}")
                else:
                    print("Error saving image")
            elif key == ord('f'):
                # Jump forward one second
                cam.set_svo_position(min(svo_position + svo_frame_rate, nb_frames - 1))
            elif key == ord('b'):
                # Jump backward one second
                cam.set_svo_position(max(svo_position - svo_frame_rate, 0))
            
            progress_bar(svo_position / nb_frames * 100, 30)
            
        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            progress_bar(100, 30)
            print("\nSVO end reached. Looping back to start.")
            cam.set_svo_position(0)
        else:
            print(f"Grab error: {err}")
            break

    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED One SVO Playback Sample")
    parser.add_argument('--input_svo_file', type=str, required=True,
                        help='Path to the SVO file to play back')
    opt = parser.parse_args()
    
    if not opt.input_svo_file.endswith((".svo", ".svo2")):
        print(f"Error: File must be .svo or .svo2: {opt.input_svo_file}")
        exit(1)
    if not os.path.isfile(opt.input_svo_file):
        print(f"Error: File not found: {opt.input_svo_file}")
        exit(1)
    
    main(opt)
