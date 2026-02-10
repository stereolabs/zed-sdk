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
ZED One SVO Recording Sample

This sample demonstrates how to record video from a ZED One camera
to SVO format using H.265 compression.
"""

import sys
import pyzed.sl as sl
from signal import signal, SIGINT
import argparse

# Global camera reference for signal handler
cam = sl.CameraOne()


def signal_handler(signal_received, frame):
    """Handle Ctrl+C for graceful shutdown."""
    print("\n\nStopping recording...")
    cam.disable_recording()
    cam.close()
    sys.exit(0)


def main(opt):
    # Register signal handler
    signal(SIGINT, signal_handler)

    # Initialize camera
    init = sl.InitParametersOne()
    status = cam.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open: {status}. Exit program.")
        exit(1)

    # Get camera info
    cam_info = cam.get_camera_information()

    # Enable recording
    recording_param = sl.RecordingParameters(opt.output_svo_file, sl.SVO_COMPRESSION_MODE.H265)
    err = cam.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Recording initialization error: {err}")
        cam.close()
        exit(1)

    # Print recording info
    print("\n=== Recording Started ===")
    print(f"Output file:  {opt.output_svo_file}")
    print(f"Resolution:   {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
    print(f"FPS:          {cam_info.camera_configuration.fps}")
    print("Compression:  H.265")
    print("=========================\n")
    print("Recording... Press Ctrl+C to stop.")

    frames_recorded = 0
    while True:
        if cam.grab() <= sl.ERROR_CODE.SUCCESS:
            frames_recorded += 1
            print(f"Frames recorded: {frames_recorded}", end="\r")

    # This point is reached only on error
    cam.disable_recording()
    cam.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED One SVO Recording Sample")
    parser.add_argument('--output_svo_file', type=str, required=True,
                        help='Path to the output SVO file')
    opt = parser.parse_args()
    
    if not opt.output_svo_file.endswith((".svo", ".svo2")):
        print(f"Error: Output file must be .svo or .svo2: {opt.output_svo_file}")
        exit(1)
    
    main(opt)
