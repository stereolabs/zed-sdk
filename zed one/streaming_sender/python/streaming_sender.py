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
ZED One Streaming Sender Sample

This sample demonstrates how to stream live video from a ZED One camera
over the network using H.265 codec.
"""

import pyzed.sl as sl
import argparse
from time import sleep


def parse_resolution(resolution_str):
    """Parse resolution string and return corresponding sl.RESOLUTION."""
    resolution_map = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA,
        "VGA": sl.RESOLUTION.VGA,
    }
    return resolution_map.get(resolution_str.upper(), sl.RESOLUTION.AUTO)


def main(opt):
    # Initialize camera
    init = sl.InitParametersOne()
    init.camera_resolution = parse_resolution(opt.resolution) if opt.resolution else sl.RESOLUTION.AUTO
    init.sdk_verbose = 1
    
    if opt.resolution:
        print(f"[Sample] Using resolution: {opt.resolution}")
    else:
        print("[Sample] Using default resolution")

    # Open camera
    cam = sl.CameraOne()
    status = cam.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open: {repr(status)}. Exit program.")
        exit(1)

    # Configure streaming
    stream_params = sl.StreamingParameters()
    stream_params.codec = sl.STREAMING_CODEC.H265
    stream_params.bitrate = 4000

    # Enable streaming
    status_streaming = cam.enable_streaming(stream_params)
    if status_streaming != sl.ERROR_CODE.SUCCESS:
        print(f"Streaming initialization error: {status_streaming}")
        cam.close()
        exit(1)

    # Print streaming info
    cam_info = cam.get_camera_information()
    print("\n=== Streaming Started ===")
    print(f"Port:       {stream_params.port}")
    print(f"Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
    print(f"FPS:        {cam_info.camera_configuration.fps}")
    print(f"Codec:      H.265")
    print(f"Bitrate:    {stream_params.bitrate} kbps")
    print("=========================\n")

    print(f"Streaming on port {stream_params.port}...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            err = cam.grab()
            if err > sl.ERROR_CODE.SUCCESS:
                sleep(0.001)
    except KeyboardInterrupt:
        print("\n\nStopping stream...")

    # Cleanup
    cam.disable_streaming()
    cam.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED One Streaming Sender Sample")
    parser.add_argument('--resolution', type=str, default='',
                        help='Resolution: HD2K, HD1200, HD1080, HD720, SVGA, VGA')
    opt = parser.parse_args()
    main(opt)
