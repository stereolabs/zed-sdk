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
ZED One Live Capture Sample

This sample demonstrates how to grab images from a ZED One camera
and display them using OpenCV.
"""

import pyzed.sl as sl
import cv2


def main():
    # Create a Camera object
    zed = sl.CameraOne()

    # Create InitParameters and set configuration
    init_params = sl.InitParametersOne()
    init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.camera_fps = 30

    # Open the camera
    err = zed.open(init_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open: {repr(err)}. Exit program.")
        exit(1)

    # Get and display camera information
    cam_info = zed.get_camera_information()
    print("\n=== Camera Information ===")
    print(f"Model:      {cam_info.camera_model}")
    print(f"Serial:     {cam_info.serial_number}")
    print(f"Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
    print(f"FPS:        {cam_info.camera_configuration.fps}")
    print("==========================\n")

    print("Press 'q' to quit...")

    # Create Mat for image storage
    image = sl.Mat()

    # Capture loop
    key = ''
    while key != ord('q'):
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image)
            cv2.imshow("ZED One - Live", image.get_data())
        
        key = cv2.waitKey(10)

    cv2.destroyAllWindows()
    zed.close()


if __name__ == "__main__":
    main()
