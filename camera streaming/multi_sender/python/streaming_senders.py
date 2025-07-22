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
This sample shows how to stream remotely the video of multiple ZED camera. 
Any application using the ZED SDK can receive and process this stream.
See Camera Streaming/Receiver example.
"""

import pyzed.sl as sl
import threading
import signal
import time
import sys


# Global variable to handle exit
exit_app = False


def signal_handler(signal, frame):
    """Handle Ctrl+C to properly exit the program"""
    global exit_app
    exit_app = True
    print("\nCtrl+C pressed. Exiting...")


def acquisition(zed):
    """Acquisition thread function to continuously grab frames"""
    infos = zed.get_camera_information()

    while not exit_app:
        if zed.grab() <= sl.ERROR_CODE.SUCCESS:
            # If you want to add more computation here and to keep maximum performance with multiple cameras:
            # 1. Minimize Python operations in the loop to avoid python to block the GIL
            # 2. Pre-allocate objects and arrays to reduce memory allocations
            # 3. Rely on the pyzed library which is optimized for performance
            pass

    print(f"{infos.camera_model}[{infos.serial_number}] QUIT")

    # disable Streaming
    zed.disable_streaming()
    # close the Camera
    zed.close()


def open_camera(zed, sn, port, camera_fps=30):
    """Open a camera with given serial number and enable streaming with given port"""
    if isinstance(zed, sl.Camera):
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.NONE  # No depth mode for this example
    elif isinstance(zed, sl.CameraOne):
        init_params = sl.CameraOne.InitParametersOne()
    else:
        print(f"Unsupported camera type: {type(zed)}")
        return False
    init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.set_from_serial_number(sn)
    init_params.camera_fps = camera_fps

    # Open the camera
    open_err = zed.open(init_params)
    if open_err == sl.ERROR_CODE.SUCCESS:
        print(f"{zed.get_camera_information().camera_model}_SN{sn} Opened")
    else:
        print(f"ZED SN{sn} Error: {open_err}")
        zed.close()
        return False

    # Enable streaming
    stream_params = sl.StreamingParameters()
    stream_params.port = port
    stream_err = zed.enable_streaming(stream_params)
    if stream_err == sl.ERROR_CODE.SUCCESS:
        print(f"{zed.get_camera_information().camera_model}_SN{sn} Enabled streaming")
    else:
        print(f"ZED SN{sn} Streaming initialization error: {stream_err}")
        zed.close()
        return False

    print(f"Streaming on port {stream_params.port}")
    return True


def print_device_info(devs):
    """Print information about detected devices"""
    for dev in devs:
        print(f"ID: {dev.id}, model: {dev.camera_model}, S/N: {dev.serial_number}, state: {dev.camera_state}")


def main():
    global exit_app

    # Get the list of available ZED cameras
    dev_stereo_list = sl.Camera.get_device_list()
    print_device_info(dev_stereo_list)
    if sys.platform != "win32":
        dev_mono_list = sl.CameraOne.get_device_list()
        print_device_info(dev_mono_list)
    else:
        dev_mono_list = []

    nb_cameras = len(dev_stereo_list) + len(dev_mono_list)
    if nb_cameras == 0:
        print("No ZED Detected, exit program")
        return 1

    zed_open = False

    # Open all cameras
    zeds = [sl.Camera() if i < len(dev_stereo_list) else sl.CameraOne()
            for i in range(nb_cameras)]
    for z in range(nb_cameras):
        port = 30000 + 2 * z  # Example port assignment
        zed_open |= open_camera(zeds[z], dev_stereo_list[z].serial_number, port)

    if not zed_open:
        print("No ZED opened, exit program")
        return 1

    # Create a grab thread for each opened camera
    threads = []
    for z in range(nb_cameras):
        if zeds[z].is_opened():
            threads.append(threading.Thread(target=acquisition, args=(zeds[z],)))
            threads[-1].start()

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    print("Press Ctrl+C to exit")

    # Main loop
    while not exit_app:
        time.sleep(0.02)

    # Wait for all threads to finish
    print("Exit signal, closing ZEDs")
    time.sleep(0.1)

    for thread in threads:
        thread.join()

    print("Program exited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
