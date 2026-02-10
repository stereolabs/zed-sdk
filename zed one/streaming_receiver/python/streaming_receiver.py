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
ZED One Streaming Receiver Sample

This sample demonstrates how to receive and display a video stream
from a remote ZED One camera with camera settings control.
"""

import pyzed.sl as sl
import cv2
import argparse
import socket

# Camera settings state
camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1
led_on = True

# ROI selection state
selection_rect = sl.Rect()
select_in_progress = False
origin_rect = (-1, -1)


def on_mouse(event, x, y, flags, param):
    """Mouse callback for ROI selection."""
    global select_in_progress, selection_rect, origin_rect
    
    if event == cv2.EVENT_LBUTTONDOWN:
        origin_rect = (x, y)
        select_in_progress = True
    elif event == cv2.EVENT_LBUTTONUP:
        select_in_progress = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        select_in_progress = False
        selection_rect = sl.Rect(0, 0, 0, 0)

    if select_in_progress:
        selection_rect.x = min(x, origin_rect[0])
        selection_rect.y = min(y, origin_rect[1])
        selection_rect.width = abs(x - origin_rect[0]) + 1
        selection_rect.height = abs(y - origin_rect[1]) + 1


def print_help():
    """Print keyboard controls."""
    print("\nCamera controls:")
    print("  +  Increase setting value")
    print("  -  Decrease setting value")
    print("  s  Switch camera setting")
    print("  l  Toggle LED")
    print("  r  Reset all settings")
    print("  a  Apply exposure ROI")
    print("  f  Reset exposure ROI")
    print("  q  Quit\n")


def switch_camera_settings():
    """Cycle through available camera settings."""
    global camera_settings, str_camera_settings, step_camera_settings
    
    settings_order = [
        (sl.VIDEO_SETTINGS.BRIGHTNESS, "Brightness", 1),
        (sl.VIDEO_SETTINGS.CONTRAST, "Contrast", 1),
        (sl.VIDEO_SETTINGS.HUE, "Hue", 1),
        (sl.VIDEO_SETTINGS.SATURATION, "Saturation", 1),
        (sl.VIDEO_SETTINGS.SHARPNESS, "Sharpness", 1),
        (sl.VIDEO_SETTINGS.GAIN, "Gain", 1),
        (sl.VIDEO_SETTINGS.EXPOSURE, "Exposure", 1),
        (sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, "White Balance", 100),
    ]
    
    # Find current index and move to next
    current_idx = 0
    for i, (setting, _, _) in enumerate(settings_order):
        if setting == camera_settings:
            current_idx = i
            break
    
    next_idx = (current_idx + 1) % len(settings_order)
    camera_settings, str_camera_settings, step_camera_settings = settings_order[next_idx]
    print(f"[Sample] Switch to camera settings: {str_camera_settings}")


def update_camera_settings(key: int, cam: sl.CameraOne) -> None:
    """Handle keyboard input for camera settings."""
    global led_on
    
    if key == ord('s'):
        switch_camera_settings()
    elif key == ord('+'):
        current_value = cam.get_camera_settings(camera_settings)[1]
        cam.set_camera_settings(camera_settings, current_value + step_camera_settings)
        print(f"{str_camera_settings}: {current_value + step_camera_settings}")
    elif key == ord('-'):
        current_value = cam.get_camera_settings(camera_settings)[1]
        if current_value >= step_camera_settings:
            cam.set_camera_settings(camera_settings, current_value - step_camera_settings)
            print(f"{str_camera_settings}: {current_value - step_camera_settings}")
    elif key == ord('r'):
        # Reset all settings
        for setting in [sl.VIDEO_SETTINGS.BRIGHTNESS, sl.VIDEO_SETTINGS.CONTRAST,
                       sl.VIDEO_SETTINGS.HUE, sl.VIDEO_SETTINGS.SATURATION,
                       sl.VIDEO_SETTINGS.SHARPNESS, sl.VIDEO_SETTINGS.GAIN,
                       sl.VIDEO_SETTINGS.EXPOSURE, sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE]:
            cam.set_camera_settings(setting, -1)
        print("[Sample] Reset all settings to default")
    elif key == ord('l'):
        led_on = not led_on
        cam.set_camera_settings(sl.VIDEO_SETTINGS.LED_STATUS, led_on)
    elif key == ord('a'):
        if not selection_rect.is_empty():
            print(f"[Sample] Set AEC_AGC_ROI: [{selection_rect.x}, {selection_rect.y}, "
                  f"{selection_rect.width}, {selection_rect.height}]")
            cam.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, selection_rect)
    elif key == ord('f'):
        print("[Sample] Reset AEC_AGC_ROI to full image")
        cam.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, selection_rect, True)


def valid_ip_or_hostname(ip_or_hostname):
    """Validate IP:port format."""
    try:
        host, port = ip_or_hostname.split(':')
        socket.inet_aton(host)
        port = int(port)
        return f"{host}:{port}"
    except (socket.error, ValueError):
        raise argparse.ArgumentTypeError(
            "Invalid format. Use: IP:port (e.g., 192.168.1.100:30000)")


def main(opt):
    # Parse IP and port
    host, port = opt.ip_address.split(':')
    
    # Initialize camera
    init_parameters = sl.InitParametersOne()
    init_parameters.sdk_verbose = 1
    init_parameters.set_from_stream(host, int(port))
    
    cam = sl.CameraOne()
    status = cam.open(init_parameters)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open: {repr(status)}. Exit program.")
        exit(1)

    # Print stream info
    cam_info = cam.get_camera_information()
    print("\n=== Stream Connected ===")
    print(f"Model:      {cam_info.camera_model}")
    print(f"Serial:     {cam_info.serial_number}")
    print(f"Resolution: {cam_info.camera_configuration.resolution.width}x"
          f"{cam_info.camera_configuration.resolution.height}")
    print("========================\n")

    print_help()

    # Setup OpenCV window
    win_name = "ZED One - Remote Control"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    mat = sl.Mat()
    switch_camera_settings()

    key = ''
    while key != ord('q'):
        err = cam.grab()
        if err <= sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat)
            cv_image = mat.get_data()
            
            # Draw selection rectangle
            if (not selection_rect.is_empty() and 
                selection_rect.is_contained(sl.Rect(0, 0, cv_image.shape[1], cv_image.shape[0]))):
                cv2.rectangle(cv_image,
                    (selection_rect.x, selection_rect.y),
                    (selection_rect.x + selection_rect.width, selection_rect.y + selection_rect.height),
                    (0, 255, 0), 2)
            
            cv2.imshow(win_name, cv_image)
        else:
            print(f"Error during capture: {err}")
            break
        
        key = cv2.waitKey(5)
        update_camera_settings(key, cam)

    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED One Streaming Receiver Sample")
    parser.add_argument('--ip_address', type=valid_ip_or_hostname, required=True,
                        help='IP:port of the sender (e.g., 192.168.1.100:30000)')
    opt = parser.parse_args()
    main(opt)
