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
    Live camera sample showing the camera information and video in real time and allows to control the different
    settings.
"""

import cv2
import numpy as np
import pyzed.sl as sl

camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

cam = sl.Camera()

drawing = False
selection_rect = sl.Rect(0,0,0,0)
origin_rect = (-1, -1)


def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(round(cam.get_camera_information().camera_resolution.width, 2), cam.get_camera_information().camera_resolution.height))
    print("Camera FPS: {0}.".format(cam.get_camera_information().camera_fps))
    print("Firmware: {0}.".format(cam.get_camera_information().camera_firmware_version))
    print("Serial number: {0}.\n".format(cam.get_camera_information().serial_number))


def print_help():
    print("Help for camera setting controls")
    print("  Reset exposure ROI to full image:   r")
    print("  Quit:                               q\n")


# mouse callback function
def draw_rect(event,x,y,flags,param):
    global drawing, selection_rect, origin_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        origin_rect = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cam.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, selection_rect, sl.SIDE.BOTH)

    if drawing:
        selection_rect.x = min(x, origin_rect[0])
        selection_rect.y = min(y, origin_rect[1])
        selection_rect.width = abs(x - origin_rect[0]) + 1
        selection_rect.height = abs(y - origin_rect[1]) + 1


def main():
    global drawing, selection_rect, cam
    #create camera object
    print("Running...")
    init = sl.InitParameters()
    # init.camera_resolution=sl.resolution.HD2K

    #open the camera
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()


    #create mouse callback
    win_name = "ROI Exposure"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, draw_rect)

    print_camera_information(cam)
    print_help()

    #capute new images until 'q' is pressed
    key = ''
    while key != 113:  # for 'q' key
        #Check that a new image is successfully acquired
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            #retrieve the left image
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            cvImage = mat.get_data()

            #Check that selection rectangle is valid and draw it on the image
            
            if (not selection_rect.is_empty() ):# and selection_rect.is_contained(sl.resolution(mat.cols, mat.rows))):
                cv2.rectangle(cvImage,(selection_rect.x,selection_rect.y,selection_rect.width, selection_rect.height),(220, 180, 20), 2)
            cv2.imshow(win_name, cvImage)
        else:
            print("Error during capture : ", err)
            break
        
        #Change camera settings with keyboard 
        key = cv2.waitKey(10)
        if key == 114:  # for 'r' key
            drawing = False
            selection_rect = sl.Rect(0,0,0,0)
            print("reset AEC_AGC_ROI to full res")
            cam.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI, selection_rect, reset=True)


    cv2.destroyAllWindows()

    cam.close()
    print("\nFINISH")


if __name__ == "__main__":
    main()