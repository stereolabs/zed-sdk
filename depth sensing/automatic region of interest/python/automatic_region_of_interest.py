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
    This sample demonstrates how to apply an exclusion ROI to all ZED SDK measures
    This can be very useful to avoid noise from a vehicle bonnet or drone propellers for instance
"""

import sys
import pyzed.sl as sl
import argparse
import cv2


def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")


def main():
    # Create a ZED Camera object
    zed = sl.Camera()

    init_parameters = sl.InitParameters()
    init_parameters.camera_resolution = sl.RESOLUTION.AUTO
    init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL
    parse_args(init_parameters)

    # Open the camera
    returned_state = zed.open(init_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print("Camera Open", returned_state, "Exit program.")
        exit()

    imWndName = "Image"
    depthWndName = "Depth"
    ROIWndName = "ROI"
    cv2.namedWindow(imWndName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(ROIWndName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(depthWndName, cv2.WINDOW_NORMAL)

    print("Press 'a' to apply the ROI")
    print("Press 'r' to reset the ROI")
    print("Press 's' to save the ROI as image file to reload it later")
    print("Press 'l' to load the ROI from an image file")

    resolution = zed.get_camera_information().camera_configuration.resolution

    # Create a Mat to store images
    zed_image = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.U8_C4, sl.MEM.CPU)
    zed_depth_image = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.U8_C4, sl.MEM.CPU)

    mask_name = "Mask.png"
    mask_roi = sl.Mat(resolution.width, resolution.height, sl.MAT_TYPE.U8_C1, sl.MEM.CPU)

    roi_running = False
    roi_param = sl.RegionOfInterestParameters()
    roi_param.auto_apply = True
    roi_param.depth_far_threshold_meters = 2.5
    roi_param.image_height_ratio_cutoff = 0.5
    zed.start_region_of_interest_auto_detection(roi_param)

    # Capture new images until 'q' is pressed
    key = ' '
    while key != 'q' and key != 27:
        # Check that a new image is successfully acquired
        returned_state = zed.grab()
        if returned_state == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(zed_image, sl.VIEW.LEFT)
            zed.retrieve_image(zed_depth_image, sl.VIEW.DEPTH)

            status = zed.get_region_of_interest_auto_detection_status()
            if roi_running:
                text = "Region of interest auto detection is running\r"
                if status == sl.REGION_OF_INTEREST_AUTO_DETECTION_STATE.READY:
                    print(text, "Region of interest auto detection is done!   ")
                    zed.getRegionOfInterest(mask_roi)
                    cvMaskROI = mask_roi.get_data()
                    cv2.imshow(ROIWndName, cvMaskROI)

            roi_running = (status == sl.REGION_OF_INTEREST_AUTO_DETECTION_STATE.RUNNING)

            cvImage = zed_image.get_data()
            cvDepthImage = zed_depth_image.get_data()
            cv2.imshow(imWndName, cvImage)
            cv2.imshow(depthWndName, cvDepthImage)

        key = cv2.waitKey(15)

        # Apply Current ROI
        if key == 'r': #Reset ROI
            if not roi_running:
                emptyROI = sl.Mat()
                zed.setRegionOfInterest(emptyROI)
            print("Resetting Auto ROI detection")
            zed.startRegionOfInterestAutoDetection(roi_param)
        elif key == 's' and mask_roi.is_init():
            print("Saving ROI to", mask_name)
            mask_roi.write(mask_name)
        elif key == 'l':
            # Load the mask from a previously saved file
            tmp = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            if not tmp.empty():
                slROI = sl.Mat(sl.Resolution(tmp.cols, tmp.rows), sl.MAT_TYPE.U8_C1, tmp.data, tmp.step)
                zed.set_region_of_interest(slROI)
            print(mask_name, "could not be found")

    # Exit
    zed.close()
    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main()