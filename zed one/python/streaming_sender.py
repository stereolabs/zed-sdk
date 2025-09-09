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
    Open the camera and start streaming images using H265 codec
"""
import sys
import pyzed.sl as sl
import argparse
from time import sleep

def parse_args(init, opt):
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

def main(opt):
    init = sl.InitParametersOne()
    init.camera_resolution = sl.RESOLUTION.AUTO
    init.sdk_verbose = 1
    parse_args(init, opt)
    cam = sl.CameraOne()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()
    stream_params = sl.StreamingParameters()
    print("Streaming on port ",stream_params.port) #Get the port used to stream
    stream_params.codec = sl.STREAMING_CODEC.H265
    stream_params.bitrate = 4000
    status_streaming = cam.enable_streaming(stream_params) #Enable streaming
    if status_streaming != sl.ERROR_CODE.SUCCESS:
        print("Streaming initialization error: ", status_streaming)
        cam.close()
        exit()
    exit_app = False 
    try : 
        while not exit_app:
            err = cam.grab()
            if err <= sl.ERROR_CODE.SUCCESS: 
                sleep(0.001)
    except KeyboardInterrupt:
        exit_app = True 

    # disable Streaming
    cam.disable_streaming()
    # close the Camera
    cam.close()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    main(opt)
