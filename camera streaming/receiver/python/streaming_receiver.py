########################################################################
#
# Copyright (c) 2021, STEREOLABS.
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
    Read a stream and display the left images using OpenCV
"""
import sys
import pyzed.sl as sl
import cv2


def main():

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    if (len(sys.argv) > 1) :
        ip = sys.argv[1]
        init.set_from_stream(ip)
    else :
        print('Usage : python3 streaming_receiver.py ip')
        exit(1)

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    key = ''
    print("  Quit : CTRL+C\n")
    while key != 113:
        err = cam.grab(runtime)
        if (err == sl.ERROR_CODE.SUCCESS) :
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            cv2.imshow("ZED", mat.get_data())
            key = cv2.waitKey(1)
        else :
            key = cv2.waitKey(1)

    cam.close()

if __name__ == "__main__":
    main()
