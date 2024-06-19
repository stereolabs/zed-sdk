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
    Read SVO sample to read the video and the information of the camera. It can pick a frame of the svo and save it as
    a JPEG or PNG file. Depth map and Point Cloud can also be saved into files.
"""
import sys
import pyzed.sl as sl
import cv2
import argparse 
import os 

def main():
    filepath = opt.input_svo_file # Path to the .svo file to be playbacked
    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)  #Set init parameter to run from the .svo 
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera opened succesfully 
        print("Camera Open", status, "Exit program.")
        exit(1)


    print('External SVOData channels:', cam.get_svo_data_keys())

    # Set a maximum resolution, for visualisation confort 
    resolution = cam.get_camera_information().camera_configuration.resolution
    low_resolution = sl.Resolution(min(720,resolution.width) * 2, min(404,resolution.height))
    svo_image = sl.Mat(min(720,resolution.width) * 2,min(404,resolution.height), sl.MAT_TYPE.U8_C4, sl.MEM.CPU)
    
    runtime = sl.RuntimeParameters()
    
    mat = sl.Mat()

    key = ' '
    print(" Press 'q' to exit...")

    svo_frame_rate = cam.get_init_parameters().camera_fps
    nb_frames = cam.get_svo_number_of_frames()
    print("[Info] SVO contains " ,nb_frames," frames")
    

    key = ''
    last_timestamp_ns = sl.Timestamp()
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)

        data_map = {}
        # print("Reading between ", str(last_timestamp_ns.data_ns), " and ", str(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns))
        ing = cam.retrieve_svo_data("TEST", data_map, last_timestamp_ns, cam.get_timestamp(sl.TIME_REFERENCE.IMAGE))
        for d in data_map:  
            s = data_map[d].get_content_as_string()
            print("Retrieved:", s);

        last_timestamp_ns = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)

        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break

    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to the SVO file', required= True)
    opt = parser.parse_args()
    if not opt.input_svo_file.endswith(".svo") and not opt.input_svo_file.endswith(".svo2"): 
        print("--input_svo_file parameter should be a .svo file but is not : ",opt.input_svo_file,"Exit program.")
        exit()
    if not os.path.isfile(opt.input_svo_file):
        print("--input_svo_file parameter should be an existing file but is not : ",opt.input_svo_file,"Exit program.")
        exit()
    main()
