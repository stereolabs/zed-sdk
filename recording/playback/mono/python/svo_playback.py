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

def progress_bar(percent_done, bar_length=50):
    #Display progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %i%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()
    
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

    # Set a maximum resolution, for visualisation confort 
    resolution = cam.get_camera_information().camera_configuration.resolution
    low_resolution = sl.Resolution(min(720,resolution.width) * 2, min(404,resolution.height))
    svo_image = sl.Mat(min(720,resolution.width) * 2,min(404,resolution.height), sl.MAT_TYPE.U8_C4, sl.MEM.CPU)
    
    runtime = sl.RuntimeParameters()
    
    mat = sl.Mat()

    key = ' '
    print(" Press 's' to save SVO image as a PNG")
    print(" Press 'f' to jump forward in the video")
    print(" Press 'b' to jump backward in the video")
    print(" Press 'q' to exit...")

    svo_frame_rate = cam.get_init_parameters().camera_fps
    nb_frames = cam.get_svo_number_of_frames()
    print("[Info] SVO contains " ,nb_frames," frames")
    

    key = ''
    
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(svo_image,sl.VIEW.SIDE_BY_SIDE,sl.MEM.CPU,low_resolution) #retrieve image left and right
            svo_position = cam.get_svo_position()
            cv2.imshow("View", svo_image.get_data()) #dislay both images to cv2
            key = cv2.waitKey(10)
            if key == 115 :# for 's' key
                #save .svo image as a png
                cam.retrieve_image(mat)
                filepath = "capture_" + str(svo_position) + ".png"
                img = mat.write(filepath)
                if img == sl.ERROR_CODE.SUCCESS:
                    print("Saved image : ",filepath)
                else:
                    print("Something wrong happened in image saving... ")
            if key == 102: # for 'f' key
                #move forward one second 
                cam.set_svo_position(svo_position+svo_frame_rate)
            if key == 98: #for 'b' key 
                #move backward one second 
                cam.set_svo_position(svo_position-svo_frame_rate)
            progress_bar(svo_position /nb_frames*100, 30) 
        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED: #Check if the .svo has ended
            progress_bar(100, 30) 
            print("SVO end has been reached. Looping back to 0")
            cam.set_svo_position(0)
        else:
            print("Grab ZED : ", err)
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
