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
    Record and play back encrypted SVO2 files with the ZED SDK.

    If the SVO file does not exist, records from a live camera.
    If the SVO file already exists, plays it back.

    The encryption key can be:
      - A passphrase (any string, key derived via PBKDF2-SHA256)
      - A path to a 32-byte binary key file
      - A 64-character hex string representing a raw 256-bit key

    Encrypted SVO2 files use AES-256-CTR and require OpenSSL to be installed.

    Usage:
      python encrypted_svo.py --svo_file output.svo2 --key "my secret passphrase"
"""

import sys
import os
import argparse
import pyzed.sl as sl
from signal import signal, SIGINT

cam = sl.Camera()

def handler(signal_received, frame):
    cam.disable_recording()
    cam.close()
    sys.exit(0)

signal(SIGINT, handler)


def record(svo_file, key):
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.async_image_retrieval = False

    status = cam.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print("[Sample][Error] Camera Open:", status)
        exit(1)

    recording_param = sl.RecordingParameters(
        svo_file,
        sl.SVO_COMPRESSION_MODE.H265,
        encryption_key=key
    )
    err = cam.enable_recording(recording_param)
    if err > sl.ERROR_CODE.SUCCESS:
        print("[Sample][Error] Enable Recording:", err, "- Is OpenSSL installed?")
        cam.close()
        exit(1)

    print("[Sample] Encrypted SVO is recording, use Ctrl-C to stop.")
    frames_recorded = 0

    while True:
        if cam.grab() <= sl.ERROR_CODE.SUCCESS:
            frames_recorded += 1
            print(f"  Frames recorded: {frames_recorded}", end="\r")
        else:
            break

    cam.disable_recording()
    cam.close()
    print(f"\n[Sample] Recording stopped. Encrypted SVO saved to {svo_file}")


def play(svo_file, key):
    import cv2

    print(f"[Sample] File exists, playing back: {svo_file}")
    input_type = sl.InputType()
    input_type.set_from_svo_file(svo_file)
    init = sl.InitParameters(input_t=input_type)
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.svo_decryption_key = key

    status = cam.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print("[Sample][Error] Camera Open:", status, "- Wrong key or corrupted file?")
        exit(1)

    resolution = cam.get_camera_information().camera_configuration.resolution
    low_resolution = sl.Resolution(min(720, resolution.width) * 2, min(404, resolution.height))
    svo_image = sl.Mat(low_resolution.width, low_resolution.height, sl.MAT_TYPE.U8_C4, sl.MEM.CPU)

    nb_frames = cam.get_svo_number_of_frames()
    svo_frame_rate = cam.get_init_parameters().camera_fps
    print(f"[Sample] Encrypted SVO contains {nb_frames} frames")
    print(" Press 's' to save a frame as PNG")
    print(" Press 'f' to jump forward")
    print(" Press 'b' to jump backward")
    print(" Press 'q' to exit...")

    key_pressed = 0
    while key_pressed != ord('q'):
        err = cam.grab()
        if err <= sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(svo_image, sl.VIEW.SIDE_BY_SIDE, sl.MEM.CPU, low_resolution)
            svo_position = cam.get_svo_position()

            cv2.imshow("Encrypted SVO Playback", svo_image.get_data())
            key_pressed = cv2.waitKey(10)

            if key_pressed == ord('s'):
                filepath = f"capture_{svo_position}.png"
                svo_image.write(filepath)
                print(f"  Saved: {filepath}")
            elif key_pressed == ord('f'):
                cam.set_svo_position(svo_position + svo_frame_rate)
            elif key_pressed == ord('b'):
                cam.set_svo_position(svo_position - svo_frame_rate)

            progress = svo_position / nb_frames * 100 if nb_frames > 0 else 0
            done = int(30 * progress / 100)
            bar = '=' * done + '-' * (30 - done)
            sys.stdout.write(f'[{bar}] {int(progress)}%\r')
            sys.stdout.flush()

        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("\n[Sample] SVO end reached. Looping back to 0")
            cam.set_svo_position(0)
        else:
            print("[Sample][Error] Grab:", err)
            break

    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record and play back encrypted SVO2 files")
    parser.add_argument('--svo_file', type=str, required=True,
                        help="Path to the SVO2 file (records if missing, plays if exists)")
    parser.add_argument('--key', type=str, required=True,
                        help="Encryption/decryption key (passphrase, key file path, or 64-char hex)")
    opt = parser.parse_args()

    if not opt.svo_file.endswith((".svo", ".svo2")):
        print("--svo_file should be a .svo/.svo2 file:", opt.svo_file)
        exit(1)

    if os.path.isfile(opt.svo_file):
        play(opt.svo_file, opt.key)
    else:
        record(opt.svo_file, opt.key)
