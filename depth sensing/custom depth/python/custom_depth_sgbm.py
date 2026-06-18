########################################################################
#
# Copyright (c) 2026, STEREOLABS.
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
    Custom depth ingest sample (DEPTH_MODE.CUSTOM) — OpenCV StereoSGBM.

    The SDK acquisition and rectification are used, the disparity is computed
    externally (cv2.StereoSGBM at half resolution), then ingested back with
    Camera.ingest_custom_depth(). The full SDK pipeline (depth measures, point
    cloud, modules) then runs on the ingested disparity.

    Per-frame sequence: read() -> retrieve rectified images -> compute ->
    ingest_custom_depth() -> grab().

    Usage: python3 custom_depth_sgbm.py [file.svo2]
"""

import sys
import cv2
import numpy as np
import pyzed.sl as sl


def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.CUSTOM  # <- no internal depth computation
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_stabilization = 0  # show the raw ingested disparity
    if len(sys.argv) > 1 and ".svo" in sys.argv[1]:
        init_params.set_from_svo_file(sys.argv[1])

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera open failed:", status)
        exit(1)

    cam_res = zed.get_camera_information().camera_configuration.resolution
    # SGBM runs at half resolution for speed: the map can be ingested at any resolution,
    # disparity values are expressed in pixels AT THE MAP RESOLUTION (rescaled internally).
    sgbm_w, sgbm_h = cam_res.width // 2, cam_res.height // 2

    block_size = 5
    sgbm = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=16 * 8,  # multiple of 16
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    left_gray = sl.Mat()
    right_gray = sl.Mat()
    depth_view = sl.Mat()
    depth_measure = sl.Mat()

    # Pre-allocated ingest maps: get_data(deep_copy=False) returns a writable view
    # over the sl.Mat memory, so writing the numpy result fills the Mat directly.
    disparity_mat = sl.Mat(sgbm_w, sgbm_h, sl.MAT_TYPE.F32_C1, sl.MEM.CPU)
    disparity_view = disparity_mat.get_data(deep_copy=False)
    confidence_mat = sl.Mat(sgbm_w, sgbm_h, sl.MAT_TYPE.F32_C1, sl.MEM.CPU)
    confidence_view = confidence_mat.get_data(deep_copy=False)

    custom_depth = sl.CustomDepthData()
    custom_depth.format = sl.CUSTOM_DEPTH_FORMAT.DISPARITY
    custom_depth.scale = 1.0 / 16.0  # SGBM outputs 16-bit fixed point (disparity * 16)
    custom_depth.confidence_convention = sl.CUSTOM_CONFIDENCE_CONVENTION.PROBABILITY

    runtime_params = sl.RuntimeParameters()

    win_name = "SDK depth from custom SGBM disparity"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("confidence", win_name, 95, 100, lambda v: None)

    print("Press 'q' to exit")
    frame = 0
    while True:
        # 1. Acquisition only: image + IMU (+ recording/streaming if enabled), no depth
        if zed.read() != sl.ERROR_CODE.SUCCESS:
            continue

        # 2. Rectified images (rectification runs on demand)
        zed.retrieve_image(left_gray, sl.VIEW.LEFT_GRAY, sl.MEM.CPU)
        zed.retrieve_image(right_gray, sl.VIEW.RIGHT_GRAY, sl.MEM.CPU)
        image_ts = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)

        left_half = cv2.resize(left_gray.get_data(deep_copy=False), (sgbm_w, sgbm_h), interpolation=cv2.INTER_AREA)
        right_half = cv2.resize(right_gray.get_data(deep_copy=False), (sgbm_w, sgbm_h), interpolation=cv2.INTER_AREA)

        # 3. External disparity computation. SGBM invalid pixels are negative: no cleanup
        #    needed, values <= 0 are classified as invalid by the SDK at ingest.
        disparity_view[:, :] = sgbm.compute(left_half, right_half).astype(np.float32)

        # Graded pseudo-confidence from the local image gradient (texture): SGBM is
        # unreliable on textureless areas. With it, the confidence threshold slider
        # sweeps the density instead of acting as a binary valid/invalid switch.
        grad_x = cv2.Sobel(left_half, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(left_half, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.boxFilter(cv2.magnitude(grad_x, grad_y), cv2.CV_32F, (2 * block_size + 1, 2 * block_size + 1))
        confidence_view[:, :] = np.minimum(grad / 64.0, 1.0)  # ~64 gray-levels of local gradient -> fully confident

        # 4. Ingest
        custom_depth.map = disparity_mat
        custom_depth.confidence = confidence_mat
        custom_depth.timestamp = image_ts.data_ns
        err = zed.ingest_custom_depth(custom_depth)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Ingest failed:", err)

        # 5. Full pipeline on the ingested disparity
        runtime_params.confidence_threshold = cv2.getTrackbarPos("confidence", win_name)
        status = zed.grab(runtime_params)
        if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if status != sl.ERROR_CODE.SUCCESS:
            print("Grab failed:", status)
            continue

        # Metric sanity check: depth at the image center, straight from the ingested disparity
        frame += 1
        if frame % 30 == 0:
            zed.retrieve_measure(depth_measure, sl.MEASURE.DEPTH, sl.MEM.CPU)
            center = depth_measure.get_data(deep_copy=False)[cam_res.height // 2, cam_res.width // 2]
            print(f"Center depth: {center:.2f} m   ", end="\r")

        # Display the SDK depth view, generated from the ingested disparity
        zed.retrieve_image(depth_view, sl.VIEW.DEPTH, sl.MEM.CPU, sl.Resolution(720, 404))
        cv2.imshow(win_name, depth_view.get_data(deep_copy=False))
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    zed.close()


if __name__ == "__main__":
    main()
