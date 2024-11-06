#!/usr/bin/env python3

import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from ultralytics.engine.results import Results

from threading import Lock, Thread
from time import sleep
from typing import List

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

# Globals
lock = Lock()
run_signal = False
exit_signal = False
image_net: np.ndarray = None
detections: List[sl.CustomMaskObjectData] = None
sl_mats: List[sl.Mat] = None  # We need it to keep the ownership of the sl.Mat


def xywh2abcd_(xywh: np.ndarray) -> np.ndarray:
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5 * xywh[2])
    x_max = (xywh[0] + 0.5 * xywh[2])
    y_min = (xywh[1] - 0.5 * xywh[3])
    y_max = (xywh[1] + 0.5 * xywh[3])

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output


def detections_to_custom_masks_(dets: Results) -> List[sl.CustomMaskObjectData]:
    global sl_mats
    output = []
    sl_mats = []
    for det in dets.cpu().numpy():
        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomMaskObjectData()

        # Bounding box
        box = det.boxes
        xywh = box.xywh[0]
        abcd = xywh2abcd_(xywh)

        obj.bounding_box_2d = abcd
        obj.label = box.cls
        obj.probability = box.conf
        obj.is_grounded = False

        # Mask
        if det.masks is not None:
            mask_bin = (det.masks.data[0] * 255).astype(np.uint8)
            mask_bin = mask_bin[int(abcd[0][1]): int(abcd[2][1]),
                                int(abcd[0][0]): int(abcd[2][0])]
            if not mask_bin.flags.c_contiguous:
                mask_bin = np.ascontiguousarray(mask_bin)

            # Mask as a sl mat
            sl_mat = sl.Mat(width=mask_bin.shape[1],
                            height=mask_bin.shape[0],
                            mat_type=sl.MAT_TYPE.U8_C1,
                            memory_type=sl.MEM.CPU)
            p_sl_as_cv = sl_mat.get_data()
            np.copyto(p_sl_as_cv, mask_bin)
            sl_mats.append(sl_mat)

            obj.box_mask = sl_mat
        else:
            print("[Warning] No mask found in the prediction. Did you use a seg model?")

        output.append(obj)

    return output


def torch_thread_(weights: str, img_size: int, conf_thres: float = 0.2, iou_thres: float = 0.45) -> None:
    global image_net, exit_signal, run_signal, detections

    print("Initializing Network...")
    model = YOLO(weights)
    print("Network Initialized...")

    while not exit_signal:
        if run_signal:
            lock.acquire()

            # Run inference
            img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
            det = model.predict(img, save=False, retina_masks=True, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=False)[0]

            # ZED CustomMasks format
            detections = detections_to_custom_masks_(det)
            lock.release()
            run_signal = False

        sleep(0.01)


def main_(args: argparse.Namespace):
    global image_net, exit_signal, run_signal, detections

    capture_thread = Thread(target=torch_thread_,
                            kwargs={'weights': args.weights, 'img_size': args.img_size, "conf_thres": args.conf_thres})
    capture_thread.start()

    # Create a InitParameters object and set configuration parameters
    input_type = sl.InputType()
    if args.svo is not None:
        input_type.set_from_svo_file(args.svo)
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    # Initialize the camera
    print("Initializing Camera...")
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    print("Camera Initialized")

    # Enable Positional Tracking
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    # Enable Object Detection
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = True
    zed.enable_object_detection(obj_param)

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()

    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

    # Prepare runtime retrieval
    runtime_params = sl.RuntimeParameters()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    cam_w_pose = sl.Pose()
    image_left_tmp = sl.Mat()
    objects = sl.Objects()

    while viewer.is_available() and not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_mask_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)

            # -- Display
            # Retrieve display data
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
            viewer.updateData(point_cloud_render, objects)
            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

            cv2.imshow("ZED | 2D View and Birds View", global_image)
            key = cv2.waitKey(10)
            if key in (27, ord('q'), ord('Q')):
                exit_signal = True
        else:
            exit_signal = True

    viewer.exit()
    exit_signal = True
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    args = parser.parse_args()

    with torch.no_grad():
        main_(args)
