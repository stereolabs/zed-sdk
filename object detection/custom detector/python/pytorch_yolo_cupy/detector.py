#!/usr/bin/env python3

import sys
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

import cupy as cp

import torch
import torch.nn.functional as F

lock = Lock()
run_signal = False
exit_signal = False


def xywh2abcd(xywh):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = max(0, xywh[0] - 0.5*xywh[2])
    x_max = (xywh[0] + 0.5*xywh[2])
    y_min = max(0, xywh[1] - 0.5*xywh[3])
    y_max = (xywh[1] + 0.5*xywh[3])

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

def detections_to_custom_box(detections, original_shape, img_size):
    """
    Convert YOLO detections to ZED CustomBoxObjectData format
    Args:
        detections: YOLO detection results
        original_shape: (height, width) of original camera image
        img_size: Size used for model inference
    """
    output = []
    orig_h, orig_w = original_shape

    # Calculate scaling factors for coordinate transformation
    scale = min(img_size / orig_h, img_size / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    pad_h = (img_size - new_h) // 2
    pad_w = (img_size - new_w) // 2

    for det in detections:
        xywh = det.xywh[0]

        # Transform coordinates back to original image space
        # Remove padding offset
        x_center = (xywh[0] - pad_w) / scale
        y_center = (xywh[1] - pad_h) / scale
        width = xywh[2] / scale
        height = xywh[3] / scale

        # Clamp coordinates to image bounds
        x_center = max(0, min(orig_w, x_center))
        y_center = max(0, min(orig_h, y_center))
        width = max(0, min(orig_w - x_center + width/2, width))
        height = max(0, min(orig_h - y_center + height/2, height))

        transformed_xywh = [x_center, y_center, width, height]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(transformed_xywh)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output

def preprocess_image_torch(image_cupy, img_size):
    """
    Preprocess ZED camera image using PyTorch operations for better performance
    """

    # Convert CuPy array to PyTorch tensor using DLPack for zero-copy transfer
    tensor = torch.from_dlpack(image_cupy.toDlpack())

    # Convert BGRA to RGB and remove alpha channel
    if tensor.shape[2] == 4:  # BGRA format
        tensor = tensor[:, :, [2, 1, 0]]  # BGR to RGB, drop alpha

    # Convert to float and normalize
    tensor = tensor.float() / 255.0

    # Transpose to CHW format and add batch dimension
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW

    # Resize with padding to maintain aspect ratio
    _, _, h, w = tensor.shape
    scale = min(img_size / h, img_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    tensor = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # Pad to square
    pad_h = (img_size - new_h) // 2
    pad_w = (img_size - new_w) // 2
    tensor = F.pad(tensor, (pad_w, img_size - new_w - pad_w, pad_h, img_size - new_h - pad_h), value=114/255.0)

    return tensor

def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net_cupy, exit_signal, run_signal, detections

    print("Intializing Network...")

    model = YOLO(weights)

    while not exit_signal:
        if run_signal:
            lock.acquire()

            # Preprocess the image for YOLO model
            tensor = preprocess_image_torch(image_net_cupy, img_size)

            # https://docs.ultralytics.com/modes/predict
            # Note: we pass the preprocessed tensor directly
            det = model.predict(tensor, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, image_net_cupy.shape[:2], img_size)
            lock.release()
            run_signal = False
        sleep(0.01)


def main(opt):
    global image_net_cupy, exit_signal, run_signal, detections

    # Determine memory type based on CuPy availability and user preference
    use_gpu_for_viz = gl.GPU_ACCELERATION_AVAILABLE and not opt.disable_gpu_data_transfer
    mem_type_viz = sl.MEM.GPU if use_gpu_for_viz else sl.MEM.CPU

    # Display memory type being used
    if use_gpu_for_viz:
        print("🚀 Using GPU data transfer with CuPy")

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)
    if status > sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat(0, 0, sl.MAT_TYPE.U8_C4, sl.MEM.GPU)

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False  # designed to give person pixel mask with internal OD
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, mem_type_viz)
    image_left = sl.Mat(0, 0, sl.MAT_TYPE.U8_C4, mem_type_viz)
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
    # Camera pose
    cam_w_pose = sl.Pose()

    while viewer.is_available() and not exit_signal:
        if zed.grab(runtime_params) <= sl.ERROR_CODE.SUCCESS:

            print(zed.get_current_fps())

            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT, sl.MEM.GPU)
            image_net_cupy = image_left_tmp.get_data(memory_type=sl.MEM.GPU, deep_copy=False)
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_custom_objects(objects, obj_runtime_param)

            # -- Display
            # Retrieve display data
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type_viz, point_cloud_res)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, mem_type_viz, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
            viewer.updateData(point_cloud, objects)
            # 2D rendering
            if use_gpu_for_viz:
                image_left_ocv = cp.asnumpy(cp.asarray(image_left.get_data(memory_type=sl.MEM.GPU, deep_copy=False)))
            else:
                np.copyto(image_left_ocv, image_left.get_data(memory_type=sl.MEM.CPU, deep_copy=False))
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            print("Generating tracking view...", image_track_ocv.shape)
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

            cv2.imshow("ZED | 2D View and Birds View", global_image)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q') or key == ord('Q'):
                exit_signal = True
        else:
            exit_signal = True

    viewer.exit()
    exit_signal = True
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo11m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file, if not passed, use the plugged camera instead')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--disable-gpu-data-transfer', action='store_true', help='Disable GPU data transfer acceleration with CuPy even if CuPy is available')
    opt = parser.parse_args()

    with torch.no_grad():
        main(opt)
