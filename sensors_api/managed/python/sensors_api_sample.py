#!/usr/bin/env python3
"""
Sensors API Sample - Demonstrates multi-sensor streaming with ZED cameras and LiDAR

This sample shows how to use the pyzed.Sensors API to manage multiple sensors
(ZED cameras and LiDAR) with a unified interface. It supports:
- Live streaming from connected devices
- Reading from recorded files (SVO for cameras, OSF for LiDAR)
- Recording to files while streaming
- Object detection on ZED cameras (3D bounding boxes)
- Visualization with OpenCV, 3D point cloud (OpenGL), or headless mode

Controls:
- R: Toggle recording (SVO for cameras, OSF for LiDAR)
- Q/ESC: Quit

Usage:
    python sensors_api_sample.py                              # Auto-detect all connected sensors
    python sensors_api_sample.py sensors_config.json          # Use JSON configuration
    python sensors_api_sample.py recording.svo                # Play back SVO file
    python sensors_api_sample.py recording.osf                # Play back OSF file
    python sensors_api_sample.py recording.svo recording.osf --realtime  # Play back at recorded frame rate
    python sensors_api_sample.py --od                         # Enable object detection on ZED cameras
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import pyzed.sl as sl
import numpy as np
import cv2

# Optional OpenGL point cloud viewer
try:
    from ogl_viewer.viewer import GLViewer
    HAS_GL_VIEWER = True
except ImportError:
    HAS_GL_VIEWER = False


# =============================================================================
# Helper Types
# =============================================================================

class ViewerType(Enum):
    OPENCV = 1
    HEADLESS = 2
    OPENGL = 3


class InputMode(Enum):
    LIVE = 1
    PLAYBACK = 2


class DeviceType(Enum):
    LIDAR = 1
    ZED = 2
    ZED_ONE = 3


@dataclass
class DeviceConfig:
    type: DeviceType
    input: sl.InputType
    pose: sl.Matrix4f = field(default_factory=sl.Matrix4f)
    has_pose: bool = False


@dataclass
class RecordingState:
    is_recording: bool = False
    toggle_requested: bool = False
    last_status: str = ""
    output_dir: str = "."


# Global recording state
g_recording_state = RecordingState()


# =============================================================================
# Helper Functions
# =============================================================================

def generate_recording_filename(prefix: str, extension: str) -> str:
    """Generate timestamped recording filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{extension}"


def parse_json_config(path: str) -> List[DeviceConfig]:
    """Parse JSON configuration file for multi-sensor setup"""
    devices = []
    
    try:
        with open(path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to open or parse config file: {e}")
        return devices
    
    # Parse ZED cameras
    if "zeds" in config and isinstance(config["zeds"], list):
        for zed in config["zeds"]:
            cfg = DeviceConfig(type=DeviceType.ZED, input=sl.InputType())
            
            if "serial" in zed:
                cfg.input.set_from_serial_number(zed["serial"])
            if "svo" in zed:
                cfg.input.set_from_svo_file(zed["svo"])
            
            if "pose" in zed and isinstance(zed["pose"], list) and len(zed["pose"]) == 16:
                cfg.pose = sl.Matrix4f()
                # Matrix4f uses 2D indexing [row, col], convert from 1D row-major index
                for i in range(16):
                    row, col = i // 4, i % 4
                    cfg.pose[row, col] = zed["pose"][i]
                cfg.has_pose = True
            
            devices.append(cfg)
    
    # Parse LiDARs
    if "lidars" in config and isinstance(config["lidars"], list):
        for lidar in config["lidars"]:
            cfg = DeviceConfig(type=DeviceType.LIDAR, input=sl.InputType())
            
            if "serial" in lidar:
                # Support serial number (string-based for Lidar)
                cfg.input.set_from_serial_number(str(lidar["serial"]))
            elif "ip" in lidar:
                cfg.input.set_from_stream(lidar["ip"])
            if "osf" in lidar:
                cfg.input.set_from_svo_file(lidar["osf"])
            if "svo" in lidar:
                cfg.input.set_from_svo_file(lidar["svo"])
            
            if "pose" in lidar and isinstance(lidar["pose"], list) and len(lidar["pose"]) == 16:
                cfg.pose = sl.Matrix4f()
                # Matrix4f uses 2D indexing [row, col], convert from 1D row-major index
                for i in range(16):
                    row, col = i // 4, i % 4
                    cfg.pose[row, col] = lidar["pose"][i]
                cfg.has_pose = True
            
            devices.append(cfg)
    
    return devices


def get_lidar_params(input_type: sl.InputType, real_time_mode: bool = False) -> sl.InitLidarParameters:
    """Get LiDAR initialization parameters"""
    params = sl.InitLidarParameters()
    params.input = input_type
    params.mode = sl.LIDAR_MODE.AUTO
    params.depth_minimum_distance = 1.0
    params.depth_maximum_distance = 80.0
    params.coordinate_units = sl.UNIT.METER
    params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    params.svo_real_time_mode = real_time_mode
    # Enable multicast AUTO mode: allows multiple clients to receive data from the same sensor
    # AUTO will join existing multicast passively or stay unicast if alone
    params.multicast_mode = sl.LIDAR_MULTICAST_MODE.AUTO
    # Enable auto recovery: SDK will automatically reconnect if sensor config changes externally
    params.auto_recovery_on_config_change = True
    return params


def get_zed_params(input_type: sl.InputType, is_playback: bool = False) -> sl.InitParameters:
    """Get ZED camera initialization parameters"""
    params = sl.InitParameters()
    params.input = input_type
    params.depth_mode = sl.DEPTH_MODE.NEURAL
    params.coordinate_units = sl.UNIT.METER
    params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    # For playback, disable real-time mode to read all frames without timing-based drops
    # For live capture, enable real-time mode for natural pacing
    params.svo_real_time_mode = not is_playback
    params.async_grab_camera_recovery = True
    return params


def get_zed_one_params(input_type: sl.InputType) -> sl.InitParametersOne:
    """Get ZED One camera initialization parameters (2D only, no depth)"""
    params = sl.InitParametersOne()
    params.input = input_type
    params.coordinate_units = sl.UNIT.METER
    params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    return params


def display_data_opencv(batched_images: Dict, batched_measures: Dict, 
                        lidar_ids: List, zed_ids: List, zed_one_ids: List, frame_count: int,
                        recording_status: str) -> bool:
    """
    Display data using OpenCV
    Returns True if should continue, False if user wants to quit
    """
    global g_recording_state
    
    images_to_display = []
    
    # Display ZED camera images
    for zed_id in zed_ids:
        if zed_id in batched_images:
            mat = batched_images[zed_id]
            if mat.is_init():
                # Convert sl.Mat to numpy array
                img = mat.get_data()
                if img is not None and img.size > 0:
                    # Convert BGRA to BGR if needed
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    
                    # Add text overlay
                    text = f"ZED {zed_id.get_serial_number()} | Frame: {frame_count}"
                    if recording_status:
                        text += f" | {recording_status}"
                    
                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    images_to_display.append(img)

    # Display ZED One camera images (2D only)
    for zed_one_id in zed_one_ids:
        if zed_one_id in batched_images:
            mat = batched_images[zed_one_id]
            if mat.is_init():
                img = mat.get_data()
                if img is not None and img.size > 0:
                    # Convert BGRA to BGR if needed
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    text = f"ZED One {zed_one_id.get_serial_number()} | Frame: {frame_count}"
                    if recording_status:
                        text += f" | {recording_status}"

                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (255, 200, 0), 2)
                    images_to_display.append(img)

    # Display LiDAR intensity images
    for lidar_id in lidar_ids:
        if lidar_id in batched_images:
            mat = batched_images[lidar_id]
            if mat.is_init():
                img = mat.get_data()
                if img is not None and img.size > 0:
                    # Normalize grayscale image for display
                    if len(img.shape) == 2:
                        normalized = np.empty_like(img, dtype=np.uint8)
                        cv2.normalize(img, normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

                    text = f"LiDAR {lidar_id.get_serial_number()} | Frame: {frame_count}"
                    if recording_status:
                        text += f" | {recording_status}"

                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 255), 2)
                    images_to_display.append(img)

    # Create composite display
    if images_to_display:
        # Stack images horizontally if multiple, or show single image
        if len(images_to_display) == 1:
            display = images_to_display[0]
        else:
            # Resize all images to same height
            target_height = 480
            resized = []
            for img in images_to_display:
                h, w = img.shape[:2]
                new_w = int(w * target_height / h)
                resized.append(cv2.resize(img, (new_w, target_height)))
            display = np.hstack(resized)
        
        cv2.imshow("Sensors API - Press R to record, Q to quit", display)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # Q or ESC
        return False
    elif key == ord('r') or key == ord('R'):
        g_recording_state.toggle_requested = True
    
    return True


# =============================================================================
# Main Sensors API Implementation
# =============================================================================

def run_sensors_impl(args, viewer_type: ViewerType):
    """Main implementation of the Sensors API sample"""
    global g_recording_state
    
    print("\n=== Sensors API Sample ===")
    
    devices = []
    input_mode = InputMode.LIVE
    real_time_playback = args.realtime
    enable_object_detection = args.od
    
    # Parse input files from command line
    for input_file in args.inputs:
        path = Path(input_file)
        suffix = path.suffix.lower()
        
        # JSON config file
        if suffix == ".json":
            devices = parse_json_config(input_file)
            if devices:
                input_mode = InputMode.PLAYBACK
                print(f"Loaded JSON config with {len(devices)} devices")
        
        # SVO file (Camera playback)
        elif suffix in [".svo", ".svo2"]:
            cfg = DeviceConfig(type=DeviceType.ZED, input=sl.InputType())
            cfg.input.set_from_svo_file(input_file)
            devices.append(cfg)
            input_mode = InputMode.PLAYBACK
            print(f"Playing back SVO file: {input_file}")
        
        # OSF file (LiDAR playback)
        elif suffix == ".osf":
            cfg = DeviceConfig(type=DeviceType.LIDAR, input=sl.InputType())
            cfg.input.set_from_svo_file(input_file)
            devices.append(cfg)
            input_mode = InputMode.PLAYBACK
            print(f"Playing back OSF file: {input_file}")
        
        else:
            print(f"Unknown file type: {input_file}")
            print("Supported: .json (config), .svo/.svo2 (camera), .osf (lidar)")
            return
    
    # If no config provided, auto-detect devices
    if not devices:
        print("No config provided, auto-detecting devices...")
        
        # Detect ZED cameras
        zed_devices = sl.Camera.get_device_list()
        for dev in zed_devices:
            cfg = DeviceConfig(type=DeviceType.ZED, input=sl.InputType())
            cfg.input.set_from_serial_number(dev.serial_number)
            devices.append(cfg)
            print(f"  Found ZED camera: {dev.serial_number}")
        
        # Detect ZED One cameras
        zed_one_devices = sl.CameraOne.get_device_list()
        for dev in zed_one_devices:
            cfg = DeviceConfig(type=DeviceType.ZED_ONE, input=sl.InputType())
            cfg.input.set_from_camera_id(dev.id)
            devices.append(cfg)
            print(f"  Found ZED One camera: {dev.serial_number}")

        # Detect LiDARs
        lidar_devices = sl.Lidar.get_device_list()
        for dev in lidar_devices:
            cfg = DeviceConfig(type=DeviceType.LIDAR, input=sl.InputType())
            # Use serial number for discovery (more reliable than IP which may change)
            serial = dev.lidar_information.serial_number
            cfg.input.set_from_serial_number(serial)
            devices.append(cfg)
            print(f"  Found LiDAR: {dev.name} (SN: {serial}, IP: {dev.ip_address})")
        
        if not devices:
            print("No devices found!")
            return
    
    # Create and initialize Sensors manager
    sensors = sl.Sensors()
    init_sensors_params = sl.InitSensorsParameters()
    init_sensors_params.coordinate_units = sl.UNIT.METER
    init_sensors_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    print("Initializing Sensors API...")
    init_err = sensors.init(init_sensors_params)
    if init_err != sl.SENSORS_ERROR_CODE.SUCCESS:
        print(f"Failed to init Sensors API: {init_err}")
        return
    
    # Add devices
    lidar_ids = []
    zed_ids = []
    zed_one_ids = []
    
    for dev in devices:
        sensor_id = sl.SensorDeviceIdentifier()
        
        if dev.type == DeviceType.LIDAR:
            # For playback: use real_time_playback flag to control pacing
            # For live: always use real-time (natural sensor rate)
            use_real_time = (input_mode == InputMode.LIVE) or real_time_playback
            params = get_lidar_params(dev.input, use_real_time)
            print(f"Opening LiDAR at {dev.input.get_configuration()}...")
            err = sensors.add(params, sensor_id)
            if err == sl.SENSORS_ERROR_CODE.SUCCESS:
                if dev.has_pose:
                    sensors.set_sensor_pose(sl.Transform(dev.pose), sensor_id)
                lidar_ids.append(sensor_id)
                print(f"  LiDAR Added. ID: {sensor_id.get_id()}")
            else:
                print(f"  Failed to add LiDAR: {err}")
        
        elif dev.type == DeviceType.ZED:
            # For playback: use real_time_playback to pace based on timestamps
            # Default playback (no --realtime): read as fast as possible
            use_real_time = (input_mode == InputMode.LIVE) or real_time_playback
            params = get_zed_params(dev.input, not use_real_time)  # isPlayback inverts the logic
            print(f"Opening ZED at {dev.input.get_configuration()}...")
            err = sensors.add(params, sensor_id)
            if err == sl.SENSORS_ERROR_CODE.SUCCESS:
                if dev.has_pose:
                    sensors.set_sensor_pose(sl.Transform(dev.pose), sensor_id)
                zed_ids.append(sensor_id)
                print(f"  ZED Added. ID: {sensor_id.get_id()}")
            else:
                print(f"  Failed to add ZED: {err}")

        elif dev.type == DeviceType.ZED_ONE:
            params = get_zed_one_params(dev.input)
            print(f"Opening ZED One at {dev.input.get_configuration()}...")
            err = sensors.add(params, sensor_id)
            if err == sl.SENSORS_ERROR_CODE.SUCCESS:
                if dev.has_pose:
                    sensors.set_sensor_pose(sl.Transform(dev.pose), sensor_id)
                zed_one_ids.append(sensor_id)
                print(f"  ZED One Added. ID: {sensor_id.get_id()}")
            else:
                print(f"  Failed to add ZED One: {err}")
    
    # Check if we have any active sensors
    if not lidar_ids and not zed_ids and not zed_one_ids:
        print("No sensors were successfully added!")
        return
    
    # Enable object detection for ZED cameras if requested
    od_enabled = False
    if enable_object_detection and zed_ids:
        od_params = sl.ObjectDetectionSensorsParameters()
        od_params.enable_tracking = True
        od_params.enable_segmentation = False
        od_params.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
        # Use all ZED cameras for object detection
        od_params.sensors_ids = zed_ids
        
        od_err = sensors.enable_object_detection(od_params)
        if od_err == sl.SENSORS_ERROR_CODE.SUCCESS:
            print(f"Object detection enabled for {len(zed_ids)} ZED camera(s)")
            od_enabled = True
            
            # Set runtime parameters
            od_rt = sl.ObjectDetectionRuntimeParameters()
            od_rt.detection_confidence_threshold = 50.0
            sensors.set_object_detection_runtime_parameters(od_rt)
        else:
            print(f"Failed to enable object detection in Sensors API: {od_err}")
    
    # In playback mode, synchronize all SVO/OSF files to a common start timestamp
    if input_mode == InputMode.PLAYBACK:
        print("Synchronizing playback files to common start timestamp...")
        sync_err = sensors.sync_svo()
        if sync_err != sl.SENSORS_ERROR_CODE.SUCCESS:
            print(f"Warning: Failed to sync some files to common timestamp: {sync_err}")
        else:
            print("Playback synchronized successfully")
    
    # Initialize viewers
    headless = (viewer_type == ViewerType.HEADLESS)
    use_opengl = (viewer_type == ViewerType.OPENGL)
    gl_viewer = None

    if use_opengl:
        gl_viewer = GLViewer()
        gl_viewer.init()  # Point cloud buffer sizes are auto-detected from sl.Mat dimensions
        cv2.namedWindow("Sensors API - Press R to record, Q to quit", cv2.WINDOW_NORMAL)
        print("\nControls: R=toggle recording, Q/ESC=quit")
        print("OpenGL point cloud viewer active (mouse to rotate/pan, scroll to zoom)")
    elif not headless:
        cv2.namedWindow("Sensors API - Press R to record, Q to quit", cv2.WINDOW_NORMAL)
        print("\nControls: R=toggle recording, Q=quit")
    else:
        print("Running in headless mode (no display)")
        print("Press Ctrl+C to stop")
    
    print("\nStarting sensor streaming...")
    
    frame_count = 0
    start_time = time.time()
    running = True
    
    # Pre-allocate Mat objects for each sensor
    # Note: Python API uses single-sensor retrieval, not batched like C++
    lidar_images = {lid: sl.Mat() for lid in lidar_ids}
    lidar_measures = {lid: sl.Mat() for lid in lidar_ids}
    zed_images = {zid: sl.Mat() for zid in zed_ids}
    zed_measures = {zid: sl.Mat() for zid in zed_ids}
    zed_one_images = {zoid: sl.Mat() for zoid in zed_one_ids}
    
    try:
        # Main loop
        while running:
            # Handle recording toggle
            if g_recording_state.toggle_requested:
                g_recording_state.toggle_requested = False
                
                if g_recording_state.is_recording:
                    # Stop recording
                    sensors.disable_recording()
                    g_recording_state.is_recording = False
                    g_recording_state.last_status = "Recording stopped"
                    print("Recording stopped")
                else:
                    # Start recording - create filenames for all devices
                    # Build video_filenames dictionary first (required for constructor)
                    video_filenames = {}
                    
                    print("Starting recording:")
                    # Set output filenames for each device
                    for sensor_id in zed_ids:
                        filename = generate_recording_filename(
                            f"zed_{sensor_id.get_serial_number()}", ".svo2")
                        video_filenames[sensor_id] = filename
                        print(f"  Camera {sensor_id.get_serial_number()} -> {filename}")
                    
                    for sensor_id in lidar_ids:
                        filename = generate_recording_filename(
                            f"lidar_{sensor_id.sensor_name}", ".osf")
                        video_filenames[sensor_id] = filename
                        print(f"  LiDAR {sensor_id.sensor_name} -> {filename}")
                    
                    # Create RecordingSensorsParameters with filenames and compression mode
                    rec_params = sl.RecordingSensorsParameters(
                        video_filenames=video_filenames,
                        compression_mode=sl.SVO_COMPRESSION_MODE.H265
                    )
                    
                    rec_err = sensors.enable_recording(rec_params)
                    if rec_err == sl.SENSORS_ERROR_CODE.SUCCESS:
                        g_recording_state.is_recording = True
                        g_recording_state.last_status = "RECORDING"
                        print("Recording started successfully")
                    else:
                        print(f"Failed to start recording: {rec_err}")
                        g_recording_state.last_status = f"Recording failed: {rec_err}"
            
            # Process data from all sensors
            process_status = sensors.process()
            if process_status != sl.SENSORS_ERROR_CODE.SUCCESS:
                if (input_mode == InputMode.PLAYBACK and 
                    process_status == sl.SENSORS_ERROR_CODE.END_OF_SVOFILE_REACHED):
                    print("End of all playback files reached")
                    break
                time.sleep(0.001)
                continue
            
            # Retrieve data from each sensor individually
            # Note: Python API uses single-sensor retrieval, not batched like C++
            # Resolution(0,0) = optimal size (internal compute size for cameras, native for LiDAR)
            
            # Retrieve LiDAR data
            for lid in lidar_ids:
                sensors.retrieve_image(lidar_images[lid], sl.VIEW.LEFT, sl.MEM.CPU, 
                                      sl.Resolution(0, 0), lid)
                sensors.retrieve_measure(lidar_measures[lid], sl.MEASURE.XYZRGBA, 
                                        sl.MEM.CPU, sl.Resolution(0, 0), lid)
            
            # Retrieve ZED camera data
            for zid in zed_ids:
                sensors.retrieve_image(zed_images[zid], sl.VIEW.LEFT, sl.MEM.CPU,
                                      sl.Resolution(0, 0), zid)
                sensors.retrieve_measure(zed_measures[zid], sl.MEASURE.XYZRGBA,
                                        sl.MEM.CPU, sl.Resolution(0, 0), zid)
                
                # Retrieve and display objects if OD is enabled
                if od_enabled:
                    objects = sl.Objects()
                    od_err = sensors.retrieve_objects(objects, zid)
                    if od_err == sl.SENSORS_ERROR_CODE.SUCCESS:
                        num_objects = len(objects.object_list)
                        if gl_viewer is not None:
                            gl_viewer.updateObjects(objects, f"ZED_{zid.get_serial_number()}")

            # Retrieve ZED One camera data (2D image only, no depth/point cloud)
            for zoid in zed_one_ids:
                sensors.retrieve_image(zed_one_images[zoid], sl.VIEW.LEFT, sl.MEM.CPU,
                                      sl.Resolution(0, 0), zoid)

            # Feed point clouds to OpenGL viewer
            if gl_viewer is not None:
                for zid in zed_ids:
                    if zid in zed_measures:
                        gl_viewer.updateData(zed_measures[zid], f"ZED_{zid.get_serial_number()}")
                for lid in lidar_ids:
                    if lid in lidar_measures:
                        gl_viewer.updateData(lidar_measures[lid], f"LiDAR_{lid.get_serial_number()}")
                if not gl_viewer.is_available():
                    running = False

            # Build batched dictionaries for display function
            batched_images = {**lidar_images, **zed_images, **zed_one_images}
            batched_measures = {**lidar_measures, **zed_measures}
            
            # Display data if not headless
            if not headless:
                if not display_data_opencv(batched_images, batched_measures, 
                                          lidar_ids, zed_ids, zed_one_ids, frame_count,
                                          g_recording_state.last_status):
                    running = False
            
            frame_count += 1
            
            # Print FPS every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                status = f"FPS: {fps:.1f} | Cameras: {len(zed_ids)} | CameraOnes: {len(zed_one_ids)} | LiDARs: {len(lidar_ids)}"
                if g_recording_state.is_recording:
                    status += " | RECORDING"
                print(status)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        if g_recording_state.is_recording:
            sensors.disable_recording()
        
        if gl_viewer is not None:
            gl_viewer.exit()

        sensors.close()
        
        if not headless:
            cv2.destroyAllWindows()
        
        print(f"\nSample finished. Total frames: {frame_count}")


# =============================================================================
# Main Entry Point
# =============================================================================

def print_usage():
    """Print detailed usage information"""
    usage = """
Usage: python sensors_api_sample.py [options] [input...]

Input (optional, multiple files supported):
  config.json     JSON configuration file for multiple sensors
  recording.svo   SVO file for camera playback
  recording.osf   OSF file for LiDAR playback
  (none)          Auto-detect all connected sensors

Examples:
  python sensors_api_sample.py                                    # Auto-detect devices
  python sensors_api_sample.py camera.svo2                        # Play single camera
  python sensors_api_sample.py lidar.osf                          # Play single LiDAR
  python sensors_api_sample.py camera.svo2 lidar.osf              # Play both together
  python sensors_api_sample.py cam1.svo2 cam2.svo2 lidar.osf      # Play multiple files
  python sensors_api_sample.py config.json                        # Use JSON config
  python sensors_api_sample.py --od                               # Enable object detection

Options:
  --headless              Run without display (data processing only)
  --opengl, --gl          Enable 3D point cloud viewer (requires PyOpenGL)
  --od                    Enable object detection on ZED cameras (3D bounding boxes)
  --realtime, -rt         Use real-time playback mode (pace by timestamps)
  --help, -h              Show this help message

Controls during runtime:
  R               Toggle recording (SVO for cameras, OSF for LiDAR)
  Q/ESC           Quit

JSON config example:
{
  "zeds": [
    {"serial": 12345678, "pose": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]},
    {"svo": "recording.svo"}
  ],
  "lidars": [
    {"serial": "991937000508"},
    {"ip": "192.168.1.100"},
    {"osf": "recording.osf", "pose": [1,0,0,1000, 0,1,0,0, 0,0,1,500, 0,0,0,1]}
  ]
}
"""
    print(usage)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Sensors API Sample - Multi-sensor streaming with ZED cameras and LiDAR",
        add_help=False
    )
    
    parser.add_argument('inputs', nargs='*', 
                       help='Input files (.json, .svo, .svo2, .osf)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without display (data processing only)')
    parser.add_argument('--opengl', '--gl', action='store_true',
                       help='Enable 3D point cloud viewer (requires PyOpenGL)')
    parser.add_argument('--od', action='store_true',
                       help='Enable object detection on ZED cameras (3D bounding boxes)')
    parser.add_argument('--realtime', '-rt', action='store_true',
                       help='Use real-time playback mode (pace by timestamps)')
    parser.add_argument('--help', '-h', action='store_true',
                       help='Show this help message')
    
    args = parser.parse_args()
    
    if args.help:
        print_usage()
        return 0
    
    # Log if object detection is enabled
    if args.od:
        print("Object detection enabled for ZED cameras")
    
    # Determine viewer type
    if args.headless:
        viewer_type = ViewerType.HEADLESS
    elif args.opengl:
        if not HAS_GL_VIEWER:
            print("Error: OpenGL viewer requires PyOpenGL. Install with: pip install PyOpenGL PyOpenGL_accelerate")
            return 1
        viewer_type = ViewerType.OPENGL
    else:
        viewer_type = ViewerType.OPENCV
    
    try:
        run_sensors_impl(args, viewer_type)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
