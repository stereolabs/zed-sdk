# This is a modified version of depth_sensing.py that uses our GLFW-based viewer
import sys
import pyzed.sl as sl
import numpy as np
import argparse
from pathlib import Path
import time

# Import our modified GLFW-based viewer module
from ogl_viewer import glfw_viewer as viewer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=str, default="HD720",
                        help='Resolution, the default is HD720')
    parser.add_argument('--rate', type=int, default=30,
                        help='Rate (fps), the default is 30')
    return parser.parse_args()

def main(opt):
    print("Running Depth Sensing sample ... Press 'Esc' to quit")
    print("Press 's' to save the point cloud")

    # Set configuration parameters
    input_type = sl.InputType()
    if len(opt.input_svo_file) > 0:
        input_type.set_from_svo_file(opt.input_svo_file)

    # Create ZED objects
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD720
    if opt.resolution == "HD2K":
        init.camera_resolution = sl.RESOLUTION.HD2K
    elif opt.resolution == "HD1080":
        init.camera_resolution = sl.RESOLUTION.HD1080
    elif opt.resolution == "HD720":
        init.camera_resolution = sl.RESOLUTION.HD720
    elif opt.resolution == "VGA":
        init.camera_resolution = sl.RESOLUTION.VGA
    else:
        print("[Sample] Using default resolution HD720")
    
    # Use the most accurate depth map
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    # init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    
    # Open the camera
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    # Get camera information
    camera_info = zed.get_camera_information()
    res = camera_info.camera_configuration.resolution
    
    # Initialize point cloud and GL viewer
    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    point_cloud_gpu = sl.Mat()
    # cpu_point_cloud = sl.Mat()
    
    # Create OpenGL viewer
    viewer3D = viewer.GLViewer()
    if not viewer3D.init(0, [], res):
        zed.close()
        exit(1)
    
    # Configure depth map
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    runtime_parameters.texture_confidence_threshold = 100
    
    while viewer3D.is_available():
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the colored point cloud in GPU memory
            # We do this by mapping the RGBA buffer to OpenGL sampling units
            # Using retrieve_measure with MEASURE.XYZRGBA maps the point cloud to the GPU memory directly
            # err = zed.retrieve_measure(point_cloud_gpu, sl.MEASURE.XYZRGBA, sl.MEM.GPU)
            zed.retrieve_measure(point_cloud_gpu, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
            point_cloud_data = point_cloud_gpu.get_data()
            if np.isnan(point_cloud_data).any():
                print("WARNING: NaN values detected in point cloud! Replacing NaNs...")
                point_cloud_data = np.nan_to_num(point_cloud_data, nan=0.0)  # Replace NaN with 0.0
            #breakpoint()
            import ctypes
            mat_ptr = point_cloud_gpu.get_pointer()
            arr_contiguous = np.ascontiguousarray(point_cloud_data)
            arr_ptr = arr_contiguous.ctypes.data_as(ctypes.c_void_p)
            buffer_size = arr_contiguous.nbytes
            ctypes.memmove(mat_ptr, arr_ptr, buffer_size)
            #point_cloud_gpu.set_from_numpy(point_cloud_data, sl.COPY_TYPE.CPU_GPU)

            #zed.retrieve_measure(cpu_point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
            #cpu_data = cpu_point_cloud.get_data()
            #print("Point Cloud Min/Max XYZ:", np.nanmin(cpu_data[..., 0:3]), np.nanmax(cpu_data[..., 0:3]))
            #print("Point Cloud First 5 Points:", cpu_data[:5, :5, :])  # Check a small section
            
            # Update the viewer
            success = viewer3D.updateData(point_cloud_gpu)
            if not success:
                print("viewer3D.updateData() failed!")
            
            # Render the point cloud in the viewer
            if not viewer3D.render():
                break
            
            # If the 's' key is pressed, save the point cloud
            if viewer3D.save_data:
                # Retrieve the point cloud
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                
                # Save the point cloud
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                filename = f"point_cloud_{timestamp}.ply"
                print(f"Saving point cloud to {filename}")
                
                err = point_cloud.write(str(Path.home() / filename), sl.MEM.CPU)
                if err != sl.ERROR_CODE.SUCCESS:
                    print(f"Failed to save point cloud: {err}")
                
                viewer3D.save_data = False

    # Close the camera
    zed.close()

if __name__ == "__main__":
    args = parse_args()
    # Add a default empty input_svo_file field to args to ensure compatibility
    args.input_svo_file = ""
    main(args)
