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
    This sample shows how to capture a real-time 3D reconstruction      
    of the scene using the Spatial Mapping API. The resulting mesh      
    is displayed as a wireframe on top of the left image using OpenGL.  
    Spatial Mapping can be started and stopped with the Space Bar key
"""
import sys
import time
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import argparse


def main():
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # OpenGL's coordinate system is right_handed    
    init.depth_maximum_distance = 8.
    parse_args(init)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()
    
    camera_infos = zed.get_camera_information()
    pose = sl.Pose()
    
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_floor_as_origin = True
    returned_state = zed.enable_positional_tracking(positional_tracking_parameters)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print("Enable Positional Tracking : "+repr(status)+". Exit program.")
        exit()
    
    if opt.build_mesh:
        spatial_mapping_parameters = sl.SpatialMappingParameters(resolution = sl.MAPPING_RESOLUTION.MEDIUM,mapping_range =  sl.MAPPING_RANGE.MEDIUM,max_memory_usage = 2048,save_texture = False,use_chunk_only = True,reverse_vertex_order = False,map_type = sl.SPATIAL_MAP_TYPE.MESH)
        pymesh = sl.Mesh() 
    else :
        spatial_mapping_parameters = sl.SpatialMappingParameters(resolution = sl.MAPPING_RESOLUTION.MEDIUM,mapping_range =  sl.MAPPING_RANGE.MEDIUM,max_memory_usage = 2048,save_texture = False,use_chunk_only = True,reverse_vertex_order = False,map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD)
        pymesh = sl.FusedPointCloud()

    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED

    
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    
    mapping_activated = False

    image = sl.Mat()  
    point_cloud = sl.Mat()                
    pose = sl.Pose() 

    viewer = gl.GLViewer()
    
    viewer.init(zed.get_camera_information().camera_configuration.calibration_parameters.left_cam, pymesh, int(opt.build_mesh))
    print("Press on 'Space' to enable / disable spatial mapping")
    print("Disable the spatial mapping after enabling it will output a .obj mesh file")
    while viewer.is_available():
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Update pose data (used for projection of the mesh over the current image)
            tracking_state = zed.get_position(pose)

            if mapping_activated:
                mapping_state = zed.get_spatial_mapping_state()
                # Compute elapsed time since the last call of Camera.request_spatial_map_async()
                duration = time.time() - last_call  
                # Ask for a mesh update if 500ms elapsed since last request
                if(duration > .5 and viewer.chunks_updated()):
                    zed.request_spatial_map_async()
                    last_call = time.time()
                
                if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_spatial_map_async(pymesh)
                    viewer.update_chunks()
                
            change_state = viewer.update_view(image, pose.pose_data(), tracking_state, mapping_state)

            if change_state:
                if not mapping_activated:
                    init_pose = sl.Transform()
                    zed.reset_positional_tracking(init_pose)

                    # Configure spatial mapping parameters
                    spatial_mapping_parameters.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.MEDIUM)
                    spatial_mapping_parameters.use_chunk_only = True
                    spatial_mapping_parameters.save_texture = False         # Set to True to apply texture over the created mesh
                    if opt.build_mesh:
                        spatial_mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.MESH
                    else:
                        spatial_mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD

                    # Enable spatial mapping
                    zed.enable_spatial_mapping(spatial_mapping_parameters)

                    # Clear previous mesh data
                    pymesh.clear()
                    viewer.clear_current_mesh()

                    # Start timer
                    last_call = time.time()

                    mapping_activated = True
                else:
                    # Extract whole mesh
                    zed.extract_whole_spatial_map(pymesh)

                    if opt.build_mesh:
                        filter_params = sl.MeshFilterParameters()
                        filter_params.set(sl.MESH_FILTER.MEDIUM) 
                        # Filter the extracted mesh
                        pymesh.filter(filter_params, True)
                        viewer.clear_current_mesh()

                        # If textures have been saved during spatial mapping, apply them to the mesh
                        if(spatial_mapping_parameters.save_texture):
                            print("Save texture set to : {}".format(spatial_mapping_parameters.save_texture))
                            pymesh.apply_texture(sl.MESH_TEXTURE_FORMAT.RGBA)

                    # Save mesh as an obj file
                    filepath = "mesh_gen.obj"
                    status = pymesh.save(filepath)
                    if status:
                        print("Mesh saved under " + filepath)
                    else:
                        print("Failed to save the mesh under " + filepath)
                    
                    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
                    mapping_activated = False
    
    image.free(memory_type=sl.MEM.CPU)
    pymesh.clear()
    # Disable modules and close camera
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()


    # Free allocated memory before closing the camera
    pymesh.clear()
    image.free()
    point_cloud.free()
    # Close the ZED
    zed.close()
   
          
def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
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



        
         
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    parser.add_argument('--build_mesh', help = 'Either the script should plot a mesh or point clouds of surroundings', action='store_true')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 
    