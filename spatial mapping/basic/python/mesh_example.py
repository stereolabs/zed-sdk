########################################################################
#
# Copyright (c) 2020, STEREOLABS.
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
    Mesh sample shows mesh information after filtering and applying texture on frames. The mesh and its filter
    parameters can be saved.
"""
import sys
import pyzed.sl as sl


def main():

    if len(sys.argv) != 2:
        print("Please specify path to .svo file.")
        exit()

    filepath = sys.argv[1]
    print("Reading SVO file: {0}".format(filepath))

    cam = sl.Camera()
    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)
    init = sl.InitParameters()
    init.input = input_type
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    spatial = sl.SpatialMappingParameters()
    transform = sl.Transform()
    tracking = sl.PositionalTrackingParameters(transform)

    cam.enable_positional_tracking(tracking)
    cam.enable_spatial_mapping(spatial)

    pymesh = sl.Mesh()
    print("Processing...")
    for i in range(200):
        cam.grab(runtime)
        cam.request_spatial_map_async()

    cam.extract_whole_spatial_map(pymesh)
    cam.disable_positional_tracking()
    cam.disable_spatial_mapping()

    filter_params = sl.MeshFilterParameters()
    filter_params.set(sl.MESH_FILTER.HIGH)
    print("Filtering params : {0}.".format(pymesh.filter(filter_params)))

    apply_texture = pymesh.apply_texture(sl.MESH_TEXTURE_FORMAT.RGBA)
    print("Applying texture : {0}.".format(apply_texture))
    print_mesh_information(pymesh, apply_texture)

    save_filter(filter_params)
    save_mesh(pymesh)
    cam.close()
    print("\nFINISH")


def print_mesh_information(pymesh, apply_texture):
    while True:
        res = input("Do you want to display mesh information? [y/n]: ")
        if res == "y":
            if apply_texture:
                print("Vertices : \n{0} \n".format(pymesh.vertices))
                print("Uv : \n{0} \n".format(pymesh.uv))
                print("Normals : \n{0} \n".format(pymesh.normals))
                print("Triangles : \n{0} \n".format(pymesh.triangles))
                break
            else:
                print("Cannot display information of the sl.")
                break
        if res == "n":
            print("Mesh information will not be displayed.")
            break
        else:
            print("Error, please enter [y/n].")


def save_filter(filter_params):
    while True:
        res = input("Do you want to save the mesh filter parameters? [y/n]: ")
        if res == "y":
            params = sl.ERROR_CODE.FAILURE
            while params != sl.ERROR_CODE.SUCCESS:
                filepath = input("Enter filepath name : ")
                params = filter_params.save(filepath)
                print("Saving mesh filter parameters: {0}".format(repr(params)))
                if params:
                    break
                else:
                    print("Help : you must enter the filepath + filename without extension.")
            break
        elif res == "n":
            print("Mesh filter parameters will not be saved.")
            break
        else:
            print("Error, please enter [y/n].")


def save_mesh(pymesh):
    while True:
        res = input("Do you want to save the mesh? [y/n]: ")
        if res == "y":
            msh = sl.ERROR_CODE.FAILURE
            while msh != sl.ERROR_CODE.SUCCESS:
                filepath = input("Enter filepath name: ")
                msh = pymesh.save(filepath)
                print("Saving mesh: {0}".format(repr(msh)))
                if msh:
                    break
                else:
                    print("Help : you must enter the filepath + filename without extension.")
            break
        elif res == "n":
            print("Mesh will not be saved.")
            break
        else:
            print("Error, please enter [y/n].")

if __name__ == "__main__":
    main()
