###########################################################################
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
###########################################################################

import pyzed.sl as sl

#
# Printing Utilities
#
RESET = "\033[0m"
RED   = "\033[31m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
YELLOW = "\033[33m"

def color_bool(val: bool) -> str:
    return f"{GREEN}true{RESET}" if bool(val) else f"{RED}false{RESET}"

def color_value(val) -> str:
    return f"{CYAN}{val}{RESET}"

def print_header(text: str):
    print()
    print("=" * 80)
    print(text.center(80))
    print("=" * 80)
    print()

def print_usage(program_name: str):
    print(f"""Usage: python3 {program_name} [options]

Options:
  --help                      Shows usage information.
  --resolution <mode>         Optional. Resolution options: (HD2K | HD1200 | HD1080 | HD720 | SVGA | VGA)
  --svo <filename.svo>        Optional. Use SVO file input. Mutually exclusive with --stream
  --stream <ip[:port]>        Optional. Use network streaming input. Mutually exclusive with --svo
  -i <input_area_file>        Optional. Input area file used in explore mode (default) or --map mode
  --map -o <output_area_file> Optional. Map mode creates or updates an .area file.
                              Requires -o <output_area_filename> for generated map
  --roi <roi_filepath>        Optional. Region of interest image mask to ignore a static area
  --custom-initial-pose       Optional. Use custom initial pose (see code comments for more detail)
  --2d-ground-mode            Optional. Enable 2D ground mode
  --export-tum                Optional. Export camera trajectory to out.tum file in TUM format

Examples:
  {program_name} --map -o new_map.area
  {program_name} --svo recording.svo2 -i map.area
""")
    
def sample_print(message: str, error_code = None, show_error_detail: bool = True):
    prefix = f"{CYAN}[Sample]{RESET}"

    if error_code is not None and error_code > sl.ERROR_CODE.SUCCESS:
        err_tag = f" {RED}[Error]{RESET} "
    elif error_code is not None and error_code < sl.ERROR_CODE.SUCCESS:
        err_tag = f" {YELLOW}[Warning]{RESET} "
    else:
        err_tag = " "

    line = f"{prefix}{err_tag}{message}"

    if error_code is not None and error_code != sl.ERROR_CODE.SUCCESS and show_error_detail:
        line += f" | {error_code.name}: {str(error_code)}"

    print(line)

def print_args(args):
    if args.resolution is not None:
        sample_print(f"Preferred Resolution: {args.resolution.name}")

    if args.svo_file is not None:
        sample_print(f"Using SVO file input: {args.svo_file}")

    if args.stream_ip is not None:
        suffix = f":{args.stream_port}" if args.stream_port is not None else ""
        sample_print(f"Using stream input {args.stream_ip}{suffix}")

    if args.map:
        sample_print("Positional tracking mode: Map")
    else:
        sample_print("Positional tracking mode: Explore (default)")

    if args.input_area_file is not None:
        sample_print(f"Using input area file: {args.input_area_file}")

    if args.output_area_file is not None:
        sample_print(f"Output file: {args.output_area_file}")
    
    if args.roi_file is not None:
        sample_print(f"Using ROI file: {args.roi_file}")

    if args.custom_initial_pose:
        sample_print("Enabled custom initial pose")

    if args.enable_2d_ground_mode:
        sample_print("Enabled 2D ground mode")

    if args.export_tum_file:
        sample_print("Enabled TUM trajectory export to out.tum")

    print()

def print_tracking_parameters(tracking_parameters: sl.PositionalTrackingParameters):
    print_header("Positional Tracking Parameters")
    sample_print("set_floor_as_origin   " + color_bool(tracking_parameters.set_floor_as_origin))
    sample_print("set_gravity_as_origin " + color_bool(tracking_parameters.set_gravity_as_origin))
    sample_print("enable_area_memory    " + color_bool(tracking_parameters.enable_area_memory))
    sample_print("enable_pose_smoothing " + color_bool(tracking_parameters.enable_pose_smoothing))
    sample_print("enable_imu_fusion     " + color_bool(tracking_parameters.enable_imu_fusion))
    sample_print("set_as_static         " + color_bool(tracking_parameters.set_as_static))
    sample_print("depth_min_range       " + color_value(tracking_parameters.depth_min_range))
    sample_print("enable_2d_ground_mode " + color_bool(tracking_parameters.enable_2d_ground_mode))
    sample_print("mode                  " + color_value(str(tracking_parameters.mode)))

    print_header("Interaction")
    sample_print("'space' to toggle camera view visibility")
    sample_print("'d' to switch background color from dark to light")
    sample_print("'p' to enable / disable current live point cloud display")
    sample_print("'l' to enable / disable landmark display")
    sample_print("'f' to follow the camera")
    sample_print("'z' to reset the view")
    sample_print("'ctrl' + drag to rotate")
    sample_print("'esc' to exit")
    print()

#
# Argument Parsing
#
def parse_args(argv, args) -> bool:
    i = 1
    while i < len(argv):
        arg = argv[i]

        if arg == "--resolution" and i + 1 < len(argv):
            resolution_str = argv[i + 1]

            resolution_map = {
                "HD2K": sl.RESOLUTION.HD2K,
                "HD1200": sl.RESOLUTION.HD1200,
                "HD1080": sl.RESOLUTION.HD1080,
                "HD720": sl.RESOLUTION.HD720,
                "SVGA": sl.RESOLUTION.SVGA,
                "VGA": sl.RESOLUTION.VGA,
            }

            if resolution_str in resolution_map:
                args.resolution = resolution_map[resolution_str]
            else:
                sample_print(f"Invalid resolution: {resolution_str}", sl.ERROR_CODE.FAILURE, False)
                return False
            
            i += 1

        elif arg == "--svo" and i + 1 < len(argv):
            if args.stream_ip:
                sample_print("Options --svo and --stream are mutually exclusive", sl.ERROR_CODE.FAILURE, False)
                return False
            
            if ".svo" not in argv[i + 1]:
                sample_print("Option --svo requires a filename that contains '.svo'", sl.ERROR_CODE.FAILURE, False)
                return False

            args.svo_file = argv[i + 1]
            i += 1

        elif arg == "--stream" and i + 1 < len(argv):
            if args.svo_file:
                sample_print("Options --stream and --svo are mutually exclusive", sl.ERROR_CODE.FAILURE, False)
                return False
            
            stream_arg = argv[i + 1]
            if ':' in stream_arg:
                ip, port = stream_arg.split(":", 1)
                args.stream_ip = ip

                try:
                    args.stream_port = int(port)
                except ValueError:
                    print(f"Invalid port in --stream: {port}", sl.ERROR_CODE.FAILURE, False)
                    return False
            else:
                args.stream_ip = stream_arg
            i += 1

        elif arg == "-i" and i + 1 < len(argv):
            args.input_area_file = argv[i + 1]
            i += 1

        elif arg == "-o" and i + 1 < len(argv):
            args.output_area_file = argv[i + 1]
            i += 1
        
        elif arg == "--roi" and i + 1 < len(argv):
            args.roi_file = argv[i + 1]
            i += 1

        elif arg == "--map":
            args.map = True

        elif arg == "--custom-initial-pose":
            args.custom_initial_pose = True
        
        elif arg == "--2d-ground-mode":
            args.enable_2d_ground_mode = True

        elif arg == "--export-tum":
            args.export_tum_file = True

        else:
            sample_print(f"Unrecognized or incomplete argument: {arg}", sl.ERROR_CODE.FAILURE, False)
            return False

        i += 1

    if args.map and args.output_area_file is None:
        sample_print("Option --map requires -o <output_area_file>", sl.ERROR_CODE.FAILURE, False)
        return False

    return True

#
# Display Utilities
#
def interpolate_color(color1, color2, ratio):
    ratio = max(0.0, min(1.0, ratio))

    return (
        color1[0] * (1 - ratio) + color2[0] * ratio,
        color1[1] * (1 - ratio) + color2[1] * ratio,
        color1[2] * (1 - ratio) + color2[2] * ratio,
        color1[3] * (1 - ratio) + color2[3] * ratio
    )
