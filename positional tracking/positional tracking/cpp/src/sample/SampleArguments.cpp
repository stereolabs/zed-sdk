///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2025, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

#include "SampleArguments.hpp"

#include <iostream>

#include "SampleConsole.hpp"

using namespace sl;

bool parseArgs(int argc, char* argv[], Arguments& args) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--resolution" && i + 1 < argc) {
            RESOLUTION resolution;
            bool error = fromString(argv[++i], resolution);

            if (error) {
                print("Invalid resolution: " + std::string(argv[i]), ERROR_CODE::FAILURE, false);
                return false;
            }
            args.resolution = resolution;
        } else if (arg == "--svo" && i + 1 < argc) {
            if (args.streamIP) {
                print("Options --svo and --stream are mutually exclusive", ERROR_CODE::FAILURE, false);
                return false;
            }

            std::string filename = argv[++i];
            if (filename.find(".svo") == std::string::npos) {
                print("Option --svo requires a filename that contains '.svo'", ERROR_CODE::FAILURE, false);
                return false;
            }

            args.svoFile = filename;
        } else if (arg == "--stream" && i + 1 < argc) {
            if (args.svoFile) {
                print("Options --stream and --svo are mutually exclusive", ERROR_CODE::FAILURE, false);
                return false;
            }

            std::string ip;
            std::optional<int> port;

            const std::string& address = argv[++i];
            auto colonPos = address.find(':');

            if (colonPos != std::string::npos) {
                ip = address.substr(0, colonPos);
                std::string portStr = address.substr(colonPos + 1);

                try {
                    port = std::stoi(portStr);
                } catch (const std::exception& e) {
                    print("Invalid port in --stream: " + portStr, ERROR_CODE::FAILURE, false);
                    return false;
                }
            } else {
                ip = address;
            }

            args.streamIP = ip;
            args.streamPort = port;
        } else if (arg == "--map") {
            args.map = true;
        } else if (arg == "-i" && i + 1 < argc) {
            args.inputAreaFile = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            args.outputAreaFile = argv[++i];
        } else if (arg == "--roi" && i + 1 < argc) {
            args.roiFile = argv[++i];
        } else if (arg == "--custom-initial-pose") {
            args.customInitialPose = true;
        } else if (arg == "--2d-ground-mode") {
            args.enable2dGroundMode = true;
        } else if (arg == "--export-tum") {
            // The filename is optional, so an interactive replay can stay as short as --export-tum.
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.trajectoryTUMFile = argv[++i];
            } else {
                args.trajectoryTUMFile = "out.tum";
            }
        } else if (arg == "--no-gui") {
            args.noGui = true;
        } else if (arg == "--export-keyframes") {
            // The filename is optional, like --export-tum's.
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.keyframeTUMFile = argv[++i];
            } else {
                args.keyframeTUMFile = "keyframes.tum";
            }
        } else if (arg == "--voxel") {
            args.voxelMode = true;
        } else if (arg == "--motion-prior" && i + 1 < argc) {
            args.motionPriorFile = argv[++i];
        } else if (arg == "--no-motion-prior") {
            args.noMotionPrior = true;
        } else if (arg == "--motion-prior-delay" && i + 1 < argc) {
            const std::string value = argv[++i];
            try {
                args.motionPriorDelayMs = std::stod(value);
            } catch (const std::exception&) {
                print("Invalid delay in --motion-prior-delay: " + value, ERROR_CODE::FAILURE, false);
                return false;
            }
            if (args.motionPriorDelayMs < 0.) {
                print("Option --motion-prior-delay takes a delay of 0 ms or more", ERROR_CODE::FAILURE, false);
                return false;
            }
        } else if (arg == "--status-log" && i + 1 < argc) {
            args.statusLogFile = argv[++i];
        } else {
            print("Unrecognized or incomplete argument: " + arg, ERROR_CODE::FAILURE, false);
            return false;
        }
    }

    if (args.noMotionPrior && args.motionPriorFile) {
        print("Options --no-motion-prior and --motion-prior are mutually exclusive", ERROR_CODE::FAILURE, false);
        return false;
    }

    if (args.map && !args.outputAreaFile) {
        print("Option --map requires -o <output_area_file>", ERROR_CODE::FAILURE, false);
        return false;
    }

    return true;
}

void printUsage(const std::string& programName) {
    std::cout << "Usage: " << programName << " [options]\n\n"
              << "Options:\n"
              << "  --help                      Shows usage information.\n"
              << "  --resolution <mode>         Optional. Resolution options: (HD2K | HD1200 | HD1080 | HD720 | SVGA | VGA)\n"
              << "  --svo <filename.svo>        Optional. Use SVO file input. Mutually exclusive with --stream\n"
              << "  --stream <ip[:port]>        Optional. Use network streaming input. Mutually exclusive with --svo\n"
              << "  -i <input_area_file>        Optional. Input area file used in explore mode (default) or --map mode\n"
              << "  --map -o <output_area_file> Optional. Map mode creates or updates an .area file.\n"
              << "                                        Requires -o <output_area_filename> for generated map\n"
              << "  --roi <roi_filepath>        Optional. Region of interest image mask to ignore a static area\n"
              << "  --custom-initial-pose       Optional. Use custom initial pose (see code comments for more detail)\n"
              << "  --2d-ground-mode            Optional. Enable 2D ground mode\n"
              << "  --export-tum [file]         Optional. Export camera trajectory in TUM format (default: out.tum)\n"
              << "  --no-gui                    Optional. Run without the viewer, ending on the end of the recording\n"
              << "                                        instead of on the window being closed. For a batch replay\n"
              << "                                        scored from --export-tum / --status-log.\n"
              << "  --export-keyframes [file]   Optional. Export the final pose of every keyframe of the map in TUM\n"
              << "                                        format, written on exit (default file: keyframes.tum)\n"
              << "  --status-log <file.csv>     Optional. Write the tracking status of every image to a CSV\n"
              << "                                        (odometry / spatial memory / fusion / motion prior\n"
              << "                                        status and the pose confidence).\n"
              << "  --voxel                     Optional. Use voxel-decimated point cloud (retrieveVoxelMeasure)\n"
              << "                                        instead of the organized grid (retrieveMeasure).\n"
              << "  --motion-prior <file.json>  Optional. Enable the motion prior and replay the motion measured by\n"
              << "                                        an external source (wheel odometry, external VIO, GNSS)\n"
              << "                                        from a JSON file. See the file format in the sample README.\n"
              << "                                        Requires the GEN_3 positional tracking mode.\n"
              << "                                        An SVO that carries a motion prior uses it on its own;\n"
              << "                                        this file wins over it.\n"
              << "  --no-motion-prior           Optional. Ignore the motion prior recorded in the SVO.\n"
              << "  --motion-prior-delay <ms>   Optional. Hand every motion prior sample to the tracking this\n"
              << "                                        much later than it arrived, to test a late source.\n"
              << "\nExamples:\n"
              << "  " << programName << " --map -o new_map.area\n"
              << "  " << programName << " --svo recording.svo2 -i map.area\n"
              << "  " << programName << " --svo recording.svo2 --motion-prior odometry.json\n"
              << "  " << programName << " --svo recording_with_prior.svo2 --motion-prior-delay 250\n";
}

void printArgs(const Arguments& args) {
    if (args.resolution) {
        // toString() is the SDK's own mapping: a hand-written switch here silently printed nothing
        // for any resolution added to the enum after it was written.
        print("Preferred Resolution: " + std::string(toString(args.resolution.value()).c_str()));
    }

    if (args.svoFile) {
        print("Using SVO file input: " + args.svoFile.value());
    }

    if (args.streamIP) {
        std::string suffix = args.streamPort ? ":" + std::to_string(args.streamPort.value()) : "";
        print("Using stream input " + args.streamIP.value() + suffix);
    }

    if (args.map) {
        print("Positional tracking mode: Map");
    } else {
        print("Positional tracking mode: Explore (default)");
    }

    if (args.inputAreaFile) {
        print("Using input area file: " + args.inputAreaFile.value());
    }

    if (args.outputAreaFile) {
        print("Output file: " + args.outputAreaFile.value());
    }

    if (args.roiFile) {
        print("Using ROI file: " + args.roiFile.value());
    }

    if (args.customInitialPose) {
        print("Enabled custom initial pose");
    }

    if (args.enable2dGroundMode) {
        print("Enabled 2D ground mode");
    }

    if (args.trajectoryTUMFile) {
        print("Enabled TUM trajectory export to " + args.trajectoryTUMFile.value());
    }

    if (args.noGui) {
        print("Running without the viewer");
    }

    if (args.keyframeTUMFile) {
        print("Enabled TUM keyframe export to " + args.keyframeTUMFile.value());
    }

    if (args.voxelMode) {
        print("Point cloud mode: voxel-decimated (retrieveVoxelMeasure)");
    }

    if (args.motionPriorFile) {
        print("Using motion prior file: " + args.motionPriorFile.value());
    }

    if (args.noMotionPrior) {
        print("Ignoring the motion prior recorded in the SVO");
    }

    if (args.motionPriorDelayMs > 0.) {
        print("Delaying every motion prior sample by " + std::to_string(args.motionPriorDelayMs) + " ms");
    }

    if (args.statusLogFile) {
        print("Writing the per-image tracking status to " + args.statusLogFile.value());
    }

    std::cout << "\n";
}
