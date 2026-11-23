///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2026, STEREOLABS.
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

#include <cassert>
#include <iostream>

#include "SampleConsole.hpp"

using namespace sl;

namespace {

    /**
     * @brief Reads the optional value of an option.
     * @param argc Number of arguments.
     * @param argv The arguments.
     * @param[in,out] i Index of the option, moved to its value when it has one.
     * @param defaultValue Value used when the next argument is missing or is another option.
     * @return The value of the option.
     */
    std::string readOptionalValue(int argc, char* argv[], int& i, const std::string& defaultValue) {
        assert(i > 0 && i < argc);
        if (i + 1 < argc && argv[i + 1][0] != '-') {
            return argv[++i];
        }
        return defaultValue;
    }

} // namespace

bool parseArgs(int argc, char* argv[], Arguments& args) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--map") {
            args.map = true;
        } else if (arg == "-i" && i + 1 < argc) {
            args.inputAreaFile = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            args.outputAreaFile = argv[++i];
        } else if (arg == "--export-tum") {
            args.trajectoryTUMFile = readOptionalValue(argc, argv, i, "out.tum");
        } else if (arg == "--no-gui") {
            args.noGui = true;
        } else if (arg == "--export-keyframes") {
            args.keyframeTUMFile = readOptionalValue(argc, argv, i, "keyframes.tum");
        } else if (arg == "--status-log" && i + 1 < argc) {
            args.statusLogFile = argv[++i];
        } else if (arg[0] != '-' && args.rigFile.empty()) {
            args.rigFile = arg;
        } else {
            print("Unrecognized or incomplete argument: " + arg, ERROR_CODE::FAILURE, false);
            return false;
        }
    }

    if (args.rigFile.empty()) {
        print("A rig file is required", ERROR_CODE::FAILURE, false);
        return false;
    }

    if (args.map && !args.outputAreaFile) {
        print("Option --map requires -o <output_area_file>", ERROR_CODE::FAILURE, false);
        return false;
    }

    return true;
}

void printUsage(const std::string& programName) {
    std::cout << "Usage: " << programName << " <rig.yaml> [options]\n\n"
              << "  <rig.yaml>                  The cameras of the rig, their inputs and where they sit on the rig.\n"
              << "                                        See rig.yaml next to this sample.\n\n"
              << "Options:\n"
              << "  --help                      Shows usage information.\n"
              << "  -i <input_area_file>        Optional. Input area file used in explore mode (default) or --map mode\n"
              << "  --map -o <output_area_file> Optional. Map mode creates or updates an .area file.\n"
              << "                                        Requires -o <output_area_filename> for generated map\n"
              << "  --export-tum [file]         Optional. Export the rig trajectory in TUM format (default: out.tum),\n"
              << "                                        and the trajectory of every camera next to it\n"
              << "                                        (out_<label>.tum)\n"
              << "  --no-gui                    Optional. Run without the viewer, ending on the end of the recordings\n"
              << "                                        instead of on the window being closed. For a batch replay\n"
              << "                                        scored from --export-tum / --status-log.\n"
              << "  --export-keyframes [file]   Optional. Export the final pose of the rig at every keyframe group of\n"
              << "                                        the map in TUM format, written on exit\n"
              << "                                        (default file: keyframes.tum)\n"
              << "  --status-log <file.csv>     Optional. Write the tracking status of every frame to a CSV\n"
              << "                                        (odometry / spatial memory / fusion status and the pose\n"
              << "                                        confidence).\n"
              << "\nExamples:\n"
              << "  " << programName << " rig.yaml\n"
              << "  " << programName << " rig.yaml --map -o new_map.area\n"
              << "  " << programName << " rig.yaml -i map.area --export-tum\n";
}

void printArgs(const Arguments& args) {
    print("Using rig file: " + args.rigFile);

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

    if (args.trajectoryTUMFile) {
        print("Enabled TUM trajectory export to " + args.trajectoryTUMFile.value());
    }

    if (args.noGui) {
        print("Running without the viewer");
    }

    if (args.keyframeTUMFile) {
        print("Enabled TUM keyframe export to " + args.keyframeTUMFile.value());
    }

    if (args.statusLogFile) {
        print("Writing the per-frame tracking status to " + args.statusLogFile.value());
    }

    std::cout << "\n";
}
