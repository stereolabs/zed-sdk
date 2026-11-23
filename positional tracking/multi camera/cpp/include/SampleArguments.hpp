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

#pragma once

#include <optional>
#include <string>

/**
 * @brief The options given on the command line.
 */
struct Arguments {
    /// YAML file describing the rig: its cameras, their inputs and their poses on the rig.
    std::string rigFile;
    /// Save the map to outputAreaFile on exit.
    bool map = false;
    /// Area file to load when the tracking starts.
    std::optional<std::string> inputAreaFile = std::nullopt;
    /// Area file the map is saved to in map mode.
    std::optional<std::string> outputAreaFile = std::nullopt;
    /// TUM file of the rig trajectory. Each camera trajectory goes next to it, suffixed with the camera label.
    std::optional<std::string> trajectoryTUMFile = std::nullopt;
    /// Run without the viewer, until the end of the recordings.
    bool noGui = false;
    /// TUM file of the rig pose at every keyframe group, written on exit.
    std::optional<std::string> keyframeTUMFile = std::nullopt;
    /// CSV file of the tracking status at every frame.
    std::optional<std::string> statusLogFile = std::nullopt;
};

/**
 * @brief Parses the command line.
 * @param argc Number of arguments.
 * @param argv The arguments, program name first.
 * @param[out] args The options found.
 * @return False, after printing why, on an unrecognized option, a missing value or an inconsistent combination.
 */
bool parseArgs(int argc, char* argv[], Arguments& args);

/**
 * @brief Prints the usage of the sample.
 * @param programName Name the program was started with.
 */
void printUsage(const std::string& programName);

/**
 * @brief Prints the options in use.
 * @param args The parsed options.
 */
void printArgs(const Arguments& args);
