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

#pragma once

#include <optional>
#include <string>

#include <sl/Camera.hpp>

//
// Command line of the sample. Everything the options do is applied in main(): this file only
// turns argv into a struct, and prints it back.
//
struct Arguments {
    std::optional<sl::RESOLUTION> resolution = std::nullopt;
    std::optional<std::string> svoFile = std::nullopt;
    std::optional<std::string> streamIP = std::nullopt;
    std::optional<int> streamPort = std::nullopt;
    bool map = false;
    std::optional<std::string> inputAreaFile = std::nullopt;
    std::optional<std::string> outputAreaFile = std::nullopt;
    std::optional<std::string> roiFile = std::nullopt;
    bool customInitialPose = false;
    bool enable2dGroundMode = false;
    // TUM file holding the pose of every image. Giving --export-tum without a name writes out.tum
    // in the current directory, which is what a single interactive replay wants; a batch run that
    // replays the same recording several times passes a name instead.
    std::optional<std::string> trajectoryTUMFile = std::nullopt;
    // Run without the OpenGL viewer: no window, no point cloud retrieval, and the replay ends on the
    // end of the recording rather than on the window being closed. For a batch replay scored from
    // its output files, where nobody is watching and the display only costs GPU time.
    bool noGui = false;
    // TUM file holding the final pose of every keyframe of the map, written once on exit so the
    // poses are the ones the last loop closure left behind.
    std::optional<std::string> keyframeTUMFile = std::nullopt;
    bool voxelMode = false;
    // JSON file holding the motion measured by an external source (wheel odometry, an external
    // VIO, a GNSS-derived displacement), handed to the tracking with Camera::ingestMotionPrior().
    // Wins over the motion prior an SVO may carry in its data channel (see MotionPriorChannel), which
    // is otherwise used automatically.
    std::optional<std::string> motionPriorFile = std::nullopt;
    // Ignore the motion prior channel of the SVO, to compare a replay with and without it.
    bool noMotionPrior = false;
    // Hand every motion prior sample to the tracking this much later than it arrived, to see what a
    // late source does to the tracking without having to record one.
    double motionPriorDelayMs = 0.;
    // CSV of the per-image tracking status, so a replay can be scored for robustness
    // (tracking losses, relocalizations, and how the motion prior was used) instead of
    // only for trajectory error. The console only reports the prior status on change.
    std::optional<std::string> statusLogFile = std::nullopt;
};

// Fills args from argv. Reports the reason and returns false on an unknown option, a missing
// value or an inconsistent combination.
bool parseArgs(int argc, char* argv[], Arguments& args);

void printUsage(const std::string& programName);

// Echoes the options that are actually in effect, so a run can be told apart from another.
void printArgs(const Arguments& args);
