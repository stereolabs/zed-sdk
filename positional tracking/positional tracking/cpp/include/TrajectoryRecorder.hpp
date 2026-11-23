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

#include <cstdint>
#include <fstream>
#include <map>
#include <string>

#include <sl/Camera.hpp>

//
// Writes what the tracking produced to disk, so a run can be compared to another afterwards:
//
//   - the trajectory in TUM format ("timestamp tx ty tz qx qy qz qw", one line per image), the
//     input most trajectory-error tools expect;
//   - the tracking status of every image as a CSV, to score a replay for robustness (tracking
//     losses, relocalizations, how the motion prior was used) rather than only for accuracy;
//   - the keyframes of the map in the same TUM format, written in one go at the end of the run.
//
// All files are optional: a recorder with none of them open does nothing. Everything is flushed
// per image, so an interrupted replay still leaves a usable file.
//
class TrajectoryRecorder {
public:
    void openTrajectoryFile(const std::string& path);
    void openStatusFile(const std::string& path);

    void record(const sl::Pose& pose, const sl::PositionalTrackingStatus& trackingStatus);

    // Writes the keyframes of the map, ordered by timestamp. Called once at the end of the run
    // rather than while grabbing: a keyframe pose keeps being refined by loop closures until then.
    // Returns false if the file cannot be written.
    bool writeKeyframeFile(const std::string& path, const std::map<uint64_t, sl::KeyFrame>& keyframes);

    void close();

private:
    std::ofstream trajectoryFile;
    std::ofstream statusFile;
    uint64_t frameIndex = 0;
};
