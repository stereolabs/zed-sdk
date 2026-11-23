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

#include "TrajectoryRecorder.hpp"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <iomanip>

using namespace sl;

namespace {

    /**
     * @brief Writes one TUM line, without its line break.
     * @param stream Stream to write to.
     * @param timestamp Time of the pose.
     * @param pose The pose.
     */
    void writeTUMLine(std::ostream& stream, Timestamp timestamp, const Transform& pose) {
        Translation translation = pose.getTranslation();
        Orientation orientation = pose.getOrientation();

        stream << std::fixed << std::setprecision(9) << timestamp.getNanoseconds() / double(1e9) << " " << translation.tx << " "
               << translation.ty << " " << translation.tz << " " << orientation.ox << " " << orientation.oy << " " << orientation.oz << " "
               << orientation.ow;
    }

} // namespace

void TrajectoryRecorder::openTrajectoryFile(const std::string& path) {
    trajectoryFile.open(path);
}

void TrajectoryRecorder::openStatusFile(const std::string& path) {
    statusFile.open(path);
    statusFile << "frame,timestamp_ns,odometry_status,spatial_memory_status,tracking_fusion_status,pose_confidence\n";
}

void TrajectoryRecorder::record(const Pose& pose, const PositionalTrackingStatus& trackingStatus) {
    if (statusFile.is_open()) {
        statusFile << frameIndex << "," << pose.timestamp.getNanoseconds() << "," << toString(trackingStatus.odometry_status) << ","
                   << toString(trackingStatus.spatial_memory_status) << "," << toString(trackingStatus.tracking_fusion_status) << ","
                   << pose.pose_confidence << "\n";
        statusFile.flush();
    }

    record(pose);
}

void TrajectoryRecorder::record(const Pose& pose) {
    if (trajectoryFile.is_open()) {
        writeTUMLine(trajectoryFile, pose.timestamp, pose.pose_data);
        trajectoryFile << std::endl;
    }

    frameIndex++;
}

void TrajectoryRecorder::close() {
    if (trajectoryFile.is_open()) {
        trajectoryFile.close();
    }

    if (statusFile.is_open()) {
        statusFile.close();
    }
}

RigRecorder::RigRecorder(const std::string& rigTrajectoryPath, const std::string& statusPath, const std::vector<std::string>& cameraLabels)
    : cameraRecorders(cameraLabels.size()) {
    if (!rigTrajectoryPath.empty()) {
        rigRecorder.openTrajectoryFile(rigTrajectoryPath);
        for (size_t i = 0; i < cameraLabels.size(); i++) {
            cameraRecorders[i].openTrajectoryFile(cameraTrajectoryPath(rigTrajectoryPath, cameraLabels[i]));
        }
    }
    if (!statusPath.empty()) {
        rigRecorder.openStatusFile(statusPath);
    }
}

void RigRecorder::record(const Pose& rigPose, const PositionalTrackingStatus& trackingStatus, const std::vector<Pose>& cameraPoses) {
    assert(cameraPoses.size() == cameraRecorders.size());

    rigRecorder.record(rigPose, trackingStatus);
    for (size_t i = 0; i < cameraPoses.size(); i++) {
        cameraRecorders[i].record(cameraPoses[i]);
    }
}

void RigRecorder::close() {
    rigRecorder.close();
    for (TrajectoryRecorder& recorder : cameraRecorders) {
        recorder.close();
    }
}

bool writeKeyframeGroupFile(const std::string& path, const std::map<uint64_t, KeyFrameGroup>& groups) {
    std::ofstream keyframeFile(path);

    if (!keyframeFile.is_open()) {
        return false;
    }

    std::vector<const KeyFrameGroup*> ordered;
    ordered.reserve(groups.size());
    for (const auto& [id, group] : groups) {
        ordered.push_back(&group);
    }
    std::sort(ordered.begin(), ordered.end(), [](const KeyFrameGroup* left, const KeyFrameGroup* right) {
        return left->timestamp.getNanoseconds() < right->timestamp.getNanoseconds();
    });

    for (const KeyFrameGroup* group : ordered) {
        writeTUMLine(keyframeFile, group->timestamp, group->pose);
        keyframeFile << "\n";
    }

    return keyframeFile.good();
}

std::string cameraTrajectoryPath(const std::string& rigTrajectoryPath, const std::string& label) {
    assert(!rigTrajectoryPath.empty());

    std::filesystem::path path(rigTrajectoryPath);
    const std::string extension = path.has_extension() ? path.extension().string() : ".tum";
    path.replace_filename(path.stem().string() + "_" + label + extension);
    return path.string();
}
