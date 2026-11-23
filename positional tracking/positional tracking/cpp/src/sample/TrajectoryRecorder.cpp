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

#include "TrajectoryRecorder.hpp"

#include <algorithm>
#include <iomanip>
#include <vector>

using namespace sl;

void TrajectoryRecorder::openTrajectoryFile(const std::string& path) {
    trajectoryFile.open(path);
}

void TrajectoryRecorder::openStatusFile(const std::string& path) {
    statusFile.open(path);
    statusFile << "frame,timestamp_ns,odometry_status,spatial_memory_status,"
                  "tracking_fusion_status,relative_pose_fusion_status,pose_confidence\n";
}

void TrajectoryRecorder::record(const Pose& pose, const PositionalTrackingStatus& trackingStatus) {
    if (trajectoryFile.is_open()) {
        Translation translation = pose.getTranslation();
        Orientation orientation = pose.getOrientation();

        // std::endl rather than '\n': it flushes, so an interrupted replay still leaves a usable file.
        trajectoryFile << std::fixed << std::setprecision(9) << pose.timestamp.getNanoseconds() / double(1e9) << " " << translation.tx
                       << " " << translation.ty << " " << translation.tz << " " << orientation.ox << " " << orientation.oy << " "
                       << orientation.oz << " " << orientation.ow << std::endl;
    }

    if (statusFile.is_open()) {
        statusFile << frameIndex << "," << pose.timestamp.getNanoseconds() << "," << toString(trackingStatus.odometry_status) << ","
                   << toString(trackingStatus.spatial_memory_status) << "," << toString(trackingStatus.tracking_fusion_status) << ","
                   << toString(trackingStatus.relative_pose_fusion_status) << "," << pose.pose_confidence << "\n";
        statusFile.flush();
    }

    frameIndex++;
}

bool TrajectoryRecorder::writeKeyframeFile(const std::string& path, const std::map<uint64_t, KeyFrame>& keyframes) {
    std::ofstream keyframeFile(path);

    if (!keyframeFile.is_open()) {
        return false;
    }

    // The map is keyed by keyframe ID, while TUM readers expect the lines in chronological order.
    std::vector<const KeyFrame*> ordered;
    ordered.reserve(keyframes.size());
    for (const auto& [id, keyframe] : keyframes) {
        ordered.push_back(&keyframe);
    }
    std::sort(ordered.begin(), ordered.end(), [](const KeyFrame* left, const KeyFrame* right) {
        return left->timestamp.getNanoseconds() < right->timestamp.getNanoseconds();
    });

    keyframeFile << std::fixed << std::setprecision(9);
    for (const KeyFrame* keyframe : ordered) {
        Translation translation = keyframe->pose.getTranslation();
        Orientation orientation = keyframe->pose.getOrientation();

        keyframeFile << keyframe->timestamp.getNanoseconds() / double(1e9) << " " << translation.tx << " " << translation.ty << " "
                     << translation.tz << " " << orientation.ox << " " << orientation.oy << " " << orientation.oz << " " << orientation.ow
                     << "\n";
    }

    return keyframeFile.good();
}

void TrajectoryRecorder::close() {
    if (trajectoryFile.is_open()) {
        trajectoryFile.close();
    }

    if (statusFile.is_open()) {
        statusFile.close();
    }
}
