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

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <sl/Sensors.hpp>

/**
 * @brief Writes one trajectory in TUM format ("timestamp tx ty tz qx qy qz qw", one line per frame) and, optionally, the
 * tracking status of every frame as a CSV. A file that was not opened is skipped. Lines are flushed as they are written.
 */
class TrajectoryRecorder {
public:
    /**
     * @brief Opens the TUM trajectory file.
     * @param path Path of the file, overwritten.
     */
    void openTrajectoryFile(const std::string& path);

    /**
     * @brief Opens the CSV status file and writes its header.
     * @param path Path of the file, overwritten.
     */
    void openStatusFile(const std::string& path);

    /**
     * @brief Writes the pose of one frame and its tracking status.
     * @param pose The pose.
     * @param trackingStatus The tracking status of the same frame.
     */
    void record(const sl::Pose& pose, const sl::PositionalTrackingStatus& trackingStatus);

    /**
     * @brief Writes the pose of one frame.
     * @param pose The pose.
     */
    void record(const sl::Pose& pose);

    /**
     * @brief Closes the files.
     */
    void close();

private:
    std::ofstream trajectoryFile;
    std::ofstream statusFile;
    uint64_t frameIndex = 0;
};

/**
 * @brief Writes the trajectory of the rig, the trajectory of each of its cameras and the tracking status, frame by frame.
 */
class RigRecorder {
public:
    /**
     * @brief Opens the files asked for.
     * @param rigTrajectoryPath TUM file of the rig, empty for none. Each camera gets cameraTrajectoryPath() next to it.
     * @param statusPath CSV status file, empty for none.
     * @param cameraLabels Label of each camera of the rig, in rig order.
     */
    RigRecorder(const std::string& rigTrajectoryPath, const std::string& statusPath, const std::vector<std::string>& cameraLabels);

    /**
     * @brief Writes one frame.
     * @param rigPose Pose of the rig.
     * @param trackingStatus Tracking status of the rig.
     * @param cameraPoses Pose of each camera, in rig order.
     */
    void record(const sl::Pose& rigPose, const sl::PositionalTrackingStatus& trackingStatus, const std::vector<sl::Pose>& cameraPoses);

    /**
     * @brief Closes the files.
     */
    void close();

private:
    TrajectoryRecorder rigRecorder;
    std::vector<TrajectoryRecorder> cameraRecorders;
};

/**
 * @brief Writes the pose of the rig at every keyframe group, in TUM format, ordered by timestamp.
 * @param path Path of the file, overwritten.
 * @param groups The keyframe groups of the map.
 * @return False if the file cannot be written.
 */
bool writeKeyframeGroupFile(const std::string& path, const std::map<uint64_t, sl::KeyFrameGroup>& groups);

/**
 * @brief Names the trajectory file of a camera after the one of the rig.
 * @param rigTrajectoryPath Trajectory file of the rig, such as out.tum.
 * @param label Label of the camera, such as front.
 * @return The trajectory file of the camera, such as out_front.tum.
 */
std::string cameraTrajectoryPath(const std::string& rigTrajectoryPath, const std::string& label);
