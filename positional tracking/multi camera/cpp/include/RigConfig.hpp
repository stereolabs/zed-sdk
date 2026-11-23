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
#include <vector>

#include <sl/Camera.hpp>

/**
 * @brief One camera of the rig, as described in the rig file.
 */
struct RigCamera {
    /// Name used on the console, in the viewer and in the per-camera output files.
    std::string label;

    /// Where the images come from: a live camera, an SVO recording or a network stream, of a ZED stereo camera or a ZED One.
    sl::InputType input;

    /// The input in words, for the console.
    std::string inputDescription;

    /// Pose of the camera on the rig: takes a point from the camera frame into the rig frame (BASELINK).
    /// Meters, camera axes (X right, Y down, Z forward).
    sl::Transform cameraToRig;
};

/**
 * @brief The rig described in the rig file: its cameras and their capture settings.
 */
struct RigConfig {
    /// Capture resolution of every live camera.
    std::optional<sl::RESOLUTION> resolution = std::nullopt;

    /// Capture frame rate of every live camera, 0 for the default of the resolution.
    int fps = 0;

    /// Label of the camera that drives the tracking, a ZED stereo camera. Empty for the first ZED stereo camera.
    std::string trackingCamera;

    /// The cameras, in the order of the rig file.
    std::vector<RigCamera> cameras;

    /**
     * @brief Tells whether every camera replays an SVO recording.
     * @return True when all inputs are SVO files.
     */
    bool allFromSVO() const;

    /**
     * @brief Finds the camera named to drive the tracking.
     * @return Its index in cameras: the camera labeled trackingCamera, or nothing when trackingCamera is empty.
     */
    std::optional<size_t> trackingCameraIndex() const;

    /**
     * @brief Lists the camera labels.
     * @return The label of each camera, in rig order.
     */
    std::vector<std::string> labels() const;
};

/**
 * @brief Reads a rig file.
 * @param path Path of the YAML rig file.
 * @param[out] rig The rig it describes.
 * @return False, after printing why, when the file cannot be read or describes an invalid rig.
 */
bool loadRigConfig(const std::string& path, RigConfig& rig);
