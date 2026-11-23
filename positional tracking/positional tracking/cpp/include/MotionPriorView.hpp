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

#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <sl/Camera.hpp>

#include "Simple3DObject.hpp"

//
// What the motion prior did, drawn over the viewer:
//
//   - in the 3D view, a short arrow per sample, from the pose where its interval starts, and a straight line
//     per keyframe link (Camera::getPositionalTrackingMotionPriorLinks()), from its from keyframe to its to
//     keyframe. Clicking a line shows that link next to what the samples it spans recorded;
//   - a panel on the right: where the samples come from, how each was used, how late they arrived;
//   - a timeline strip along the bottom: the last 2 s of images, samples and keyframes.
//
// Colors say what became of a sample or a link:
//   teal    used for the pose of the image it ends at
//   orange  used by the map only (it arrived too late for its image)
//   red     rejected: it disagreed with what the camera sees
//   gray    unused, or not processed yet
//
class MotionPriorView {
public:
    // sensorTransform turns a sample, measured along the axes of the source, into the motion of the camera.
    void enable(const std::string& source, const sl::Transform& sensorTransform);
    bool isEnabled() const {
        return _enabled;
    }

    // Forget everything drawn so far: the replay went back in time.
    void clear();

    // arrivalTimestamp: when the sample reached the application.
    void addSample(const sl::RelativePoseData& sample, uint64_t arrivalTimestamp);
    void addImage(uint64_t imageTimestamp, const sl::Transform& pose, sl::RELATIVE_POSE_FUSION_STATUS status);
    void updateKeyframes(const std::map<uint64_t, sl::KeyFrame>& keyframes);
    void updateLinks(const std::map<uint64_t, sl::MotionPriorLink>& links);

    void toggleScene() {
        _sceneVisible = !_sceneVisible;
    }
    bool isSceneVisible() const {
        return _sceneVisible;
    }
    void toggleTimeline() {
        _timelineVisible = !_timelineVisible;
    }

    // Window pixels taken at the bottom by the timeline, for the rest of the overlay to stay above it.
    int reservedBottomHeight() const;

    // The 3D part. Called with the per-vertex color shader bound and its MVP set to viewProjection,
    // which is also what a click is matched against.
    void drawScene(const sl::Transform& viewProjection);

    // The 2D part, in window pixels.
    void drawOverlay(int width, int height, bool darkMode);

    // A click at window position (x, y), y from the top as GLUT gives it. Selects the link closest to it,
    // or clears the selection. Returns whether the click was taken.
    bool pick(int x, int y, int width, int height);

private:
    enum class Fate {
        PENDING,
        IMAGE,
        MAP,
        REJECTED,
        UNUSED
    };

    struct SampleRecord {
        uint64_t start = 0;
        uint64_t end = 0;
        uint64_t arrival = 0;
        sl::Transform cameraMotion; // the sample, as the motion of the camera
    };

    bool _enabled = false;
    bool _sceneVisible = true;
    bool _timelineVisible = true;
    std::string _source;
    sl::Transform _sensorTransform;

    std::vector<SampleRecord> _samples;
    std::map<uint64_t, sl::Transform> _poses;                           // by image timestamp
    std::map<uint64_t, sl::RELATIVE_POSE_FUSION_STATUS> _imageStatuses; // by image timestamp
    std::map<uint64_t, sl::KeyFrame> _keyframes;
    std::map<uint64_t, sl::MotionPriorLink> _links;
    std::optional<uint64_t> _selectedLink; // its to_keyframe_id

    // What became of each sample, refreshed a few times a second: the statuses change on every image.
    std::vector<Fate> _fates;
    bool _fatesDirty = true;
    std::chrono::steady_clock::time_point _lastFatesUpdate;
    size_t _fateCounts[5] = {0, 0, 0, 0, 0};
    double _delayMedianMs = 0.;
    double _delayP95Ms = 0.;
    // [start, end] of the accepted links, sorted, for a sample to be looked up under.
    std::vector<std::pair<uint64_t, uint64_t>> _acceptedLinkIntervals;

    // Points along each drawn link, in world coordinates, for the click to be matched against.
    std::map<uint64_t, std::vector<sl::float3>> _linkSegments;
    sl::Transform _viewProjection;

    Simple3DObject _arrowLines;
    Simple3DObject _linkLines;
    Simple3DObject _selectedLinkLines;
    Simple3DObject _keyframePoints;
    bool _sceneDirty = true;
    std::chrono::steady_clock::time_point _lastSceneBuild;

    Fate fateOf(const SampleRecord& sample) const;
    void updateFates(bool force = false);
    sl::float4 fateColor(Fate fate) const;
    bool isUsedByImages(const sl::MotionPriorLink& link) const;
    sl::float4 linkColor(const sl::MotionPriorLink& link) const;
    bool isLinkDashed(const sl::MotionPriorLink& link) const;

    void rebuildScene();
    void drawSelectedLinkEnds(int width, int height, bool darkMode);
    void drawPanel(int width, int height, bool darkMode);
    void drawLinkDetail(float x, float y, float width, bool darkMode);
    void drawTimeline(int width, bool darkMode);
};
