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

#include "MotionPriorView.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "Overlay2D.hpp"

namespace {

    const float to_f = 1.f / 255.f;
    const sl::float4 colorImage(45 * to_f, 212 * to_f, 191 * to_f, 1.f);     // teal
    const sl::float4 colorMap(251 * to_f, 146 * to_f, 60 * to_f, 1.f);       // orange
    const sl::float4 colorRejected(248 * to_f, 113 * to_f, 113 * to_f, 1.f); // red
    const sl::float4 colorUnused(139 * to_f, 149 * to_f, 163 * to_f, 1.f);   // gray
    const sl::float4 colorKeyframe(230 * to_f, 232 * to_f, 235 * to_f, 1.f);
    const sl::float4 colorTextDark(230 * to_f, 232 * to_f, 235 * to_f, 1.f);
    const sl::float4 colorTextLight(21 * to_f, 24 * to_f, 29 * to_f, 1.f);
    const sl::float4 colorMutedDark(150 * to_f, 158 * to_f, 170 * to_f, 1.f);
    const sl::float4 colorMutedLight(89 * to_f, 97 * to_f, 110 * to_f, 1.f);
    const sl::float4 colorPanelDark(14 * to_f, 16 * to_f, 20 * to_f, 0.85f);
    const sl::float4 colorPanelLight(250 * to_f, 250 * to_f, 252 * to_f, 0.9f);

    void* const fontSmall = GLUT_BITMAP_HELVETICA_12;
    void* const fontBody = GLUT_BITMAP_HELVETICA_18;

    const uint64_t timelineSpan = 2000000000ULL; // ns
    const int timelineHeight = 130;
    const int panelWidth = 380;
    const int linkSegments = 24;

    // M_PI is not standard C++: MSVC only defines it with _USE_MATH_DEFINES.
    const float radToDeg = 180.f / 3.14159265f;

    // The sample uses COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP: arrowhead barbs are laid out around it.
    const sl::float3 worldUp(0.f, 1.f, 0.f);

    sl::float3 cross(const sl::float3& a, const sl::float3& b) {
        return sl::float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }

    sl::float3 normalized(sl::float3 v, const sl::float3& fallback) {
        const float n = v.norm();
        return n > 1e-6f ? v / n : fallback;
    }

    // "0.98 m · 2.1°": how far and how much the camera turned.
    std::string describeMotion(const sl::Transform& motion) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << motion.getTranslation().norm() << " m, " << std::setprecision(1)
            << motion.getRotationVector().norm() * radToDeg << " deg";
        return out.str();
    }

    // Window pixels of a world point, y from the bottom. False behind the camera.
    bool project(const sl::Transform& viewProjection, const sl::float3& point, int width, int height, float& x, float& y) {
        const float* m = viewProjection.m;
        const float cx = m[0] * point.x + m[1] * point.y + m[2] * point.z + m[3];
        const float cy = m[4] * point.x + m[5] * point.y + m[6] * point.z + m[7];
        const float cw = m[12] * point.x + m[13] * point.y + m[14] * point.z + m[15];
        if (cw <= 1e-6f) {
            return false;
        }
        x = (cx / cw + 1.f) * 0.5f * width;
        y = (cy / cw + 1.f) * 0.5f * height;
        return true;
    }

    float distanceToSegment(float px, float py, float ax, float ay, float bx, float by) {
        const float dx = bx - ax, dy = by - ay;
        const float lengthSquared = dx * dx + dy * dy;
        float t = lengthSquared > 0.f ? ((px - ax) * dx + (py - ay) * dy) / lengthSquared : 0.f;
        t = std::max(0.f, std::min(1.f, t));
        const float ex = ax + t * dx - px, ey = ay + t * dy - py;
        return std::sqrt(ex * ex + ey * ey);
    }

    double percentile(std::vector<double> values, double fraction) {
        if (values.empty()) {
            return 0.;
        }
        std::sort(values.begin(), values.end());
        return values[std::min(values.size() - 1, static_cast<size_t>(fraction * values.size()))];
    }

} // namespace

void MotionPriorView::enable(const std::string& source, const sl::Transform& sensorTransform) {
    _enabled = true;
    _source = source;
    _sensorTransform = sensorTransform;

    _arrowLines.setDrawingType(GL_LINES);
    _linkLines.setDrawingType(GL_LINES);
    _selectedLinkLines.setDrawingType(GL_LINES);
    _keyframePoints.setDrawingType(GL_POINTS);
}

void MotionPriorView::clear() {
    _samples.clear();
    _poses.clear();
    _imageStatuses.clear();
    _links.clear();
    _linkSegments.clear();
    _fates.clear();
    _acceptedLinkIntervals.clear();
    _selectedLink.reset();
    _sceneDirty = true;
    _fatesDirty = true;
}

void MotionPriorView::addSample(const sl::RelativePoseData& sample, uint64_t arrivalTimestamp) {
    SampleRecord record;
    record.start = sample.start_timestamp.getNanoseconds();
    record.end = sample.end_timestamp.getNanoseconds();
    record.arrival = arrivalTimestamp;

    // What the tracking does with it too: moved from the axes of the source to those of the camera.
    sl::Transform motion = sample.motion;
    record.cameraMotion = _sensorTransform * motion * sl::Transform::inverse(_sensorTransform);

    _samples.push_back(record);
    _fates.push_back(Fate::PENDING);
    _sceneDirty = true;
    _fatesDirty = true;
}

void MotionPriorView::addImage(uint64_t imageTimestamp, const sl::Transform& pose, sl::RELATIVE_POSE_FUSION_STATUS status) {
    _poses[imageTimestamp] = pose;
    _imageStatuses[imageTimestamp] = status;
    _sceneDirty = true;
    _fatesDirty = true;
}

void MotionPriorView::updateKeyframes(const std::map<uint64_t, sl::KeyFrame>& keyframes) {
    _keyframes = keyframes;
    _sceneDirty = true;
}

void MotionPriorView::updateLinks(const std::map<uint64_t, sl::MotionPriorLink>& links) {
    _links = links;
    if (_selectedLink && !_links.count(*_selectedLink)) {
        _selectedLink.reset();
    }

    _acceptedLinkIntervals.clear();
    for (const auto& entry : _links) {
        const sl::MotionPriorLink& link = entry.second;
        if (link.status == sl::RELATIVE_POSE_FUSION_STATUS::OK) {
            _acceptedLinkIntervals.emplace_back(link.start_timestamp.getNanoseconds(), link.end_timestamp.getNanoseconds());
        }
    }
    std::sort(_acceptedLinkIntervals.begin(), _acceptedLinkIntervals.end());

    _sceneDirty = true;
    _fatesDirty = true;
}

int MotionPriorView::reservedBottomHeight() const {
    return _enabled && _timelineVisible ? timelineHeight + 20 : 0;
}

//
// What became of a sample, from what the tracking reported for the image the sample ends at, and from
// the links: a sample its image did not use, under an accepted link, went to the map.
//
MotionPriorView::Fate MotionPriorView::fateOf(const SampleRecord& sample) const {
    auto image = _imageStatuses.lower_bound(sample.end);
    if (image == _imageStatuses.end()) {
        return Fate::PENDING;
    }

    if (image->second == sl::RELATIVE_POSE_FUSION_STATUS::OK) {
        return Fate::IMAGE;
    }

    // The last accepted link starting at or before the sample: links never overlap, so it is the only
    // one that can hold it.
    auto link = std::upper_bound(_acceptedLinkIntervals.begin(), _acceptedLinkIntervals.end(), std::make_pair(sample.start, UINT64_MAX));
    if (link != _acceptedLinkIntervals.begin() && sample.end <= std::prev(link)->second) {
        return Fate::MAP;
    }

    return image->second == sl::RELATIVE_POSE_FUSION_STATUS::REJECTED ? Fate::REJECTED : Fate::UNUSED;
}

void MotionPriorView::updateFates(bool force) {
    const auto now = std::chrono::steady_clock::now();
    if (!force && (!_fatesDirty || now - _lastFatesUpdate < std::chrono::milliseconds(200))) {
        return;
    }

    _fates.resize(_samples.size());
    std::fill(std::begin(_fateCounts), std::end(_fateCounts), 0);
    std::vector<double> delays;
    delays.reserve(_samples.size());

    for (size_t i = 0; i < _samples.size(); i++) {
        _fates[i] = fateOf(_samples[i]);
        _fateCounts[static_cast<int>(_fates[i])]++;
        delays.push_back((static_cast<double>(_samples[i].arrival) - static_cast<double>(_samples[i].end)) / 1e6);
    }

    _delayMedianMs = percentile(delays, 0.5);
    _delayP95Ms = percentile(delays, 0.95);
    _fatesDirty = false;
    _lastFatesUpdate = now;
}

sl::float4 MotionPriorView::fateColor(Fate fate) const {
    switch (fate) {
        case Fate::IMAGE:
            return colorImage;
        case Fate::MAP:
            return colorMap;
        case Fate::REJECTED:
            return colorRejected;
        default:
            return colorUnused;
    }
}

// The images used the motion of a link when every image it spans took its motion prior; otherwise only
// the map did.
bool MotionPriorView::isUsedByImages(const sl::MotionPriorLink& link) const {
    auto image = _imageStatuses.upper_bound(link.start_timestamp.getNanoseconds());
    const auto end = _imageStatuses.upper_bound(link.end_timestamp.getNanoseconds());
    if (image == end) {
        return false;
    }
    for (; image != end; ++image) {
        if (image->second != sl::RELATIVE_POSE_FUSION_STATUS::OK) {
            return false;
        }
    }
    return true;
}

sl::float4 MotionPriorView::linkColor(const sl::MotionPriorLink& link) const {
    if (link.status == sl::RELATIVE_POSE_FUSION_STATUS::OK) {
        return isUsedByImages(link) ? colorImage : colorMap;
    }
    return link.status == sl::RELATIVE_POSE_FUSION_STATUS::REJECTED ? colorRejected : colorUnused;
}

bool MotionPriorView::isLinkDashed(const sl::MotionPriorLink& link) const {
    return !(link.status == sl::RELATIVE_POSE_FUSION_STATUS::OK && isUsedByImages(link))
        && link.status != sl::RELATIVE_POSE_FUSION_STATUS::REJECTED;
}

//
// 3D
//
void MotionPriorView::rebuildScene() {
    _arrowLines.clear();
    _linkLines.clear();
    _selectedLinkLines.clear();
    _keyframePoints.clear();
    _linkSegments.clear();

    // A short arrow per sample: from the pose of the image its interval starts at, to where the sample says
    // the camera went.
    updateFates(true);
    for (size_t i = 0; i < _samples.size(); i++) {
        const SampleRecord& sample = _samples[i];
        auto startPose = _poses.lower_bound(sample.start);
        if (startPose == _poses.end()) {
            continue;
        }

        const sl::Transform& pose = startPose->second;
        const sl::float3 from = pose.getTranslation();
        const sl::float3 to = sl::Transform(pose * sample.cameraMotion).getTranslation();
        const sl::float4 color = fateColor(_fates[i]);

        _arrowLines.addPoint(from, color);
        _arrowLines.addPoint(to, color);

        sl::float3 direction = to - from;
        const float length = direction.norm();
        if (length > 1e-4f) {
            const sl::float3 back = direction / length * (-0.3f * length);
            const sl::float3 side = normalized(cross(direction, worldUp), sl::float3(1.f, 0.f, 0.f)) * (0.15f * length);
            _arrowLines.addPoint(to, color);
            _arrowLines.addPoint(to + back + side, color);
            _arrowLines.addPoint(to, color);
            _arrowLines.addPoint(to + back - side, color);
        }
    }

    // A straight line per link, from the pose of its from keyframe to the pose of its to keyframe, with an
    // arrowhead on the to end. Split in segments for a dashed link to show gaps.
    for (const auto& entry : _links) {
        const sl::MotionPriorLink& link = entry.second;
        auto from = _keyframes.find(link.from_keyframe_id);
        auto to = _keyframes.find(link.to_keyframe_id);
        if (from == _keyframes.end() || to == _keyframes.end()) {
            continue;
        }

        const sl::float3 a = from->second.pose.getTranslation();
        const sl::float3 b = to->second.pose.getTranslation();

        std::vector<sl::float3>& points = _linkSegments[entry.first];
        for (int i = 0; i <= linkSegments; i++) {
            const float t = static_cast<float>(i) / linkSegments;
            points.push_back(a * (1.f - t) + b * t);
        }

        const bool selected = _selectedLink && *_selectedLink == entry.first;
        Simple3DObject& lines = selected ? _selectedLinkLines : _linkLines;
        const sl::float4 color = linkColor(link);
        const bool dashed = isLinkDashed(link);

        for (int i = 0; i < linkSegments; i++) {
            if (dashed && i % 2 == 1) {
                continue;
            }
            lines.addPoint(points[i], color);
            lines.addPoint(points[i + 1], color);
        }

        // Four barbs around the line, so the head reads from any viewpoint, dashed or not.
        sl::float3 chord = b - a;
        const float chordLength = chord.norm();
        if (chordLength > 1e-6f) {
            const sl::float3 forward = chord / chordLength;
            const float size = std::max(0.03f, std::min(0.4f, 0.15f * chordLength));
            const sl::float3 side = normalized(cross(forward, worldUp), sl::float3(1.f, 0.f, 0.f)) * (0.5f * size);
            const sl::float3 normal = cross(forward, side);
            const sl::float3 back = b - forward * size;
            for (const sl::float3& tip : {back + side, back - side, back + normal, back - normal}) {
                lines.addPoint(b, color);
                lines.addPoint(tip, color);
            }
        }

        _keyframePoints.addPoint(a, colorKeyframe);
        _keyframePoints.addPoint(b, colorKeyframe);
    }

    _arrowLines.pushToGPU();
    _linkLines.pushToGPU();
    _selectedLinkLines.pushToGPU();
    _keyframePoints.pushToGPU();
}

void MotionPriorView::drawScene(const sl::Transform& viewProjection) {
    _viewProjection = viewProjection;

    if (!_enabled || !_sceneVisible) {
        return;
    }

    // Rebuilt a few times a second at most: the statuses change every image, which is too often to
    // re-upload every arrow for.
    const auto now = std::chrono::steady_clock::now();
    if (_sceneDirty && now - _lastSceneBuild > std::chrono::milliseconds(200)) {
        rebuildScene();
        _sceneDirty = false;
        _lastSceneBuild = now;
    }

    glLineWidth(1.5f);
    _arrowLines.draw();
    glLineWidth(2.5f);
    _linkLines.draw();
    glLineWidth(5.f);
    _selectedLinkLines.draw();
    glPointSize(7.f);
    _keyframePoints.draw();
    glPointSize(1.f);
}

bool MotionPriorView::pick(int x, int y, int width, int height) {
    if (!_enabled || !_sceneVisible) {
        return false;
    }

    const float px = static_cast<float>(x);
    const float py = static_cast<float>(height - y);
    const float maxDistance = 10.f;

    float best = maxDistance;
    std::optional<uint64_t> picked;

    for (const auto& entry : _linkSegments) {
        float ax = 0.f, ay = 0.f;
        bool previousVisible = false;
        for (const sl::float3& point : entry.second) {
            float bx = 0.f, by = 0.f;
            const bool visible = project(_viewProjection, point, width, height, bx, by);
            if (visible && previousVisible) {
                const float distance = distanceToSegment(px, py, ax, ay, bx, by);
                if (distance < best) {
                    best = distance;
                    picked = entry.first;
                }
            }
            ax = bx;
            ay = by;
            previousVisible = visible;
        }
    }

    // A click on the same link, or away from every link, clears the selection.
    if (picked && _selectedLink && *picked == *_selectedLink) {
        picked.reset();
    }
    _selectedLink = picked;
    _sceneDirty = true;
    _lastSceneBuild = std::chrono::steady_clock::time_point(); // redraw the selection at once
    return true;
}

//
// 2D
//
void MotionPriorView::drawOverlay(int width, int height, bool darkMode) {
    if (!_enabled) {
        return;
    }

    updateFates();

    overlay::begin(width, height);
    if (_sceneVisible) {
        drawSelectedLinkEnds(width, height, darkMode);
    }
    drawPanel(width, height, darkMode);
    if (_timelineVisible) {
        drawTimeline(width, darkMode);
    }
    overlay::end();
}

// Names the two ends of the selected link in the 3D view: the line starts on its from keyframe and ends,
// at the arrowhead, on its to keyframe.
void MotionPriorView::drawSelectedLinkEnds(int width, int height, bool darkMode) {
    auto selected = _selectedLink ? _links.find(*_selectedLink) : _links.end();
    if (selected == _links.end()) {
        return;
    }
    const sl::MotionPriorLink& link = selected->second;
    auto from = _keyframes.find(link.from_keyframe_id);
    auto to = _keyframes.find(link.to_keyframe_id);
    if (from == _keyframes.end() || to == _keyframes.end()) {
        return;
    }

    const sl::float4 text = darkMode ? colorTextDark : colorTextLight;
    const auto label = [&](const sl::float3& point, const std::string& str, float dx) {
        float x = 0.f, y = 0.f;
        if (!project(_viewProjection, point, width, height, x, y)) {
            return;
        }
        // Left of the from end and right of the to end, so the two labels never sit on the same spot.
        const float textX = dx < 0.f ? x + dx - overlay::textWidth(fontSmall, str) : x + dx;
        overlay::text(textX, y - 4.f, fontSmall, text, str);
    };
    label(from->second.pose.getTranslation(), "from KF " + std::to_string(link.from_keyframe_id), -10.f);
    label(to->second.pose.getTranslation(), "to KF " + std::to_string(link.to_keyframe_id), 10.f);
}

void MotionPriorView::drawPanel(int width, int height, bool darkMode) {
    const sl::float4 text = darkMode ? colorTextDark : colorTextLight;
    const sl::float4 muted = darkMode ? colorMutedDark : colorMutedLight;

    const float x = static_cast<float>(width - panelWidth - 30);
    const float top = static_cast<float>(height - 30);
    const float bottom = static_cast<float>(reservedBottomHeight() + 30);
    overlay::fillRect(x, bottom, panelWidth, top - bottom, darkMode ? colorPanelDark : colorPanelLight);

    const float left = x + 16.f;
    const float right = x + panelWidth - 16.f;
    float y = top - 28.f;
    const auto row = [&](const std::string& label, const std::string& value, const sl::float4& valueColor) {
        overlay::text(left, y, fontSmall, muted, label);
        overlay::text(right - overlay::textWidth(fontSmall, value), y, fontSmall, valueColor, value);
        y -= 20.f;
    };

    overlay::text(left, y, fontBody, text, "Motion prior");
    y -= 24.f;

    // Where the samples come from, and how fast.
    std::string source = _source;
    if (source.size() > 52) {
        source = "..." + source.substr(source.size() - 49);
    }
    overlay::text(left, y, fontSmall, muted, source);
    y -= 20.f;

    std::ostringstream rate;
    if (_samples.size() > 1) {
        const double span = static_cast<double>(_samples.back().end - _samples.front().start) / 1e9;
        rate << _samples.size() << " samples, " << std::fixed << std::setprecision(1) << (span > 0. ? _samples.size() / span : 0.) << " Hz";
    } else {
        rate << _samples.size() << " samples";
    }
    overlay::text(left, y, fontSmall, muted, rate.str());
    y -= 30.f;

    // How each sample was used.
    const size_t* counts = _fateCounts;

    const auto fateRow = [&](const std::string& label, size_t count, const sl::float4& color) {
        overlay::fillRect(left, y - 1.f, 10.f, 10.f, color);
        overlay::text(left + 18.f, y, fontSmall, text, label);
        const std::string value = std::to_string(count);
        overlay::text(right - overlay::textWidth(fontSmall, value), y, fontSmall, text, value);
        y -= 20.f;
    };
    fateRow("Used for the image pose", counts[static_cast<int>(Fate::IMAGE)], colorImage);
    fateRow("Used by the map only", counts[static_cast<int>(Fate::MAP)], colorMap);
    fateRow("Rejected", counts[static_cast<int>(Fate::REJECTED)], colorRejected);
    fateRow("Unused", counts[static_cast<int>(Fate::UNUSED)], colorUnused);
    y -= 10.f;

    // The last image, and how late the samples were.
    if (!_imageStatuses.empty()) {
        const sl::RELATIVE_POSE_FUSION_STATUS last = _imageStatuses.rbegin()->second;
        const Fate lastFate = last == sl::RELATIVE_POSE_FUSION_STATUS::OK ? Fate::IMAGE
            : last == sl::RELATIVE_POSE_FUSION_STATUS::REJECTED           ? Fate::REJECTED
                                                                          : Fate::UNUSED;
        row("Last image", sl::toString(last).c_str(), fateColor(lastFate));
    }

    if (!_samples.empty()) {
        std::ostringstream delay;
        delay << std::fixed << std::setprecision(0) << "median " << _delayMedianMs << " ms, p95 " << _delayP95Ms << " ms";
        row("Arrival delay", delay.str(), text);
    }

    size_t mapOnly = 0;
    for (const auto& link : _links) {
        mapOnly += link.second.status == sl::RELATIVE_POSE_FUSION_STATUS::OK && !isUsedByImages(link.second) ? 1 : 0;
    }
    row("Keyframe links", std::to_string(_links.size()) + (mapOnly ? " (" + std::to_string(mapOnly) + " map only)" : ""), text);
    y -= 14.f;

    drawLinkDetail(left, y, right - left, darkMode);

    // The keys, at the bottom of the panel.
    overlay::text(left, bottom + 32.f, fontSmall, muted, "m   show / hide motion prior");
    overlay::text(left, bottom + 14.f, fontSmall, muted, "t   show / hide timeline");
}

//
// The selected link as the SDK reports it, next to what the samples it spans recorded: the two should
// tell the same motion.
//
void MotionPriorView::drawLinkDetail(float x, float y, float width, bool darkMode) {
    const sl::float4 text = darkMode ? colorTextDark : colorTextLight;
    const sl::float4 muted = darkMode ? colorMutedDark : colorMutedLight;

    auto selected = _selectedLink ? _links.find(*_selectedLink) : _links.end();
    if (selected == _links.end()) {
        overlay::text(x, y, fontSmall, muted, _links.empty() ? "No keyframe link yet" : "Click a keyframe link to see its status");
        return;
    }

    const sl::MotionPriorLink& link = selected->second;
    const sl::float4 color = linkColor(link);
    const auto row = [&](const std::string& label, const std::string& value, const sl::float4& valueColor) {
        overlay::text(x, y, fontSmall, muted, label);
        overlay::text(x + width - overlay::textWidth(fontSmall, value), y, fontSmall, valueColor, value);
        y -= 20.f;
    };

    overlay::line(x, y + 16.f, x + width, y + 16.f, color, 2.f);
    overlay::text(
        x,
        y - 4.f,
        fontSmall,
        text,
        "Link KF " + std::to_string(link.from_keyframe_id) + " to KF " + std::to_string(link.to_keyframe_id)
    );
    y -= 26.f;

    row("status", sl::toString(link.status).c_str(), color);
    const bool usedByImages = isUsedByImages(link);
    row("used by the images", usedByImages ? "yes" : "no, map only", usedByImages ? text : colorMap);
    row("motion", describeMotion(link.motion), text);
    {
        std::ostringstream span;
        span << std::fixed << std::setprecision(0) << (link.end_timestamp.getNanoseconds() - link.start_timestamp.getNanoseconds()) / 1e6
             << " ms";
        row("interval", span.str(), text);
    }

    // The samples within the link, composed in order: the same motion, before the tracking touched it.
    std::vector<const SampleRecord*> within;
    const uint64_t tolerance = 1000000; // 1 ms
    for (const SampleRecord& sample : _samples) {
        if (sample.start + tolerance >= link.start_timestamp.getNanoseconds()
            && sample.end <= link.end_timestamp.getNanoseconds() + tolerance) {
            within.push_back(&sample);
        }
    }
    std::sort(within.begin(), within.end(), [](const SampleRecord* a, const SampleRecord* b) {
        return a->start < b->start;
    });

    y -= 8.f;
    overlay::text(x, y, fontSmall, muted, "From the samples it spans");
    y -= 22.f;

    sl::Transform composed;
    double latest = 0.;
    for (const SampleRecord* sample : within) {
        composed = composed * sample->cameraMotion;
        latest = std::max(latest, (static_cast<double>(sample->arrival) - static_cast<double>(sample->end)) / 1e6);
    }

    row("samples", std::to_string(within.size()), text);
    if (!within.empty()) {
        row("their motion", describeMotion(composed), text);
        std::ostringstream delay;
        delay << std::fixed << std::setprecision(0) << latest << " ms";
        row("latest arrival", delay.str(), text);
    }
}

//
// The last 2 s: images and keyframes as ticks, each sample as a bar over the interval it measures with a
// whisker to the moment it was handed to the tracking.
//
void MotionPriorView::drawTimeline(int width, bool darkMode) {
    if (_imageStatuses.empty()) {
        return;
    }

    const sl::float4 text = darkMode ? colorTextDark : colorTextLight;
    const sl::float4 muted = darkMode ? colorMutedDark : colorMutedLight;

    const float x = 30.f;
    const float y = 20.f;
    const float w = static_cast<float>(width - 60);
    overlay::fillRect(x, y, w, timelineHeight, darkMode ? colorPanelDark : colorPanelLight);

    const float labelWidth = 110.f;
    const float plotLeft = x + labelWidth;
    const float plotWidth = w - labelWidth - 50.f;
    const uint64_t now = _imageStatuses.rbegin()->first;
    const uint64_t from = now > timelineSpan ? now - timelineSpan : 0;
    const auto toX = [&](uint64_t t) {
        const double clamped = static_cast<double>(std::max(from, std::min(now, t)));
        return plotLeft + static_cast<float>((clamped - static_cast<double>(from)) / static_cast<double>(timelineSpan)) * plotWidth;
    };

    const float top = y + timelineHeight;
    overlay::text(x + 14.f, top - 20.f, fontSmall, text, "Timeline");
    overlay::text(
        x + 90.f,
        top - 20.f,
        fontSmall,
        muted,
        "last 2 s   bar = interval the sample measures   whisker = until it reached the SDK"
    );

    const float imagesLane = top - 46.f;
    const float priorLane = top - 82.f;
    const float keyframeLane = top - 112.f;
    overlay::text(x + 14.f, imagesLane - 4.f, fontSmall, muted, "images");
    overlay::text(x + 14.f, priorLane - 4.f, fontSmall, muted, "motion prior");
    overlay::text(x + 14.f, keyframeLane - 4.f, fontSmall, muted, "keyframes");

    for (auto image = _imageStatuses.lower_bound(from); image != _imageStatuses.end(); ++image) {
        overlay::line(toX(image->first), imagesLane - 8.f, toX(image->first), imagesLane + 8.f, colorUnused, 2.f);
    }

    for (const auto& keyframe : _keyframes) {
        const uint64_t t = keyframe.second.timestamp.getNanoseconds();
        if (t >= from && t <= now) {
            overlay::line(toX(t), imagesLane - 12.f, toX(t), imagesLane + 12.f, colorKeyframe, 3.f);
            overlay::diamond(toX(t), keyframeLane, 5.f, colorKeyframe);
        }
    }

    // Newest first, stopping once a sample both ends and is handed over before the window.
    for (size_t i = _samples.size(); i-- > 0;) {
        const SampleRecord& sample = _samples[i];
        if (sample.end < from && sample.arrival < from) {
            break;
        }

        const Fate fate = _fates[i];
        const sl::float4 color = fateColor(fate);
        const float lane = priorLane + (i % 2 == 0 ? 7.f : -7.f);

        overlay::fillRect(toX(sample.start) + 1.f, lane - 4.f, std::max(2.f, toX(sample.end) - toX(sample.start) - 2.f), 8.f, color);
        if (sample.arrival > sample.end) {
            overlay::line(toX(sample.end), lane, toX(sample.arrival), lane, color, 2.f, fate == Fate::IMAGE ? 0.f : 4.f);
        }
    }

    overlay::line(plotLeft + plotWidth, y + 8.f, plotLeft + plotWidth, top - 30.f, muted, 1.f);
    overlay::text(plotLeft + plotWidth + 6.f, y + 10.f, fontSmall, muted, "now");
}
