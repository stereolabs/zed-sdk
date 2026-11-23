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

#include "MotionPriorFile.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

#include "SampleConsole.hpp"

using namespace sl;

//
// Reading and writing the JSON. Any missing optional field falls back to the sl::RelativePoseData
// default, so a minimal sample only needs a timestamp, a motion and an uncertainty.
//
namespace {

    // In the bit order of sl::POSE_AXIS, which is also the order of the covariance rows.
    const char* const axisNames[6] = {"TX", "TY", "TZ", "RX", "RY", "RZ"};

    // In the order of sl::COORDINATE_SYSTEM. The enumerator names rather than sl::toString(), whose
    // wording is meant for display.
    const char* const coordinateSystemNames[]
        = {"IMAGE", "LEFT_HANDED_Y_UP", "RIGHT_HANDED_Y_UP", "RIGHT_HANDED_Z_UP", "LEFT_HANDED_Z_UP", "RIGHT_HANDED_Z_UP_X_FWD"};
    const int coordinateSystemCount = static_cast<int>(sizeof(coordinateSystemNames) / sizeof(coordinateSystemNames[0]));

    bool readFloat3(const nlohmann::json& node, const std::string& key, sl::float3& out) {
        if (!node.contains(key)) {
            return false;
        }

        const auto& value = node.at(key);
        if (!value.is_array() || value.size() != 3) {
            throw std::runtime_error("'" + key + "' must be an array of 3 numbers");
        }

        out = sl::float3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
        return true;
    }

    nlohmann::json writeFloat3(const sl::float3& value) {
        return nlohmann::json::array({value.x, value.y, value.z});
    }

    // Reads "rotation_vector" (axis scaled by the angle, in radians) or "quaternion" ([x, y, z, w])
    // into the rotation part of a transform. Neither present leaves it at identity.
    void readRotation(const nlohmann::json& node, sl::Transform& transform) {
        sl::float3 rotationVector;
        if (readFloat3(node, "rotation_vector", rotationVector)) {
            transform.setRotationVector(rotationVector);
            return;
        }

        if (node.contains("quaternion")) {
            const auto& value = node.at("quaternion");
            if (!value.is_array() || value.size() != 4) {
                throw std::runtime_error("'quaternion' must be an array of 4 numbers ([x, y, z, w])");
            }

            transform.setOrientation(
                sl::Orientation(sl::float4(value[0].get<float>(), value[1].get<float>(), value[2].get<float>(), value[3].get<float>()))
            );
        }
    }

    void readTranslation(const nlohmann::json& node, sl::Transform& transform) {
        sl::float3 translation;
        if (readFloat3(node, "translation", translation)) {
            transform.setTranslation(sl::Translation(translation.x, translation.y, translation.z));
        }
    }

    // Fills the covariance from the full 36-value matrix when given, otherwise from the
    // per-axis standard deviations, which cover the common case of an uncorrelated source.
    void readCovariance(const nlohmann::json& node, float (&covariance)[36]) {
        if (node.contains("covariance")) {
            const auto& value = node.at("covariance");
            if (!value.is_array() || value.size() != 36) {
                throw std::runtime_error("'covariance' must be an array of 36 numbers (row-major, [tx, ty, tz, rx, ry, rz])");
            }

            for (size_t i = 0; i < 36; i++) {
                covariance[i] = value[i].get<float>();
            }
            return;
        }

        sl::float3 translationStd, rotationStd;

        if (readFloat3(node, "translation_std", translationStd)) {
            for (int i = 0; i < 3; i++) {
                const float sigma = translationStd[i];
                covariance[i * 6 + i] = sigma * sigma;
            }
        }

        if (readFloat3(node, "rotation_std", rotationStd)) {
            for (int i = 0; i < 3; i++) {
                const float sigma = rotationStd[i];
                covariance[(3 + i) * 6 + (3 + i)] = sigma * sigma;
            }
        }
    }

    //
    // What was loaded, in the terms that decide whether it will be used: the window the samples cover
    // has to overlap the image timestamps of the input, and the interval of a sample has to match the
    // interval between two images, or the tracking drops it as out of sync.
    //
    void printTraceSummary(const MotionPriorTrace& trace) {
        uint64_t windowStart = UINT64_MAX, windowEnd = 0;
        uint64_t shortestInterval = UINT64_MAX, longestInterval = 0;
        double travelled = 0.;

        // Every sample covers a real interval (the reader refuses one that does not), so the window
        // runs from the earliest start to the latest end.
        for (const RelativePoseData& sample : trace.samples) {
            const uint64_t start = sample.start_timestamp.getNanoseconds();
            const uint64_t end = sample.end_timestamp.getNanoseconds();

            windowStart = std::min(windowStart, start);
            windowEnd = std::max(windowEnd, end);
            shortestInterval = std::min(shortestInterval, end - start);
            longestInterval = std::max(longestInterval, end - start);
            travelled += sample.motion.getTranslation().norm();
        }

        std::ostringstream summary;
        summary << std::fixed << std::setprecision(3) << "Motion prior from " << trace.source << ": " << trace.samples.size()
                << " samples over [" << windowStart << ", " << windowEnd << "] ns (" << (windowEnd - windowStart) / 1e9 << " s), interval "
                << shortestInterval / 1e6 << " ms to " << longestInterval / 1e6 << " ms, " << std::setprecision(2) << travelled
                << " m of motion";
        print(summary.str());
    }

    // How long after the end of its interval each sample reached the application. A sample can only shape
    // the pose of the image it ends at if it is there before that image is processed.
    void printArrivalDelays(const MotionPriorTrace& trace) {
        std::vector<double> delays;
        delays.reserve(trace.samples.size());

        for (size_t i = 0; i < trace.samples.size(); i++) {
            const double end = static_cast<double>(trace.samples[i].end_timestamp.getNanoseconds());
            delays.push_back((static_cast<double>(trace.arrivalTimestamps[i]) - end) / 1e6);
        }

        std::sort(delays.begin(), delays.end());

        std::ostringstream message;
        message << std::fixed << std::setprecision(1) << "Motion prior arriving " << delays[delays.size() / 2]
                << " ms after the end of its interval (median), " << delays[(delays.size() * 95) / 100] << " ms (p95), " << delays.back()
                << " ms at most";
        print(message.str());
    }

    // Where the external source sits on the rig, as read from the file. Printed because it is
    // applied to every sample by the tracking: a wrong one turns good measurements into drift.
    void printSensorTransform(const MotionPriorTrace& trace) {
        const sl::float3 translation = trace.sensorTransform.getTranslation();
        const sl::float3 rotation = trace.sensorTransform.getRotationVector();

        std::ostringstream message;
        message << std::fixed << std::setprecision(3) << "Motion prior sensor transform: translation [" << translation.x << ", "
                << translation.y << ", " << translation.z << "], rotation vector [" << rotation.x << ", " << rotation.y << ", "
                << rotation.z << "] rad, in " << writeCoordinateSystem(trace.sensorTransformCoordinateSystem).get<std::string>();
        print(message.str());
    }

    // Same question as printTraceSummary, but answered against the images actually being replayed.
    void printImageAlignment(const MotionPriorTrace& trace, Timestamp imageTimestamp) {
        const uint64_t image = imageTimestamp.getNanoseconds();
        const uint64_t firstSample = trace.samples.front().end_timestamp.getNanoseconds();
        const uint64_t lastSample = trace.samples.back().end_timestamp.getNanoseconds();

        std::ostringstream message;
        message << std::fixed << std::setprecision(3) << "First image at " << image << " ns, motion prior samples end between "
                << firstSample << " and " << lastSample << " ns (" << (double(firstSample) - double(image)) / 1e9 << " s to "
                << (double(lastSample) - double(image)) / 1e9 << " s from that image)";
        print(message.str());

        if (lastSample < image || firstSample > image + 60ULL * 60ULL * 1000000000ULL) {
            print(
                "The motion prior samples do not overlap the images: check that they were generated from "
                "this recording and carry nanosecond timestamps on the same reference as TIME_REFERENCE::IMAGE",
                ERROR_CODE::FAILURE,
                false
            );
        }
    }

} // namespace

// sl::fromString() knows the single axes (TX ... RZ) as well as TRANSLATION, ROTATION, ALL and NONE.
POSE_AXIS readMeasuredAxes(const nlohmann::json& value) {
    if (!value.is_array()) {
        throw std::runtime_error("'measured_axes' must be an array of axis names");
    }

    POSE_AXIS axes = POSE_AXIS::NONE;
    for (const auto& name : value) {
        POSE_AXIS axis;
        if (sl::fromString(sl::String(name.get<std::string>().c_str()), axis)) { // returns true on error
            throw std::runtime_error("unknown axis '" + name.get<std::string>() + "'");
        }
        axes |= axis;
    }

    return axes;
}

nlohmann::json writeMeasuredAxes(POSE_AXIS axes) {
    nlohmann::json names = nlohmann::json::array();
    for (unsigned axis = 0; axis < 6; axis++) {
        if ((axes & static_cast<POSE_AXIS>(1u << axis)) != POSE_AXIS::NONE) {
            names.push_back(axisNames[axis]);
        }
    }
    return names;
}

void readSensorTransform(const nlohmann::json& node, sl::Transform& transform) {
    readTranslation(node, transform);
    readRotation(node, transform);
}

nlohmann::json writeSensorTransform(const sl::Transform& transform) {
    return {
        {"translation",     writeFloat3(transform.getTranslation())   },
        {"rotation_vector", writeFloat3(transform.getRotationVector())}
    };
}

COORDINATE_SYSTEM readCoordinateSystem(const nlohmann::json& value) {
    if (!value.is_string()) {
        throw std::runtime_error("'coordinate_system' must be a string such as \"IMAGE\" or \"RIGHT_HANDED_Y_UP\"");
    }

    const std::string name = value.get<std::string>();
    for (int i = 0; i < coordinateSystemCount; i++) {
        if (name == coordinateSystemNames[i]) {
            return static_cast<COORDINATE_SYSTEM>(i);
        }
    }
    throw std::runtime_error("unknown coordinate system '" + name + "'");
}

nlohmann::json writeCoordinateSystem(COORDINATE_SYSTEM coordinateSystem) {
    const int index = static_cast<int>(coordinateSystem);
    if (index < 0 || index >= coordinateSystemCount) {
        throw std::runtime_error("no name for coordinate system " + std::to_string(index));
    }
    return coordinateSystemNames[index];
}

bool expressSensorTransformIn(MotionPriorTrace& trace, COORDINATE_SYSTEM cameraCoordinateSystem) {
    if (trace.sensorTransformCoordinateSystem == cameraCoordinateSystem) {
        return true;
    }

    const Matrix4f conversion = getCoordinateTransformConversion4f(trace.sensorTransformCoordinateSystem, cameraCoordinateSystem);

    // A conversion between a left-handed and a right-handed system mirrors an axis: applied to the
    // transform it would no longer be a rotation, and ingestMotionPrior() refuses those.
    const float determinant = conversion(0, 0) * (conversion(1, 1) * conversion(2, 2) - conversion(1, 2) * conversion(2, 1))
        - conversion(0, 1) * (conversion(1, 0) * conversion(2, 2) - conversion(1, 2) * conversion(2, 0))
        + conversion(0, 2) * (conversion(1, 0) * conversion(2, 1) - conversion(1, 1) * conversion(2, 0));
    if (determinant < 0.f) {
        print(
            "The motion prior sensor transform is written in "
                + writeCoordinateSystem(trace.sensorTransformCoordinateSystem).get<std::string>() + " and the camera uses "
                + writeCoordinateSystem(cameraCoordinateSystem).get<std::string>()
                + ", which have opposite handedness: convert the samples and the transform before replaying them",
            ERROR_CODE::FAILURE,
            false
        );
        return false;
    }

    // On the left only: the columns of the rotation are the axes of the source, written in camera
    // coordinates, so each is converted like a point. The samples themselves are along the axes of the
    // source and do not change.
    trace.sensorTransform = Transform(conversion * trace.sensorTransform);
    trace.sensorTransformCoordinateSystem = cameraCoordinateSystem;
    return true;
}

RelativePoseData readMotionPriorSample(const nlohmann::json& node, Timestamp previousEnd, POSE_AXIS defaultAxes) {
    RelativePoseData sample;

    if (!node.contains("end_timestamp")) {
        throw std::runtime_error("a sample is missing its 'end_timestamp'");
    }

    sample.end_timestamp = sl::Timestamp(node.at("end_timestamp").get<uint64_t>());

    // Chaining is the common case: one sample starts where the previous one ended, so the
    // tracking sees no gap. Only the first sample has nothing to chain onto.
    if (!node.contains("start_timestamp") && previousEnd == sl::Timestamp(0)) {
        throw std::runtime_error("the first sample must carry a 'start_timestamp' saying when its interval starts");
    }

    sample.start_timestamp = node.contains("start_timestamp") ? sl::Timestamp(node.at("start_timestamp").get<uint64_t>()) : previousEnd;

    // The same interval ingestMotionPrior() requires: a real start instant, strictly before the end.
    if (sample.start_timestamp == sl::Timestamp(0) || sample.end_timestamp <= sample.start_timestamp) {
        throw std::runtime_error(
            "a sample must cover a strictly increasing, non-zero interval (got " + std::to_string(sample.end_timestamp.getNanoseconds())
            + " after " + std::to_string(sample.start_timestamp.getNanoseconds()) + ")"
        );
    }

    readTranslation(node, sample.motion);
    readRotation(node, sample.motion);
    readCovariance(node, sample.covariance);

    sample.measured_axes = node.contains("measured_axes") ? readMeasuredAxes(node.at("measured_axes")) : defaultAxes;
    sample.is_stationary = node.value("is_stationary", false);

    return sample;
}

nlohmann::json writeMotionPriorSample(const RelativePoseData& sample) {
    nlohmann::json node;
    node["start_timestamp"] = sample.start_timestamp.getNanoseconds();
    node["end_timestamp"] = sample.end_timestamp.getNanoseconds();
    node["translation"] = writeFloat3(sample.motion.getTranslation());
    node["rotation_vector"] = writeFloat3(sample.motion.getRotationVector());
    node["covariance"] = std::vector<float>(std::begin(sample.covariance), std::end(sample.covariance));
    node["measured_axes"] = writeMeasuredAxes(sample.measured_axes);
    if (sample.is_stationary) {
        node["is_stationary"] = true;
    }
    return node;
}

bool loadMotionPriorFile(const std::string& path, MotionPriorTrace& trace) {
    std::ifstream file(path);
    if (!file.is_open()) {
        print("Cannot open " + path, ERROR_CODE::FAILURE, false);
        return false;
    }

    trace.source = path;

    try {
        nlohmann::json root = nlohmann::json::parse(file, nullptr, true, true); // allow comments

        if (root.contains("coordinate_system")) {
            trace.sensorTransformCoordinateSystem = readCoordinateSystem(root.at("coordinate_system"));
        }

        // Left at identity when the file carries no "sensor_transform", which is what a source
        // already measuring the motion of the camera itself needs.
        if (root.contains("sensor_transform")) {
            readSensorTransform(root.at("sensor_transform"), trace.sensorTransform);
            trace.hasSensorTransform = true;
        }

        POSE_AXIS fileAxes = POSE_AXIS::ALL;
        if (root.contains("measured_axes")) {
            fileAxes = readMeasuredAxes(root.at("measured_axes"));
        }

        if (!root.contains("samples") || !root.at("samples").is_array()) {
            print("Missing 'samples' array in " + path, ERROR_CODE::FAILURE, false);
            return false;
        }

        sl::Timestamp previousEnd = 0;
        trace.samples.reserve(root.at("samples").size());
        trace.releaseTimestamps.reserve(root.at("samples").size());
        trace.arrivalTimestamps.reserve(root.at("samples").size());

        for (const auto& node : root.at("samples")) {
            RelativePoseData sample = readMotionPriorSample(node, previousEnd, fileAxes);

            // Without a recorded arrival, the sample is taken to arrive the instant its interval ends,
            // which is the most favourable case: it is always there in time for its image.
            const uint64_t arrival = node.value("arrival_timestamp", sample.end_timestamp.getNanoseconds());
            trace.releaseTimestamps.push_back(arrival);
            trace.arrivalTimestamps.push_back(arrival);

            previousEnd = sample.end_timestamp;
            trace.samples.push_back(sample);
        }
    } catch (const std::exception& e) {
        print("Invalid motion prior file " + path + ": " + e.what(), ERROR_CODE::FAILURE, false);
        return false;
    }

    if (trace.samples.empty()) {
        print("No motion prior sample in " + path, ERROR_CODE::FAILURE, false);
        return false;
    }

    return true;
}

void finalizeMotionPriorTrace(MotionPriorTrace& trace, double delayMilliseconds) {
    const uint64_t delay = static_cast<uint64_t>(std::max(0., delayMilliseconds) * 1e6);
    for (uint64_t& release : trace.releaseTimestamps) {
        release += delay;
    }
    for (uint64_t& arrival : trace.arrivalTimestamps) {
        arrival += delay;
    }

    // Stable, so samples released together go in the order they arrived in.
    std::vector<size_t> order(trace.samples.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&trace](size_t a, size_t b) {
        return trace.releaseTimestamps[a] < trace.releaseTimestamps[b];
    });

    std::vector<RelativePoseData> samples;
    std::vector<uint64_t> releases, arrivals;
    samples.reserve(order.size());
    releases.reserve(order.size());
    arrivals.reserve(order.size());
    for (size_t i : order) {
        samples.push_back(trace.samples[i]);
        releases.push_back(trace.releaseTimestamps[i]);
        arrivals.push_back(trace.arrivalTimestamps[i]);
    }
    trace.samples.swap(samples);
    trace.releaseTimestamps.swap(releases);
    trace.arrivalTimestamps.swap(arrivals);
    trace.next = 0;

    printTraceSummary(trace);
    printArrivalDelays(trace);

    if (trace.hasSensorTransform) {
        printSensorTransform(trace);
    }
}

void ingestMotionPriorUpTo(Camera& zed, MotionPriorTrace& trace, Timestamp imageTimestamp) {
    if (trace.samples.empty()) {
        return;
    }

    static bool alignmentReported = false;
    if (!alignmentReported) {
        alignmentReported = true;
        printImageAlignment(trace, imageTimestamp);
    }

    while (trace.next < trace.samples.size() && trace.releaseTimestamps[trace.next] <= imageTimestamp.getNanoseconds()) {
        // Where the source sits on the rig.
        RelativePoseData sample = trace.samples[trace.next];
        sample.sensor_transform = trace.sensorTransform;
        ERROR_CODE status = zed.ingestMotionPrior(sample);

        // Counted rather than printed: a source that is late on purpose (--motion-prior-delay) would
        // otherwise print a line per sample.
        if (status != ERROR_CODE::SUCCESS) {
            if (trace.refusals.empty()) {
                print("Motion prior sample " + std::to_string(trace.next) + " refused, counting the others until the end", status);
            }
            trace.refusals[status]++;
        }

        trace.next++;
    }
}

void rewindMotionPrior(MotionPriorTrace& trace, Timestamp imageTimestamp) {
    const uint64_t image = imageTimestamp.getNanoseconds();

    // The first sample ending after that image: anything earlier describes motion the restarted tracking
    // will never see an image for.
    trace.next = 0;
    while (trace.next < trace.samples.size() && trace.samples[trace.next].end_timestamp.getNanoseconds() <= image) {
        trace.next++;
    }
}

void recordRelativePoseFusionStatus(MotionPriorTrace& trace, RELATIVE_POSE_FUSION_STATUS status) {
    static RELATIVE_POSE_FUSION_STATUS lastStatus = RELATIVE_POSE_FUSION_STATUS::LAST;

    trace.statusCounts[status]++;
    trace.imagesWithStatus++;

    if (status != lastStatus) {
        lastStatus = status;
        print("Relative pose fusion status: " + std::string(toString(lastStatus).c_str()));
    }
}

void printMotionPriorSummary(const MotionPriorTrace& trace) {
    if (trace.imagesWithStatus == 0) {
        return;
    }

    printHeader("Motion Prior");

    // Anything short of all of the samples means the run stopped early, or the samples are stamped
    // past the last image and were never handed over at all.
    print("Ingested " + std::to_string(trace.next) + " of " + std::to_string(trace.samples.size()) + " samples from " + trace.source);

    for (const auto& entry : trace.refusals) {
        print("Refused " + std::to_string(entry.second) + " samples", entry.first);
    }

    for (const auto& entry : trace.statusCounts) {
        const double percent = 100. * static_cast<double>(entry.second) / static_cast<double>(trace.imagesWithStatus);

        std::ostringstream line;
        line << std::string(toString(entry.first).c_str()) << " on " << entry.second << " images (" << std::fixed << std::setprecision(1)
             << percent << "%)";
        print(line.str());
    }

    // What to do about it. Each of these is a configuration problem with a different fix, and the
    // status alone does not say which one: see RELATIVE_POSE_FUSION_STATUS in the API documentation.
    if (trace.statusCounts.count(RELATIVE_POSE_FUSION_STATUS::OUT_OF_SYNC)) {
        print("Some samples did not cover the interval of an image: check that your motion source and the "
              "camera share a time reference");
    }

    if (trace.statusCounts.count(RELATIVE_POSE_FUSION_STATUS::REJECTED)) {
        print("Some samples disagreed with what the camera sees and were discarded: check the "
              "sensor transform, the sign convention and the units of the source, and the "
              "uncertainty given with each sample");
    }

    if (!trace.statusCounts.count(RELATIVE_POSE_FUSION_STATUS::OK)) {
        print("The external motion was never used by the tracking", ERROR_CODE::FAILURE, false);
    }
}
