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

#include "MotionPriorChannel.hpp"

#include <map>

#include "SampleConsole.hpp"

using namespace sl;

namespace {

    ERROR_CODE writeMessage(Camera& zed, const std::string& key, const nlohmann::json& body, uint64_t timestamp) {
        SVOData data;
        data.key = key;
        data.timestamp_ns = Timestamp(timestamp);
        data.setContent(body.dump());
        return zed.ingestDataIntoSVO(data);
    }

} // namespace

ERROR_CODE MotionPriorRecorder::writeConfig(Camera& zed, const Transform& sensorTransform, POSE_AXIS measuredAxes, Timestamp timestamp) {
    nlohmann::json body;
    body["v"] = motion_prior_channel::formatVersion;
    // The coordinate system the camera was opened with, which is the one the SDK reads the transform in.
    body["coordinate_system"] = writeCoordinateSystem(zed.getInitParameters().coordinate_system);
    body["sensor_transform"] = writeSensorTransform(sensorTransform);
    body["measured_axes"] = writeMeasuredAxes(measuredAxes);

    const uint64_t at = timestamp.getNanoseconds() > 0 ? timestamp.getNanoseconds() : getCurrentTimeStamp().getNanoseconds();
    _lastMessageTimestamp = std::max(_lastMessageTimestamp + 1, at);
    return writeMessage(zed, motion_prior_channel::configKey, body, _lastMessageTimestamp);
}

ERROR_CODE MotionPriorRecorder::write(Camera& zed, const std::vector<RelativePoseData>& samples, Timestamp lastImageTimestamp) {
    return write(zed, samples, lastImageTimestamp, getCurrentTimeStamp());
}

ERROR_CODE MotionPriorRecorder::write(
    Camera& zed,
    const std::vector<RelativePoseData>& samples,
    Timestamp lastImageTimestamp,
    Timestamp arrivalTimestamp
) {
    if (samples.empty()) {
        return ERROR_CODE::SUCCESS;
    }

    nlohmann::json body;
    body["v"] = motion_prior_channel::formatVersion;
    body["last_image_timestamp"] = lastImageTimestamp.getNanoseconds();
    body["samples"] = nlohmann::json::array();
    for (const RelativePoseData& sample : samples) {
        body["samples"].push_back(writeMotionPriorSample(sample));
    }

    _lastMessageTimestamp = std::max(_lastMessageTimestamp + 1, arrivalTimestamp.getNanoseconds());
    return writeMessage(zed, motion_prior_channel::sampleKey, body, _lastMessageTimestamp);
}

bool hasMotionPriorChannel(Camera& zed) {
    for (const std::string& key : zed.getSVODataKeys()) {
        if (key == motion_prior_channel::sampleKey) {
            return true;
        }
    }
    return false;
}

bool loadMotionPriorChannel(Camera& zed, MotionPriorTrace& trace) {
    trace.source = "the SVO data channel '" + motion_prior_channel::sampleKey + "'";

    // The whole channel in one read: 0, 0 is the full time range.
    std::map<Timestamp, SVOData> messages;
    ERROR_CODE status = zed.retrieveSVOData(motion_prior_channel::sampleKey, messages, 0, 0);
    if (status != ERROR_CODE::SUCCESS) {
        print("Cannot read the motion prior channel of the SVO", status);
        return false;
    }

    POSE_AXIS defaultAxes = POSE_AXIS::ALL;

    try {
        std::map<Timestamp, SVOData> configs;
        if (zed.retrieveSVOData(motion_prior_channel::configKey, configs, 0, 0) == ERROR_CODE::SUCCESS && !configs.empty()) {
            std::string content;
            configs.begin()->second.getContent(content);
            const nlohmann::json config = nlohmann::json::parse(content);

            if (config.contains("coordinate_system")) {
                trace.sensorTransformCoordinateSystem = readCoordinateSystem(config.at("coordinate_system"));
            }
            if (config.contains("sensor_transform")) {
                readSensorTransform(config.at("sensor_transform"), trace.sensorTransform);
                trace.hasSensorTransform = true;
            }
            if (config.contains("measured_axes")) {
                defaultAxes = readMeasuredAxes(config.at("measured_axes"));
            }
        }

        // The map is ordered by timestamp, which is the order the samples arrived in.
        Timestamp previousEnd = 0;
        for (const auto& message : messages) {
            std::string content;
            message.second.getContent(content);
            const nlohmann::json body = nlohmann::json::parse(content);

            if (body.value("v", 0) != motion_prior_channel::formatVersion) {
                throw std::runtime_error("unsupported message version " + std::to_string(body.value("v", 0)));
            }

            // Handed over before the first image after the last one grabbed when it arrived. Recordings
            // without it fall back on the arrival time.
            const uint64_t lastImage = body.value("last_image_timestamp", uint64_t(0));
            const uint64_t release = lastImage > 0 ? lastImage + 1 : message.first.getNanoseconds();

            for (const auto& node : body.at("samples")) {
                RelativePoseData sample = readMotionPriorSample(node, previousEnd, defaultAxes);
                previousEnd = sample.end_timestamp;
                trace.samples.push_back(sample);
                trace.releaseTimestamps.push_back(release);
                trace.arrivalTimestamps.push_back(message.first.getNanoseconds());
            }
        }
    } catch (const std::exception& e) {
        print("Invalid motion prior channel in the SVO: " + std::string(e.what()), ERROR_CODE::FAILURE, false);
        return false;
    }

    if (trace.samples.empty()) {
        print("The motion prior channel of the SVO holds no sample", ERROR_CODE::FAILURE, false);
        return false;
    }

    return true;
}
