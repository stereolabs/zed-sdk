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

#include <algorithm>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

#include <sl/Camera.hpp>

//
// Writes a motion prior into the SVO being recorded, in the two sl::SVOData channels the Positional
// Tracking sample replays on its own (see its MotionPriorChannel.hpp for the reading side):
//
//   "motion_prior_config"   once, before the first sample:
//     { "v": 1, "coordinate_system": "IMAGE",
//       "sensor_transform": { "translation": [...], "rotation_vector": [...] } }
//
//   "motion_prior_samples"  one message per arrival:
//     { "v": 1, "last_image_timestamp": 1712345678866567890,
//       "samples": [ { "start_timestamp": ..., "end_timestamp": ...,
//                      "translation": [...], "rotation_vector": [...],
//                      "covariance": [ ...36 values... ], "measured_axes": ["RX", "RY", "RZ"] } ] }
//
// "last_image_timestamp" is the last image read when the sample arrived: the replay hands the sample to
// the tracking before the image that follows it, exactly as it was live.
//
// The format belongs to the samples, not to the ZED SDK API. It is plain JSON written by hand here so
// this sample needs no JSON library.
//
class MotionPriorChannel {
public:
    // Once, before the first sample. The sensor transform is written in the coordinate system the camera
    // was opened with, which must be one the reading side knows by name: this sample uses IMAGE.
    sl::ERROR_CODE writeConfig(sl::Camera& zed, const sl::Transform& sensorTransform, sl::Timestamp timestamp) {
        std::ostringstream body;
        body.imbue(std::locale::classic()); // JSON numbers, whatever locale the application set
        body << std::setprecision(9) << "{\"v\":1,\"coordinate_system\":\"IMAGE\",\"sensor_transform\":{\"translation\":"
             << float3(sensorTransform.getTranslation()) << ",\"rotation_vector\":" << float3(sensorTransform.getRotationVector()) << "}}";
        return write(zed, "motion_prior_config", body.str(), timestamp);
    }

    // One sample, as it reaches the application: written as given to ingestMotionPrior(), its covariance
    // and measured axes included, so the replay hands the tracking exactly the same thing.
    sl::ERROR_CODE
    writeSample(sl::Camera& zed, const sl::RelativePoseData& sample, sl::Timestamp lastImageTimestamp, sl::Timestamp arrivalTimestamp) {
        std::ostringstream body;
        body.imbue(std::locale::classic()); // JSON numbers, whatever locale the application set
        body << std::setprecision(9) << "{\"v\":1,\"last_image_timestamp\":" << lastImageTimestamp.getNanoseconds()
             << ",\"samples\":[{\"start_timestamp\":" << sample.start_timestamp.getNanoseconds()
             << ",\"end_timestamp\":" << sample.end_timestamp.getNanoseconds()
             << ",\"translation\":" << float3(sample.motion.getTranslation())
             << ",\"rotation_vector\":" << float3(sample.motion.getRotationVector()) << ",\"covariance\":[";
        for (int i = 0; i < 36; i++) {
            body << (i > 0 ? "," : "") << sample.covariance[i];
        }
        body << "],\"measured_axes\":" << measuredAxes(sample.measured_axes);
        if (sample.is_stationary) {
            body << ",\"is_stationary\":true";
        }
        body << "}]}";
        return write(zed, "motion_prior_samples", body.str(), arrivalTimestamp);
    }

private:
    // The axes set in axes, by the names the reading side knows.
    static std::string measuredAxes(sl::POSE_AXIS axes) {
        static const char* const names[6] = {"TX", "TY", "TZ", "RX", "RY", "RZ"};
        std::string list = "[";
        for (unsigned axis = 0; axis < 6; axis++) {
            if ((axes & static_cast<sl::POSE_AXIS>(1u << axis)) != sl::POSE_AXIS::NONE) {
                list += (list.size() > 1 ? ",\"" : "\"") + std::string(names[axis]) + "\"";
            }
        }
        return list + "]";
    }

    struct float3 {
        explicit float3(const sl::float3& value)
            : v(value) { }
        sl::float3 v;
        friend std::ostream& operator<<(std::ostream& os, const float3& f) {
            return os << "[" << f.v.x << "," << f.v.y << "," << f.v.z << "]";
        }
    };

    sl::ERROR_CODE write(sl::Camera& zed, const std::string& key, const std::string& body, sl::Timestamp timestamp) {
        // The SVO keys its data by timestamp: two messages at the same instant would lose one.
        _lastTimestamp = std::max(_lastTimestamp + 1, timestamp.getNanoseconds());

        sl::SVOData data;
        data.key = key;
        data.timestamp_ns = sl::Timestamp(_lastTimestamp);
        data.setContent(body);
        return zed.ingestDataIntoSVO(data);
    }

    uint64_t _lastTimestamp = 0;
};
