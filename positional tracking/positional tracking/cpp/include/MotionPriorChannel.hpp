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

#include <string>
#include <vector>

#include <sl/Camera.hpp>

#include "MotionPriorFile.hpp"

//
// The motion prior, stored in the SVO next to the images it was used with, so a replay hands each
// sample to the tracking at the same point it reached the application live: late samples stay late.
//
// Two data channels (sl::SVOData), both JSON:
//
//   "motion_prior_config"   once, before the first sample:
//     { "v": 1,
//       "coordinate_system": "RIGHT_HANDED_Y_UP",   // the one the camera was opened with when recording
//       "sensor_transform": { "translation": [...], "rotation_vector": [...] },
//       "measured_axes": ["TX", "TZ", "RY"] }
//
//   "motion_prior_samples"  one message per arrival, stamped with the time it reached the application:
//     { "v": 1,
//       "last_image_timestamp": 1712345678866567890,   // the last image grabbed when it arrived
//       "samples": [ { ...one sample, same fields as the --motion-prior file... } ] }
//
// "last_image_timestamp" is what the replay goes by: the samples are handed over before the image
// that follows it. The arrival time alone would make every replay look later than live, because
// the image timestamp is the capture time and grab() runs some tens of ms after it.
//
// The format belongs to this sample: it is not part of the ZED SDK API.
//
// Neither key may be contained in the other, nor in any other channel of the SVO: retrieveSVOData()
// reads every channel whose name contains the key asked for.
//
namespace motion_prior_channel {
    const std::string sampleKey = "motion_prior_samples";
    const std::string configKey = "motion_prior_config";
    const int formatVersion = 1;
} // namespace motion_prior_channel

//
// Recording side: what an application calls while recording an SVO (sl::Camera::enableRecording()).
//
class MotionPriorRecorder {
public:
    // Once, before the first samples. Stamped with the current time unless timestamp is given.
    sl::ERROR_CODE writeConfig(
        sl::Camera& zed,
        const sl::Transform& sensorTransform,
        sl::POSE_AXIS measuredAxes,
        sl::Timestamp timestamp = sl::Timestamp(0)
    );

    // Every sample that reached the application together, as one message. Call it where the samples
    // are passed to ingestMotionPrior(), with the timestamp of the last image grabbed so far.
    // The message is stamped with the current time, kept strictly increasing: the SVO keys its data
    // by timestamp, so two messages at the same instant would lose one.
    sl::ERROR_CODE write(sl::Camera& zed, const std::vector<sl::RelativePoseData>& samples, sl::Timestamp lastImageTimestamp);

    // Same, with the arrival instant given instead of read from the clock: for a tool that re-records
    // an SVO and simulates when each sample would have arrived.
    sl::ERROR_CODE write(
        sl::Camera& zed,
        const std::vector<sl::RelativePoseData>& samples,
        sl::Timestamp lastImageTimestamp,
        sl::Timestamp arrivalTimestamp
    );

private:
    uint64_t _lastMessageTimestamp = 0;
};

//
// Replay side.
//

// Whether the SVO being replayed carries a motion prior channel.
bool hasMotionPriorChannel(sl::Camera& zed);

// Reads the whole channel at once (an hour of a 10 Hz source is a few MB) into trace, release times
// included. Reports the reason and returns false on a malformed message.
bool loadMotionPriorChannel(sl::Camera& zed, MotionPriorTrace& trace);
