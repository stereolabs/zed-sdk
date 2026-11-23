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

#include <map>
#include <string>
#include <vector>

#include <sl/Camera.hpp>

#include "json.hpp"

//
// Motion measured by an external source, replayed to the SDK. It comes either from a JSON file
// (--motion-prior) or from the data channel of the SVO being replayed (see MotionPriorChannel).
// File layout (translations in the coordinate units selected at init, rotations in radians):
//
//   {
//     "coordinate_system": "IMAGE",               // optional, default IMAGE: the sl::COORDINATE_SYSTEM
//                                                 // sensor_transform is written in, converted on load
//     "sensor_transform": {                       // optional, default identity: where the source
//       "translation": [0.0, 0.0, 0.0],           // sits on the rig and where its axes point
//       "rotation_vector": [0.0, 0.0, 0.0]        // "quaternion": [x, y, z, w] also accepted
//     },                                          // applied to every sample, see sensorTransform
//
//     "measured_axes": ["TX", "TY", "RZ"],        // optional, default all six
//     "samples": [
//       {
//         "end_timestamp": 1712345678901234567,   // ns, same reference as TIME_REFERENCE::IMAGE
//         "start_timestamp": 1712345678801234567, // defaults to the previous sample; required on the first one
//         "arrival_timestamp": 1712345678931234567, // optional, when the application received it: the
//                                                 // replay hands it over then, not at end_timestamp
//         "translation": [0.02, 0.0, 0.0],
//         "rotation_vector": [0.0, 0.0, 0.004],   // "quaternion": [x, y, z, w] also accepted
//         "translation_std": [0.01, 0.01, 0.01],  // shorthand for a diagonal covariance
//         "rotation_std": [0.001, 0.001, 0.002],
//         "covariance": [ ... 36 values ... ],    // optional, wins over the *_std shorthand
//         "is_stationary": false,                 // optional
//         "measured_axes": ["TX", "TY", "RZ"]     // optional, overrides the file-level value
//       }
//     ]
//   }
//
struct MotionPriorTrace {
    std::vector<sl::RelativePoseData> samples;

    // When each sample is handed to the tracking, in the image clock: samples[i] is ingested before the
    // first image stamped at or after releaseTimestamps[i]. Live, that is when the sample reached the
    // application; replayed, it is the arrival recorded with it, or its end_timestamp when nothing was
    // recorded. Kept sorted with the samples, so the replay is a single cursor.
    std::vector<uint64_t> releaseTimestamps;

    // When each sample reached the application, for showing how late it was. Not always the release:
    // a sample recorded in an SVO is released by the last image grabbed when it arrived.
    std::vector<uint64_t> arrivalTimestamps;

    // Where the samples come from, for the console and the viewer.
    std::string source;

    // Where the external source sits on the rig and where its axes point, read from the
    // "sensor_transform" of the file or the configuration of the SVO channel. Handed to the tracking
    // as RelativePoseData::sensor_transform of every sample, which is what lets it move the
    // measurements of the source into the camera frame. Identity when none is given, meaning the
    // samples are taken to already be the motion of the camera itself.
    // Expressed in sensorTransformCoordinateSystem, the one the file or the channel declares, until
    // expressSensorTransformIn() moves it into the coordinate system the camera was opened with.
    sl::Transform sensorTransform;
    bool hasSensorTransform = false;
    sl::COORDINATE_SYSTEM sensorTransformCoordinateSystem = sl::COORDINATE_SYSTEM::IMAGE;

    size_t next = 0; // first sample not handed to the tracking yet

    // How many images the tracking reported each outcome on, i.e. whether the external source is
    // really being used: a source that never reaches the tracking and one that is discarded both leave
    // poses that look as if it had never been connected.
    std::map<sl::RELATIVE_POSE_FUSION_STATUS, size_t> statusCounts;
    size_t imagesWithStatus = 0;

    // Samples ingestMotionPrior() refused, by the reason it gave. A sample arriving too late for the
    // tracking to take it any more is refused here, so this is where --motion-prior-delay shows.
    std::map<sl::ERROR_CODE, size_t> refusals;
};

//
// One sample, in the JSON layout above. Shared by the file and the SVO channel, so both read and
// write exactly the same fields. readMotionPriorSample() throws std::runtime_error on a malformed
// sample; previousEnd is the end of the sample before it, which a sample without a start chains onto.
//
sl::RelativePoseData readMotionPriorSample(const nlohmann::json& node, sl::Timestamp previousEnd, sl::POSE_AXIS defaultAxes);
nlohmann::json writeMotionPriorSample(const sl::RelativePoseData& sample);

sl::POSE_AXIS readMeasuredAxes(const nlohmann::json& value);
nlohmann::json writeMeasuredAxes(sl::POSE_AXIS axes);

void readSensorTransform(const nlohmann::json& node, sl::Transform& transform);
nlohmann::json writeSensorTransform(const sl::Transform& transform);

// The sl::COORDINATE_SYSTEM enumerator name ("IMAGE", "RIGHT_HANDED_Y_UP", ...). readCoordinateSystem()
// throws std::runtime_error on any other value.
sl::COORDINATE_SYSTEM readCoordinateSystem(const nlohmann::json& value);
nlohmann::json writeCoordinateSystem(sl::COORDINATE_SYSTEM coordinateSystem);

// RelativePoseData::sensor_transform is read in the coordinate system the camera
// was opened with, so the transform must be converted when the file or the channel uses another one: left
// as is, its rotation turns the motion about the wrong axes, and the tracking rejects it as soon as the
// camera rotates. Only the camera side changes, the axes of the source stay where they are.
// Reports the reason and returns false when the two coordinate systems differ in handedness, which a
// rotation cannot express.
bool expressSensorTransformIn(MotionPriorTrace& trace, sl::COORDINATE_SYSTEM cameraCoordinateSystem);

// Reads the whole trace up front, so the replay costs nothing in the grab loop. Reports the
// reason and returns false on a missing, malformed or empty file.
bool loadMotionPriorFile(const std::string& path, MotionPriorTrace& trace);

// Once the samples are in, from either source: delays every release and arrival by delayMilliseconds (to
// see what late samples do to the tracking), sorts by release and prints what was loaded.
void finalizeMotionPriorTrace(MotionPriorTrace& trace, double delayMilliseconds);

// Hands the tracking every sample released at or before the image about to be processed, and advances
// trace.next. Must be called before grab(): samples ingested afterwards would only be available for the
// next image. A live application would instead call ingestMotionPrior() from the thread reading its
// motion source, as samples arrive.
//
// On the first call it also reports whether the samples and the images share a clock: a trace that ends
// before the first image, or starts long after it, leaves the prior unavailable for the whole run.
void ingestMotionPriorUpTo(sl::Camera& zed, MotionPriorTrace& trace, sl::Timestamp imageTimestamp);

// The replay went back in time (an SVO seek or loop): move the cursor back to the first sample still
// ahead of that image. The tracking has to be restarted too, as it refuses motion older than what it
// already has.
void rewindMotionPrior(MotionPriorTrace& trace, sl::Timestamp imageTimestamp);

// Records how the external motion was used for this image, and prints it whenever that changes.
// Silent while nothing changes, so it can be called on every image. The counts it keeps are what
// printMotionPriorSummary() reports, a status appearing on a few images being easy to miss on screen.
void recordRelativePoseFusionStatus(MotionPriorTrace& trace, sl::RELATIVE_POSE_FUSION_STATUS status);

// How much of the trace reached the tracking and what the tracking did with it, once the run is
// over. Prints nothing if no image ever carried a status.
void printMotionPriorSummary(const MotionPriorTrace& trace);
