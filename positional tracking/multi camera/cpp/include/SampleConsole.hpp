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

#include <sl/Sensors.hpp>

#include "RigConfig.hpp"

/**
 * @brief Prints a banner.
 * @param text Text centered in the banner.
 */
void printHeader(const std::string& text);

/**
 * @brief Prints a line prefixed with [Sample].
 * @param message The line.
 * @param errorCode Optional code: an error or warning adds a colored tag.
 * @param showErrorDetail Whether an error appends the SDK explanation of its code.
 */
void print(const std::string& message, std::optional<sl::ERROR_CODE> errorCode = std::nullopt, bool showErrorDetail = true);

/**
 * @brief Prints a line prefixed with [Sample], tagged with a Sensors API code.
 * @param message The line.
 * @param errorCode The code: an error or warning adds a colored tag and the code.
 */
void print(const std::string& message, sl::SENSORS_ERROR_CODE errorCode);

/**
 * @brief Prints the code of every camera that failed in the last Sensors call reporting for several cameras.
 * @param sensors The Sensors instance.
 * @param rig The rig, for the camera labels.
 * @param cameraIds Identifier of each camera of the rig, in rig order.
 * @param afterProcess True for the last process(), false for the last call taking several cameras.
 */
void printCameraErrors(
    sl::Sensors& sensors,
    const RigConfig& rig,
    const std::vector<sl::SensorDeviceIdentifier>& cameraIds,
    bool afterProcess
);

/**
 * @brief Prints each camera of the rig: model, serial number, resolution, frame rate and position on the rig.
 * @param sensors The Sensors instance the cameras are opened in.
 * @param rig The rig, for the camera labels.
 * @param cameraIds Identifier of each camera of the rig, in rig order.
 * @param trackingCamera Index of the camera that drives the tracking.
 */
void printRig(sl::Sensors& sensors, const RigConfig& rig, const std::vector<sl::SensorDeviceIdentifier>& cameraIds, size_t trackingCamera);

/**
 * @brief Prints the tracking parameters in use.
 * @param trackingParameters The parameters the tracking was enabled with.
 */
void printTrackingParameters(const sl::PositionalTrackingSensorsParameters& trackingParameters);

/**
 * @brief Prints the keyboard shortcuts of the viewer.
 */
void printInteraction();
