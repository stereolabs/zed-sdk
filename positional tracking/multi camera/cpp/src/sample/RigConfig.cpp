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

#include "RigConfig.hpp"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#include <opencv2/core.hpp>

#include "SampleConsole.hpp"

using namespace sl;

namespace {

    /**
     * @brief Prints a rig file error.
     * @param message What is wrong.
     * @return False, so a reader can return it directly.
     */
    bool fail(const std::string& message) {
        print(message, ERROR_CODE::FAILURE, false);
        return false;
    }

    /**
     * @brief Reads a string node.
     * @param node The node to read.
     * @return Its value, or an empty string when the node is missing or not a string.
     */
    std::string readString(const cv::FileNode& node) {
        return node.isString() ? static_cast<std::string>(node) : std::string();
    }

    /**
     * @brief Resolves a path written in the rig file.
     * @param path The path as written.
     * @param rigDirectory Directory of the rig file.
     * @return The path itself when absolute, otherwise the path relative to rigDirectory.
     */
    std::string resolvePath(const std::string& path, const std::filesystem::path& rigDirectory) {
        std::filesystem::path filePath(path);
        return filePath.is_relative() ? (rigDirectory / filePath).string() : path;
    }

    /**
     * @brief Reads the input of an svo camera.
     * @param node The input node.
     * @param rigDirectory Directory of the rig file, which relative paths start from.
     * @param[in,out] camera The camera, labeled, whose input is set.
     * @return False, after printing why, when the path is missing.
     */
    bool readSVOInput(const cv::FileNode& node, const std::filesystem::path& rigDirectory, RigCamera& camera) {
        const std::string path = readString(node["path"]);
        if (path.empty()) {
            return fail("Camera '" + camera.label + "': an svo input needs a path");
        }
        const std::string resolved = resolvePath(path, rigDirectory);
        camera.input.setFromSVOFile(resolved.c_str());
        camera.inputDescription = "SVO " + resolved;
        return true;
    }

    /**
     * @brief Reads the input of a live camera.
     * @param node The input node.
     * @param[in,out] camera The camera whose input is set: the camera with the given serial number, or the first one found.
     */
    void readLiveInput(const cv::FileNode& node, RigCamera& camera) {
        if (node["serial_number"].isInt()) {
            const int serialNumber = static_cast<int>(node["serial_number"]);
            camera.input.setFromSerialNumber(serialNumber);
            camera.inputDescription = "live camera " + std::to_string(serialNumber);
        } else {
            camera.inputDescription = "first live camera found";
        }
    }

    /**
     * @brief Reads the input of a streamed camera.
     * @param node The input node.
     * @param[in,out] camera The camera, labeled, whose input is set.
     * @return False, after printing why, when the ip is missing.
     */
    bool readStreamInput(const cv::FileNode& node, RigCamera& camera) {
        const std::string ip = readString(node["ip"]);
        if (ip.empty()) {
            return fail("Camera '" + camera.label + "': a stream input needs an ip");
        }
        if (node["port"].isInt()) {
            const int port = static_cast<int>(node["port"]);
            camera.input.setFromStream(ip.c_str(), port);
            camera.inputDescription = "stream " + ip + ":" + std::to_string(port);
        } else {
            camera.input.setFromStream(ip.c_str());
            camera.inputDescription = "stream " + ip;
        }
        return true;
    }

    /**
     * @brief Reads the input of a camera, whatever its type.
     * @param node The input node.
     * @param rigDirectory Directory of the rig file, which relative paths start from.
     * @param[in,out] camera The camera, labeled, whose input is set.
     * @return False, after printing why, when the input is invalid.
     */
    bool readInput(const cv::FileNode& node, const std::filesystem::path& rigDirectory, RigCamera& camera) {
        const std::string type = readString(node["type"]);

        if (type == "svo") {
            return readSVOInput(node, rigDirectory, camera);
        }
        if (type == "live") {
            readLiveInput(node, camera);
            return true;
        }
        if (type == "stream") {
            return readStreamInput(node, camera);
        }
        return fail("Camera '" + camera.label + "': input type must be svo, live or stream, not '" + type + "'");
    }

    /**
     * @brief Reads the pose of a camera on the rig.
     * @param node The T_camera_baselink node: 16 values, row by row.
     * @param[in,out] camera The camera, labeled, whose pose is set: identity when the node is missing.
     * @return False, after printing why, when the node is not a 4x4 matrix.
     */
    bool readPlacement(const cv::FileNode& node, RigCamera& camera) {
        if (node.empty() || node.isNone()) {
            camera.cameraToRig.setIdentity();
            return true;
        }

        if (!node.isSeq() || node.size() != 16) {
            return fail("Camera '" + camera.label + "': T_camera_baselink must hold the 16 values of a 4x4 matrix, row by row");
        }

        float values[16];
        for (int i = 0; i < 16; i++) {
            values[i] = static_cast<float>(node[i]);
        }
        camera.cameraToRig = Transform(Matrix4f(values));
        return true;
    }

    /**
     * @brief Parses YAML text with OpenCV, which requires a version directive.
     * @param yaml The YAML text, with or without the directive.
     * @param path Path of the file it comes from, for the error message.
     * @param[out] storage The parsed document.
     * @return False, after printing why, on a syntax error.
     */
    bool parseYAML(std::string yaml, const std::string& path, cv::FileStorage& storage) {
        if (yaml.rfind("%YAML", 0) != 0) {
            yaml = "%YAML:1.0\n" + yaml;
        }

        try {
            storage.open(yaml, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_YAML);
        } catch (const cv::Exception& exception) {
            return fail("Cannot parse the rig file " + path + ": " + exception.msg);
        }
        return true;
    }

    /**
     * @brief Reads the global section of the rig file.
     * @param global The global node, possibly missing.
     * @param[in,out] rig The rig whose capture settings and tracking camera are set.
     * @return False, after printing why, on an invalid resolution.
     */
    bool readGlobal(const cv::FileNode& global, RigConfig& rig) {
        if (global.empty()) {
            return true;
        }

        const std::string resolution = readString(global["resolution"]);
        if (!resolution.empty()) {
            RESOLUTION value;
            // fromString() returns true on failure.
            if (fromString(String(resolution.c_str()), value)) {
                return fail("Invalid resolution in the rig file: " + resolution);
            }
            rig.resolution = value;
        }
        if (global["fps"].isInt()) {
            rig.fps = static_cast<int>(global["fps"]);
        }
        rig.trackingCamera = readString(global["tracking_camera"]);
        return true;
    }

    /**
     * @brief Reads the cameras section of the rig file.
     * @param cameras The cameras node.
     * @param rigDirectory Directory of the rig file, which relative paths start from.
     * @param[in,out] rig The rig whose cameras are filled.
     * @return False, after printing why, when the section is empty, a label is repeated or a camera is invalid.
     */
    bool readCameras(const cv::FileNode& cameras, const std::filesystem::path& rigDirectory, RigConfig& rig) {
        if (!cameras.isSeq() || cameras.size() == 0) {
            return fail("The rig file lists no camera under 'cameras'");
        }

        std::set<std::string> labels;
        for (const cv::FileNode& node : cameras) {
            RigCamera camera;
            camera.label = readString(node["label"]);
            if (camera.label.empty()) {
                camera.label = "camera_" + std::to_string(rig.cameras.size());
            }
            if (!labels.insert(camera.label).second) {
                return fail("Two cameras of the rig file are labeled '" + camera.label + "'");
            }

            if (!readInput(node["input"], rigDirectory, camera) || !readPlacement(node["T_camera_baselink"], camera)) {
                return false;
            }

            rig.cameras.push_back(camera);
        }

        if (!rig.trackingCamera.empty() && labels.count(rig.trackingCamera) == 0) {
            return fail("tracking_camera '" + rig.trackingCamera + "' is not one of the camera labels");
        }
        return true;
    }

} // namespace

bool RigConfig::allFromSVO() const {
    return !cameras.empty() && std::all_of(cameras.begin(), cameras.end(), [](const RigCamera& camera) {
        return camera.input.getType() == InputType::INPUT_TYPE::SVO_FILE;
    });
}

std::optional<size_t> RigConfig::trackingCameraIndex() const {
    if (trackingCamera.empty()) {
        return std::nullopt;
    }

    const auto found = std::find_if(cameras.begin(), cameras.end(), [this](const RigCamera& camera) {
        return camera.label == trackingCamera;
    });
    assert(found != cameras.end() && "loadRigConfig() checks that tracking_camera is a camera label");
    return static_cast<size_t>(std::distance(cameras.begin(), found));
}

std::vector<std::string> RigConfig::labels() const {
    std::vector<std::string> cameraLabels;
    for (const RigCamera& camera : cameras) {
        cameraLabels.push_back(camera.label);
    }
    return cameraLabels;
}

bool loadRigConfig(const std::string& path, RigConfig& rig) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return fail("Cannot open the rig file " + path);
    }

    std::stringstream content;
    content << file.rdbuf();

    cv::FileStorage storage;
    if (!parseYAML(content.str(), path, storage)) {
        return false;
    }

    const std::filesystem::path rigDirectory = std::filesystem::absolute(path).parent_path();
    return readGlobal(storage["global"], rig) && readCameras(storage["cameras"], rigDirectory, rig);
}
