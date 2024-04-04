///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2024, STEREOLABS.
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
#include <sl/Camera.hpp>

// If the current project uses openCV
#if defined (__OPENCV_ALL_HPP__) || defined(OPENCV_ALL_HPP)
// Conversion function between sl::Mat and cv::Mat
cv::Mat slMat2cvMat(sl::Mat &input) {
    int cv_type = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}
#endif

// Compute the starting frame of all data if started out of sync

/*Camera idx / SVO frame*/
std::map<int, int> syncDATA(const std::map<int, std::string> &svo_files) {
    std::map<int, int> output; // map of camera index and frame index of the starting point for each

    // Open all SVO
    std::map<int, std::shared_ptr<sl::Camera>> p_zeds;

    for (auto &it : svo_files) {
        auto p_zed = std::make_shared<sl::Camera>();

        sl::InitParameters init_param;
        init_param.depth_mode = sl::DEPTH_MODE::NONE;
        init_param.camera_disable_self_calib = true;
        init_param.input.setFromSVOFile(it.second.c_str());

        if (p_zed->open(init_param) == sl::ERROR_CODE::SUCCESS)
            p_zeds.insert(std::make_pair(it.first, p_zed));
    }

    // Compute the starting point, we have to take the latest one
    sl::Timestamp start_ts = 0;
    for (auto &it : p_zeds) {
        it.second->grab();
        auto ts = it.second->getTimestamp(sl::TIME_REFERENCE::IMAGE);

        if (ts > start_ts)
            start_ts = ts;
    }

    std::cout << "Found SVOs common starting time: " << start_ts << std::endl;

    // The starting point is now known, let's find the frame idx for all corresponding
    for (auto &it : p_zeds) {
        auto frame_position_at_ts = it.second->getSVOPositionAtTimestamp(start_ts);

        if (frame_position_at_ts != -1)
            output.insert(std::make_pair(it.first, frame_position_at_ts));
    }

    for (auto &it : p_zeds) it.second->close();

    return output;
}
