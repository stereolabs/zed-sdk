#pragma once

#include <sl/Camera.hpp>

/**
* @brief Compute the start frame of each SVO for playback to be synced
*
* @param svo_files Map camera index to SVO file path
* @return Map camera index to starting SVO frame for synced playback
*/
std::map<int, int> syncDATA(std::map<int, std::string> svo_files) {
    std::map<int, int> output; // map of camera index and frame index of the starting point for each

    // Open all SVO
    std::map<int, std::shared_ptr<sl::Camera>> p_zeds;

    for (auto &it : svo_files) {
        auto p_zed = std::make_shared<sl::Camera>();

        sl::InitParameters init_param;
        init_param.depth_mode = sl::DEPTH_MODE::NONE;
        init_param.camera_disable_self_calib = true;
        init_param.input.setFromSVOFile(it.second.c_str());

        auto error = p_zed->open(init_param);
        if (error == sl::ERROR_CODE::SUCCESS)
            p_zeds.insert(std::make_pair(it.first, p_zed));
        else {
            std::cerr << "Could not open file " << it.second.c_str() << ": " << sl::toString(error) << ". Skipping" << std::endl;
        }
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
