#pragma once

#include <sl/Camera.hpp>

inline bool renderObject(const sl::ObjectData& i, const bool isTrackingON) {
    if (isTrackingON)
        return (i.tracking_state == sl::OBJECT_TRACKING_STATE::OK);
    else
        return (i.tracking_state == sl::OBJECT_TRACKING_STATE::OK || i.tracking_state == sl::OBJECT_TRACKING_STATE::OFF);
}

float const id_colors[5][3] = {
    { 232.0f, 176.0f ,59.0f },
    { 175.0f, 208.0f ,25.0f },
    { 102.0f, 205.0f ,105.0f},
    { 185.0f, 0.0f   ,255.0f},
    { 99.0f, 107.0f  ,252.0f}
};

inline sl::float4 generateColorID_u(int idx) {
    if (idx < 0) return sl::float4(236, 184, 36, 255);
    int color_idx = idx % 5;
    return sl::float4(id_colors[color_idx][0], id_colors[color_idx][1], id_colors[color_idx][2], 255);
}

inline sl::float4 generateColorID_f(int idx) {
    auto clr_u = generateColorID_u(idx);
    return sl::float4(static_cast<float>(clr_u[0]) / 255.f, static_cast<float>(clr_u[1]) / 255.f, static_cast<float>(clr_u[2]) / 255.f, 1.f);
}

float const class_colors[6][3] = {
    { 44.0f, 117.0f, 255.0f}, // PEOPLE
    { 255.0f, 0.0f, 255.0f}, // VEHICLE
    { 0.0f, 0.0f, 255.0f},
    { 0.0f, 255.0f, 255.0f},
    { 0.0f, 255.0f, 0.0f},
    { 255.0f, 255.0f, 255.0f}
};

inline sl::float4 getColorClass(int idx) {
    idx = std::min(5, idx);
    sl::float4 clr(class_colors[idx][0], class_colors[idx][1], class_colors[idx][2], 1.f);
    return clr / 255.f;
}

/**
* @brief Compute the start frame of each SVO for playback to be synced
*
* @param svo_files Map camera index to SVO file path
* @return Map camera index to starting SVO frame for synced playback
*/
static std::map<int, int> syncDATA(std::map<int, std::string> svo_files) {
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