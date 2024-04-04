#ifndef GPSD_Replay_H
#define GPSD_Replay_H

#include <iostream>
#include <sl/Fusion.hpp>

#include "json.hpp"

/**
 * @brief GNSSReplay is a common interface that read GNSS saved data
 */
class GNSSReplay {
public:
    GNSSReplay(std::string file_name, sl::Camera *zed = 0);
    ~GNSSReplay();
    /**
     * @brief Initialize the GNSS sensor and is waiting for the first GNSS fix.
     *
     */
    void initialize_from_json();

    void initialize_from_svov2(sl::Camera *zed);

    void close();


    sl::FUSION_ERROR_CODE grab(sl::GNSSData & current_data, uint64_t current_timestamp);

protected:

    sl::GNSSData getNextGNSSValue(uint64_t current_timestamp);

    std::string _file_name;
    unsigned current_gnss_idx = 0;
    unsigned long long previous_ts = 0;
    unsigned long long last_cam_ts = 0;
    nlohmann::json gnss_data;
};

#endif