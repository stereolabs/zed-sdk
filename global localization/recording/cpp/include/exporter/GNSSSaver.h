#ifndef GNSS_SAVER_H
#define GNSS_SAVER_H

#include <string>
#include <vector>
#include <sl/Fusion.hpp>
#include "TimestampUtils.h"

class GNSSSaver {
public:
    /**
     * @brief Construct a new GNSSSaver object
     * Either recorded to a json file or directly in SVO2 custom data
     *
     */
    GNSSSaver(sl::Camera *zed = 0);
    /**
     * @brief Destroy the GNSSSaver object
     *
     */
    ~GNSSSaver();
    /**
     * @brief Add the input gnss_data into the exported GNSS json file
     *
     * @param gnss_data gnss data to add
     */
    void addGNSSData(sl::GNSSData gnss_data);

protected:
    /**
     * @brief Save all added data into the exported json file
     *
     */
    void saveAllData();
    std::string file_path;
    std::vector<sl::GNSSData> all_gnss_data;
    sl::Camera *p_zed = 0;
};

#endif