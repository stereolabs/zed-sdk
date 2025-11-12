#ifndef GNSS_READER_H
#define GNSS_READER_H

#include <sl/Fusion.hpp>

#include "json.hpp"

/**
 * @brief IGNSSRead is a common interface for reading data from an external GNSS sensor.
 * You can write your own gnss reader that match with your GNSS sensor:
 */
class IGNSSReader {
public:
    ~IGNSSReader() { }
    virtual void initialize(sl::Camera* zed, bool read_from_svo) = 0;

    virtual sl::FUSION_ERROR_CODE grab(sl::GNSSData& current_data, uint64_t current_timestamp) = 0;

    static std::shared_ptr<IGNSSReader> create(sl::Camera* zed, bool read_from_svo = false);
};

#ifdef USE_GPSD
    #include <libgpsmm.h>

/**
 * @brief GPSDReader is a common interface that use GPSD for retrieving GNSS data.
 *
 */
class GPSDReader : public IGNSSReader {
public:
    GPSDReader();
    ~GPSDReader();

    void initialize(sl::Camera* zed, bool read_from_svo) override;
    sl::FUSION_ERROR_CODE grab(sl::GNSSData& current_data, uint64_t current_timestamp) override;

protected:
    sl::GNSSData getNextGNSSValue();
    void grabGNSSData();

    std::thread grab_gnss_data;
    bool continue_to_grab = true;
    bool new_data = false;
    bool is_initialized = false;
    std::mutex is_initialized_mtx;
    sl::GNSSData current_gnss_data;
    std::unique_ptr<gpsmm> gnss_getter;
};
#endif // USE_GPSD

/**
 * @brief GNSSReplay is a common interface that read GNSS saved data
 */
class SVOReader : public IGNSSReader {
public:
    SVOReader();
    ~SVOReader();

    void initialize(sl::Camera* zed, bool read_from_svo) override;
    sl::FUSION_ERROR_CODE grab(sl::GNSSData& current_data, uint64_t current_timestamp) override;

private:
    sl::GNSSData getNextGNSSValue(uint64_t current_timestamp);

    std::string _file_name;
    unsigned current_gnss_idx = 0;
    unsigned long long previous_ts = 0;
    unsigned long long last_cam_ts = 0;
    nlohmann::json gnss_data;
};

#endif // GNSS_READER_H
