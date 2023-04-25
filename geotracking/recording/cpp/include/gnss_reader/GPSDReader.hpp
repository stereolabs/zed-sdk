#ifndef GPSD_Reader_H
#define GPSD_Reader_H

#include <iostream>
#include <sl/Fusion.hpp>
#ifdef GPSD_FOUND
#include <libgpsmm.h>
#endif
#include "IGNSSReader.h"

/**
 * @brief GPSDReader is a common interface that use GPSD for retrieving GNSS data.
 *
 */
class GPSDReader : public IGNSSReader
{
public:
    GPSDReader();
    ~GPSDReader();
    /**
     * @brief Initialize the GNSS sensor and is waiting for the first GNSS fix.
     *
     */
    virtual void initialize();
    /**
     * @brief read next GNSS measurement. This function block until a GNSS measurement is retrieved.
     *
     * @return sl::GNSSData next GNSS measurement.
     */
    virtual sl::GNSSData getNextGNSSValue();

    virtual sl::ERROR_CODE grab(sl::GNSSData & current_data);

protected:
    void grabGNSSData();
    std::thread grab_gnss_data;
    bool continue_to_grab = true;
    bool new_data=false;
    bool is_initialized=false;
    std::mutex is_initialized_mtx;
    sl::GNSSData current_gnss_data;
#ifdef GPSD_FOUND
    std::unique_ptr<gpsmm> gnss_getter;    
#endif
};

#endif