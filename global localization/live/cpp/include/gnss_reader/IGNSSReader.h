#ifndef I_GNSS_READER_H
#define I_GNSS_READER_H

#include <sl/Fusion.hpp>

/**
 * @brief IGNSSRead is a common interface for reading data from an external GNSS sensor.
 * You can write your own gnss reader that match with your GNSS sensor:
 */
class IGNSSReader{
    public:
        /**
         * @brief Initialize the GNSS sensor and is waiting for the first GNSS fix.
         * 
         */
        virtual void initialize() = 0;
        /**
         * @brief read next GNSS measurement. This function block until a GNSS measurement is retrieved.
         * 
         * @return sl::GNSSData next GNSS measurement.
         */
        virtual sl::GNSSData getNextGNSSValue() = 0;

        virtual sl::ERROR_CODE grab(sl::GNSSData & current_data) = 0;
};

#endif