#include "gnss_reader/GPSDReader.hpp"

GPSDReader::GPSDReader() {

}

GPSDReader::~GPSDReader() {
    continue_to_grab = false;
    grab_gnss_data.join();
#ifdef GPSD_FOUND

#else
    std::cerr << "[library not found] GPSD library was not found ... please install it before using this sample" << std::endl;
#endif
}

void GPSDReader::initialize() {
    grab_gnss_data = std::thread(&GPSDReader::grabGNSSData, this);
#ifdef GPSD_FOUND
    gnss_getter.reset(new gpsmm("localhost", DEFAULT_GPSD_PORT));
    if (gnss_getter->stream(WATCH_ENABLE | WATCH_JSON) == nullptr) {
        std::cerr << "No GPSD running .. exit" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Successfully opened GPS stream" << std::endl;
    std::cout << "Waiting for GNSS fix" << std::endl;

    bool received_fix = false;
    struct gps_data_t *gpsd_data;
    while (!received_fix) {
        if (!gnss_getter->waiting(0))
            continue;
        if ((gpsd_data = gnss_getter->read()) == NULL) {
            std::cerr << "[GNSS] read error ... exit program" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (gpsd_data->fix.mode >= MODE_2D)
            received_fix = true;
    }
    std::cout << "Fix found !!!" << std::endl;
    is_initialized_mtx.lock();
    is_initialized = true;
    is_initialized_mtx.unlock();
#else
    std::cerr << "[library not found] GPSD library was not found ... please install it before using this sample" << std::endl;
#endif
}

sl::GNSSData GPSDReader::getNextGNSSValue() {
#ifdef GPSD_FOUND
    // 0. Check if GNSS is initialized:
    // 1. Get GNSS datas:
    struct gps_data_t *gpsd_data;
    while ((gpsd_data = gnss_getter->read()) == NULL)
        ;
    if (gpsd_data->fix.mode >= MODE_2D) {
        int nb_low_snr = 0;
        for (int i = 0; i < gpsd_data->satellites_visible; i++) {
            satellite_t &satellite = gpsd_data->skyview[i];
            if (satellite.used && satellite.ss < 16) nb_low_snr++;
        }
        if (nb_low_snr > 0) std::cout << "[Warning] Low SNR (<16) on " << nb_low_snr << " satellite(s) (using " << gpsd_data->satellites_used << " out of " << gpsd_data->satellites_visible << " visible)" << std::endl;

        sl::GNSSData current_gnss_data;
        // Fill out coordinates:
        current_gnss_data.setCoordinates(gpsd_data->fix.latitude, gpsd_data->fix.longitude, gpsd_data->fix.altMSL, false);
        // Fill out default standard deviation:
        current_gnss_data.longitude_std = current_gnss_data.latitude_std = 0.001f;
        current_gnss_data.altitude_std = 1.f;
        // Fill out covariance [must be not null]
        std::array<double, 9> position_covariance;
        position_covariance[0] = gpsd_data->fix.eph * gpsd_data->fix.eph;
        position_covariance[1 * 3 + 1] = gpsd_data->fix.eph * gpsd_data->fix.eph;
        position_covariance[2 * 3 + 2] = gpsd_data->fix.epv * gpsd_data->fix.epv;
        current_gnss_data.position_covariance = position_covariance;
        // Compute timestamp
        uint64_t current_ts_gps = gpsd_data->fix.time.tv_sec * 1000000;
        uint64_t current_tns_gps = gpsd_data->fix.time.tv_nsec / float(1000);
        auto current_gnss_timestamp = current_ts_gps + current_tns_gps;
        current_gnss_data.ts.setMicroseconds(current_gnss_timestamp);

        int gpsd_mode = gpsd_data->fix.mode;
        sl::GNSS_MODE sl_mode = sl::GNSS_MODE::UNKNOWN;

        switch (gpsd_mode) {
            case 0: // MODE_NOT_SEEN
                sl_mode = sl::GNSS_MODE::UNKNOWN;
                break;
            case 1: // MODE_NO_FIX
                sl_mode = sl::GNSS_MODE::NO_FIX;
                break;
            case 2: // MODE_2D
                sl_mode = sl::GNSS_MODE::FIX_2D;
                break;
            case 3: // MODE_3D
                sl_mode = sl::GNSS_MODE::FIX_3D;
                break;
            default:
                sl_mode = sl::GNSS_MODE::UNKNOWN;
                break;
        }

        int gpsd_status = gpsd_data->fix.status;
        sl::GNSS_STATUS sl_status = sl::GNSS_STATUS::UNKNOWN;

        switch (gpsd_status) {
            case 0: // STATUS_UNK
                sl_status = sl::GNSS_STATUS::UNKNOWN;
                break;
            case 1: // STATUS_GPS
                sl_status = sl::GNSS_STATUS::SINGLE;
                break;
            case 2: // STATUS_DGPS
                sl_status = sl::GNSS_STATUS::DGNSS;
                break;
            case 3: // STATUS_RTK_FIX
                sl_status = sl::GNSS_STATUS::RTK_FIX;
                break;
            case 4: // STATUS_RTK_FLT
                sl_status = sl::GNSS_STATUS::RTK_FLOAT;
                break;
            case 5: // STATUS_DR
                sl_status = sl::GNSS_STATUS::SINGLE;
                break;
            case 6: // STATUS_GNSSDR
                sl_status = sl::GNSS_STATUS::DGNSS;
                break;
            case 7: // STATUS_TIME
                sl_status = sl::GNSS_STATUS::UNKNOWN;
                break;
            case 8: // STATUS_SIM
                sl_status = sl::GNSS_STATUS::UNKNOWN;
                break;
            case 9: // STATUS_PPS_FIX
                sl_status = sl::GNSS_STATUS::SINGLE;
                break;
            default:
                sl_status = sl::GNSS_STATUS::UNKNOWN;
                break;
        }

        current_gnss_data.gnss_status = sl_status;
        current_gnss_data.gnss_mode = sl_mode;

        return current_gnss_data;
    } else {
        std::cout << "Fix lost: reinit GNSS" << std::endl;
        initialize();
        return getNextGNSSValue();
    }
#else
    std::cerr << "[library not found] GPSD library was not found ... please install it before using this sample" << std::endl;
#endif

    return sl::GNSSData();
}

sl::ERROR_CODE GPSDReader::grab(sl::GNSSData & current_data) {
    if (new_data) {
        new_data = false;
        current_data = current_gnss_data;
        return sl::ERROR_CODE::SUCCESS;
    }
    return sl::ERROR_CODE::FAILURE;
}

void GPSDReader::grabGNSSData() {
    while (1) {
        is_initialized_mtx.lock();
        if (is_initialized) {
            is_initialized_mtx.unlock();
            break;
        }
        is_initialized_mtx.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    while (continue_to_grab) {
#ifdef GPSD_FOUND
        current_gnss_data = getNextGNSSValue();
        new_data = true;
#endif
    }

}