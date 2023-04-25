#include "gnss_reader/GPSDReader.hpp"

GPSDReader::GPSDReader(){
    
}

GPSDReader::~GPSDReader()
{
    continue_to_grab = false;
    grab_gnss_data.join();
#ifdef GPSD_FOUND
    
#else
    std::cerr << "[library not found] GPSD library was not found ... please install it before using this sample" << std::endl;
#endif
}
void GPSDReader::initialize()
{
    std::cout << "initialize " << std::endl;
    grab_gnss_data = std::thread(&GPSDReader::grabGNSSData, this);
#ifdef GPSD_FOUND
    std::cout << "Create new object" << std::endl;
    gnss_getter.reset(new gpsmm("localhost", DEFAULT_GPSD_PORT));
    if (gnss_getter->stream(WATCH_ENABLE | WATCH_JSON) == nullptr)
    {
        std::cerr << "No GPSD running .. exit" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Successfully opened GPS stream" << std::endl;
    std::cout << "Waiting for GNSS fix" << std::endl;

    bool received_fix = false;
    struct gps_data_t *gpsd_data;
    while (!received_fix)
    {
        if (!gnss_getter->waiting(0))
            continue;
        if ((gpsd_data = gnss_getter->read()) == NULL)
        {
            std::cerr << "[GNSS] read error ... exit program" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (gpsd_data->fix.mode >= MODE_2D)
            received_fix = true;
    }
    is_initialized_mtx.lock();
    is_initialized = true;
    is_initialized_mtx.unlock();
    std::cout << "Fix found !!!" << std::endl;
#else
    std::cerr << "[library not found] GPSD library was not found ... please install it before using this sample" << std::endl;
#endif
}

sl::GNSSData GPSDReader::getNextGNSSValue()
{
#ifdef GPSD_FOUND
    // 0. Check if GNSS is initialized:
    // 1. Get GNSS datas:
    struct gps_data_t *gpsd_data;
    while ((gpsd_data = gnss_getter->read()) == NULL)
        ;
    if (gpsd_data->fix.mode >= MODE_2D)
    {
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


        return current_gnss_data;
    }
    else
    {
        std::cout << "Fix lost: reinit GNSS" << std::endl;
        initialize();
        return getNextGNSSValue();
    }
#else
    std::cerr << "[library not found] GPSD library was not found ... please install it before using this sample" << std::endl;
#endif

    return sl::GNSSData();
}

sl::ERROR_CODE GPSDReader::grab(sl::GNSSData & current_data){
    if(new_data){
        new_data=false;
        current_data = current_gnss_data;
        return sl::ERROR_CODE::SUCCESS;
    }
    return sl::ERROR_CODE::FAILURE;
}

void GPSDReader::grabGNSSData(){
    while(1){
        is_initialized_mtx.lock();
        if(is_initialized){
            is_initialized_mtx.unlock();
            break;
        }
        is_initialized_mtx.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    while (continue_to_grab)
    {
        #ifdef GPSD_FOUND
        current_gnss_data = getNextGNSSValue();
        new_data = true;
        #endif
    }
    
}