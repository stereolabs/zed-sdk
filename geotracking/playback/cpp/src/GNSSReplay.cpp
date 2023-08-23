#include "GNSSReplay.hpp"

using json = nlohmann::json;

inline bool is_microseconds(uint64_t timestamp) {
    // Check if the timestamp is in microseconds
    return (1'000'000'000'000'000 <= timestamp && timestamp < 10'000'000'000'000'000ULL);
}

inline bool is_nanoseconds(uint64_t timestamp) {
    // Check if the timestamp is in microseconds
    return (1'000'000'000'000'000'000 <= timestamp && timestamp < 10'000'000'000'000'000'000ULL);
}

GNSSReplay::GNSSReplay(std::string file_name)
{
    _file_name = file_name;
    initialize();
}

GNSSReplay::~GNSSReplay()
{
}

void GNSSReplay::initialize()
{
    std::ifstream gnss_file_data;
    gnss_file_data.open(_file_name);
    if (!gnss_file_data.is_open())
    {
        std::cerr << "Unable to open " << _file_name << std::endl;
        exit(EXIT_FAILURE);
    }
    try {
        gnss_data = json::parse(gnss_file_data);
    } catch (const std::runtime_error &e) {
        std::cerr << "Error while reading GNSS data: " << e.what() << std::endl;
    }
    current_gnss_idx = 0;
    previous_ts = 0;
}

void GNSSReplay::close(){
    gnss_data.clear();
    current_gnss_idx = 0;
}


sl::GNSSData getGNSSData(json &gnss_data, int gnss_idx){
    sl::GNSSData current_gnss_data;
    current_gnss_data.ts = 0;

    // If we are at the end of GNSS data, exit
    if (gnss_idx >= gnss_data["GNSS"].size()){
        std::cout << "Reached the end of the GNSS playback data." << std::endl;
        return current_gnss_data;
    }

    json current_gnss_data_json = gnss_data["GNSS"][gnss_idx];
    // Check inputs:
    if (
    current_gnss_data_json["coordinates"].is_null() 
    || current_gnss_data_json["coordinates"]["latitude"].is_null()
    || current_gnss_data_json["coordinates"]["longitude"].is_null()
    || current_gnss_data_json["coordinates"]["altitude"].is_null()
    || current_gnss_data_json["ts"].is_null()
    )
    {
        std::cout << "Null GNSS playback data." << std::endl; 
        return current_gnss_data;
    }
    
    auto gnss_timestamp = current_gnss_data_json["ts"].get<uint64_t>();
    // Fill out timestamp:
    if (is_microseconds(gnss_timestamp))
        current_gnss_data.ts.setMicroseconds(gnss_timestamp);
    else if (is_nanoseconds(gnss_timestamp))
        current_gnss_data.ts.setNanoseconds(gnss_timestamp);
    else
        std::cerr << "Warning: Invalid timestamp format from GNSS file" << std::endl;

    // Fill out coordinates:
    current_gnss_data.setCoordinates(current_gnss_data_json["coordinates"]["latitude"].get<float>(),
            current_gnss_data_json["coordinates"]["longitude"].get<float>(),
                    current_gnss_data_json["coordinates"]["altitude"].get<float>(),
                            false);

    // Fill out default standard deviation:
    current_gnss_data.longitude_std = current_gnss_data_json["longitude_std"];
    current_gnss_data.latitude_std = current_gnss_data_json["latitude_std"];
    current_gnss_data.altitude_std = current_gnss_data_json["altitude_std"];
    // Fill out covariance [must be not null]
    std::array<double, 9> position_covariance;
    for (unsigned i = 0; i < 9; i++)
        position_covariance[i] = 0.0; // initialize empty covariance

    // set covariance diagonal
    position_covariance[0] = current_gnss_data.longitude_std * current_gnss_data.longitude_std;
    position_covariance[1 * 3 + 1] = current_gnss_data.latitude_std * current_gnss_data.latitude_std;
    position_covariance[2 * 3 + 2] = current_gnss_data.altitude_std * current_gnss_data.altitude_std;
    current_gnss_data.position_covariance = position_covariance;

    return current_gnss_data;
} 

sl::GNSSData GNSSReplay::getNextGNSSValue(uint64_t current_timestamp)
{
    sl::GNSSData current_gnss_data = getGNSSData(gnss_data, current_gnss_idx);

    if(current_gnss_data.ts.data_ns == 0)
        return current_gnss_data;

    if(current_gnss_data.ts.data_ns > current_timestamp){
        current_gnss_data.ts.data_ns = 0;
        return current_gnss_data;
    }

    sl::GNSSData last_data;
    int step = 1;
    while(1){
        last_data = current_gnss_data;
        int diff_last = current_timestamp - current_gnss_data.ts.data_ns;
        current_gnss_data = getGNSSData(gnss_data, current_gnss_idx + step++);        
        if(current_gnss_data.ts.data_ns==0) //error / end of file 
            break;

        if(current_gnss_data.ts.data_ns > current_timestamp){
            if((current_gnss_data.ts.data_ns - current_timestamp) > diff_last) // keep last
                current_gnss_data = last_data;
            break;
        }
        current_gnss_idx++;
    }

    return current_gnss_data;
}

sl::FUSION_ERROR_CODE GNSSReplay::grab(sl::GNSSData &current_data, uint64_t current_timestamp)
{
    current_data.ts.data_ns = 0;

    if(current_timestamp>0 && (current_timestamp > last_cam_ts) )
        current_data = getNextGNSSValue(current_timestamp);

    if(current_data.ts.data_ns == previous_ts)
        current_data.ts.data_ns = 0;

    last_cam_ts = current_timestamp;

    if (current_data.ts.data_ns == 0) // Invalid data
        return sl::FUSION_ERROR_CODE::FAILURE;

    previous_ts = current_data.ts.data_ns;
    return sl::FUSION_ERROR_CODE::SUCCESS;
}