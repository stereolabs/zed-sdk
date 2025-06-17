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

GNSSReplay::GNSSReplay(std::string file_name, sl::Camera *zed) {
    if (!file_name.empty()) {
        _file_name = file_name;
        initialize_from_json();
    } else if (zed != 0) {
        initialize_from_svov2(zed);
    }
}

GNSSReplay::~GNSSReplay() {
}

void GNSSReplay::initialize_from_json() {
    std::ifstream gnss_file_data;
    gnss_file_data.open(_file_name);
    if (!gnss_file_data.is_open()) {
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

void GNSSReplay::initialize_from_svov2(sl::Camera *zed) {

    auto svo_custom_data_keys = zed->getSVODataKeys();
    std::string gnss_key = "GNSS_json";
    bool found = false;
    for (auto &it : svo_custom_data_keys) {
        if (it.find(gnss_key) != std::string::npos) {
            found = true;
            break;
        }
    }

    std::map<sl::Timestamp, sl::SVOData> data;
    auto status = zed->retrieveSVOData(gnss_key, data); // Get ALL

    /*
     We handle 2 formats:
     * 
     * {
            "coordinates": {
                "latitude": XXX,
                "longitude": XXX,
                "altitude": XXX
            },
            "ts": 1694263390000000,
            "latitude_std": 0.51,
            "longitude_std": 0.51,
            "altitude_std": 0.73,
            "position_covariance": [
                0.2601,
                0,
                0,
                0,
                0.2601,
                0,
                0,
                0,
                0.5328999999999999
            ]
        },
     *********
     * Or
     * this one will be converted to the format above
        {
            "Eph": 0.467,
            "EpochTimeStamp": 1694266998000000,
            "Epv": 0.776,
            "Geopoint": {
                "Altitude": XXX,
                "Latitude": XXX,
                "Longitude": XXX
            },
            "Position": [
                [
                    XXX,
                    XXX,
                    XXX
                ]
            ],
            "Velocity": [
                [
                    -0.63,
                    0.25,
                    0.53
                ]
            ]
        }
     */


    auto tmp_array = json::array();
    for (auto &it : data) {
        try {
            auto gnss_data_point = json::parse(it.second.content.begin(), it.second.content.end());
            auto gnss_data_point_formatted = json::object();

            if (!gnss_data_point["Geopoint"].is_null()) {
                gnss_data_point_formatted["coordinates"] = {
                    {"latitude", gnss_data_point["Geopoint"]["Latitude"]},
                    {"longitude", gnss_data_point["Geopoint"]["Longitude"]},
                    {"altitude", gnss_data_point["Geopoint"]["Altitude"]},
                };
                gnss_data_point_formatted["ts"] = gnss_data_point["EpochTimeStamp"];

                float latitude_std = gnss_data_point["Eph"];
                float longitude_std = gnss_data_point["Eph"];
                float altitude_std = gnss_data_point["Epv"];

                gnss_data_point_formatted["latitude_std"] = latitude_std;
                gnss_data_point_formatted["longitude_std"] = longitude_std;
                gnss_data_point_formatted["altitude_std"] = altitude_std;

                gnss_data_point_formatted["position_covariance"] = json::array({
                    longitude_std * longitude_std, 0, 0, 0, latitude_std * latitude_std, 0, 0, 0, altitude_std * altitude_std
                });

                gnss_data_point_formatted["original_gnss_data"] = gnss_data_point;

            } else if (!gnss_data_point["coordinates"].is_null() && !gnss_data_point["latitude_std"].is_null() && !gnss_data_point["longitude_std"].is_null()) {
                // no conversion
                gnss_data_point_formatted = gnss_data_point;
            }

            tmp_array.push_back(gnss_data_point_formatted);

        } catch (const std::runtime_error &e) {
            std::cerr << "Error while reading GNSS data: " << e.what() << std::endl;
        }
    }
    gnss_data["GNSS"] = tmp_array;

    current_gnss_idx = 0;
    previous_ts = 0;
}

void GNSSReplay::close() {
    gnss_data.clear();
    current_gnss_idx = 0;
}

inline std::string gps_status2str(int status) {
    std::string out;
    switch (status) {
        case 1:
            out = "STATUS_GPS";
            break;
        case 2:
            out = "STATUS_DGPS";
            break;
        case 3:
            out = "STATUS_RTK_FIX";
            break;
        case 4:
            out = "STATUS_RTK_FLT";
            break;
        case 5:
            out = "STATUS_DR";
            break;
        case 6:
            out = "STATUS_GNSSDR";
            break;
        case 7:
            out = "STATUS_TIME";
            break;
        case 8:
            out = "STATUS_SIM";
            break;
        case 9:
            out = "STATUS_PPS_FIX";
            break;
        default:
        case 0:
            out = "STATUS_UNK";
            break;
    };
    return out;
}

inline std::string gps_mode2str(int status) {
    std::string out;
    switch (status) {
        case 1:
            out = "MODE_NO_FIX";
            break;
        case 2:
            out = "MODE_2D";
            break;
        case 3:
            out = "MODE_3D";
            break;
        default:
        case 0:
            out = "MODE_NOT_SEEN";
            break;
    };
    return out;
}

sl::GNSSData getGNSSData(json &gnss_data, int gnss_idx) {
    sl::GNSSData current_gnss_data;
    current_gnss_data.ts = 0;

    // If we are at the end of GNSS data, exit
    if (gnss_idx >= gnss_data["GNSS"].size()) {
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
            ) {
        std::cout << "Null GNSS playback data." << std::endl;
        return current_gnss_data;
    }

    if (!current_gnss_data_json["original_gnss_data"].is_null()) {
        if (!current_gnss_data_json["original_gnss_data"]["fix"].is_null()) {
            if (!current_gnss_data_json["original_gnss_data"]["fix"]["status"].is_null())
                std::cout << std::setprecision(3) << "GNSS info: " << gps_status2str(current_gnss_data_json["original_gnss_data"]["fix"]["status"]) << " " << float(current_gnss_data_json["longitude_std"]) << " " << float(current_gnss_data_json["altitude_std"]) << "\r";
        }
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

    if (current_gnss_data_json.contains("status"))
        current_gnss_data.gnss_status = sl::GNSS_STATUS(current_gnss_data_json["status"].get<int>());

    if (current_gnss_data_json.contains("mode"))
        current_gnss_data.gnss_mode = sl::GNSS_MODE(current_gnss_data_json["mode"].get<int>());

    if (!current_gnss_data_json["original_gnss_data"].is_null())
        if (!current_gnss_data_json["original_gnss_data"]["fix"].is_null())
            if (!current_gnss_data_json["original_gnss_data"]["fix"]["status"].is_null()) {

                // Acquisition comes from GPSD https://gitlab.com/gpsd/gpsd/-/blob/master/include/gps.h#L183-211
                int gpsd_mode = current_gnss_data_json["original_gnss_data"]["fix"]["mode"];
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

                int gpsd_status = current_gnss_data_json["original_gnss_data"]["fix"]["status"];
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
            }

    return current_gnss_data;
}

sl::GNSSData GNSSReplay::getNextGNSSValue(uint64_t current_timestamp) {
    sl::GNSSData current_gnss_data = getGNSSData(gnss_data, current_gnss_idx);

    if (current_gnss_data.ts.data_ns == 0)
        return current_gnss_data;

    if (current_gnss_data.ts.data_ns > current_timestamp) {
        current_gnss_data.ts.data_ns = 0;
        return current_gnss_data;
    }

    sl::GNSSData last_data;
    int step = 1;
    while (1) {
        last_data = current_gnss_data;
        int diff_last = current_timestamp - current_gnss_data.ts.data_ns;
        current_gnss_data = getGNSSData(gnss_data, current_gnss_idx + step++);
        if (current_gnss_data.ts.data_ns == 0) //error / end of file 
            break;

        if (current_gnss_data.ts.data_ns > current_timestamp) {
            if ((current_gnss_data.ts.data_ns - current_timestamp) > diff_last) // keep last
                current_gnss_data = last_data;
            break;
        }
        current_gnss_idx++;
    }

    return current_gnss_data;
}

sl::FUSION_ERROR_CODE GNSSReplay::grab(sl::GNSSData &current_data, uint64_t current_timestamp) {
    current_data.ts.data_ns = 0;

    if (current_timestamp > 0 && (current_timestamp > last_cam_ts))
        current_data = getNextGNSSValue(current_timestamp);

    if (current_data.ts.data_ns == previous_ts)
        current_data.ts.data_ns = 0;

    last_cam_ts = current_timestamp;

    if (current_data.ts.data_ns == 0) // Invalid data
        return sl::FUSION_ERROR_CODE::FAILURE;

    previous_ts = current_data.ts.data_ns;
    return sl::FUSION_ERROR_CODE::SUCCESS;
}