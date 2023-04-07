#include "GNSSSaver.h"
#include "json.hpp"

/**
 * @brief Construct a new GNSSSaver object
 *
 */
GNSSSaver::GNSSSaver()
{
    std::string current_date = getCurrentDatetime();
    this->file_path = "GNSS_" + current_date + ".json";
}

/**
 * @brief Destroy the GNSSSaver object
 *
 */
GNSSSaver::~GNSSSaver()
{
    saveAllData();
}

/**
 * @brief Add the input gnss_data into the exported GNSS json file
 *
 * @param gnss_data gnss data to add
 */
void GNSSSaver::addGNSSData(sl::GNSSData gnss_data)
{
    all_gnss_data.push_back(gnss_data);
}

/**
 * @brief Save all added data into the exported json file
 *
 */
void GNSSSaver::saveAllData()
{
    std::vector<nlohmann::json> all_gnss_measurements;
    for (unsigned i = 0; i < all_gnss_data.size(); i++)
    {
        double latitude, longitude, altitude;
        all_gnss_data[i].getCoordinates(latitude, longitude, altitude, false);
        nlohmann::json gnss_measure;
        gnss_measure["ts"] = all_gnss_data[i].ts.getNanoseconds();
        gnss_measure["coordinates"] = {
            {"latitude", latitude},
            {"longitude", longitude},
            {"altitude", altitude}};
        std::array<double, 9> position_covariance;
        for (unsigned j = 0; j < 9; j++)
        {
            position_covariance[j] = all_gnss_data[i].position_covariance[j];
        }
        gnss_measure["position_covariance"] = position_covariance;
        gnss_measure["longitude_std"] = sqrt(position_covariance[0 * 3 + 0]);
        gnss_measure["latitude_std"] = sqrt(position_covariance[1 * 3 + 1]);
        gnss_measure["altitude_std"] = sqrt(position_covariance[2 * 3 + 2]);
        all_gnss_measurements.push_back(gnss_measure);
    }

    nlohmann::json final_json;
    final_json["GNSS"] = all_gnss_measurements;
    std::ofstream gnss_file(file_path);
    gnss_file << final_json.dump();
    gnss_file.close();
    std::cout << "All GNSS data saved" << std::endl;
}
