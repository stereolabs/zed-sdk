#include <iostream>
#include "json.hpp"
#include <sl/Camera.hpp>

/**
 * @brief Function that serialize sl::SensorData into json
 *
 * @param sensors_data data to serialize
 * @return nlohmann::json serialized data
 */
nlohmann::json SensorsData2Json(sl::SensorsData sensors_data);
/**
 * @brief Function that serialize sl::SensorsData::BarometerData into json
 *
 * @param barometer_data data to serialize
 * @return nlohmann::json serialized data
 */
nlohmann::json BarometerData2Json(sl::SensorsData::BarometerData barometer_data);
/**
 * @brief Function that serialize sl::SensorsData::IMUData into json
 *
 * @param imu_data data to serialize
 * @return nlohmann::json serialized data
 */
nlohmann::json IMUData2Json(sl::SensorsData::IMUData imu_data);
/**
 * @brief Function that serialize sl::SensorsData ::MagnetometerData into json
 *
 * @param magnetometer_data data to serialize
 * @return nlohmann::json serialized data
 */
nlohmann::json MagnetometerData2Json(sl::SensorsData::MagnetometerData magnetometer_data);
/**
 * @brief Function that serialize sl::SensorsData::TemperatureData into json
 *
 * @param temperature_data data to serialize
 * @return nlohmann::json serialized data
 */
nlohmann::json TemperatureData2Json(sl::SensorsData::TemperatureData temperature_data);

int main(int argc, char **argv)
{
    // Open camera:
    sl::Camera zed;
    zed.open();

    // Grab data:
    std::cout << "Start grabbing 4000 IMU data" << std::endl;
    unsigned imu_data_grabbed_number = 0;
    sl::Timestamp old_imu_timestamp;
    std::vector<nlohmann::json> all_sensors_data_serialized;
    while (imu_data_grabbed_number < 4000)
    {
        sl::SensorsData sensor_data;
        if (zed.getSensorsData(sensor_data, sl::TIME_REFERENCE::CURRENT) == sl::ERROR_CODE::SUCCESS)
        {
            // Check if data is new
            if (old_imu_timestamp != sensor_data.imu.timestamp)
            {
                old_imu_timestamp = sensor_data.imu.timestamp;
                imu_data_grabbed_number++;
                nlohmann::json sensors_data_serialized = SensorsData2Json(sensor_data);
                all_sensors_data_serialized.push_back(sensors_data_serialized);
            }
        }
    }

    // Write it into file:
    nlohmann::json final_json;
    final_json["Sensors"] = all_sensors_data_serialized;
    std::ofstream output_file;
    output_file.open("Sensors.json");
    output_file << final_json.dump(4);
    output_file.close();
    std::cout << "Data were saved into ./Sensors.json" << std::endl;
    return 0;
}

nlohmann::json SensorsData2Json(sl::SensorsData sensors_data)
{
    nlohmann::json out;
    out["barometer"] = BarometerData2Json(sensors_data.barometer);
    out["temperature"] = TemperatureData2Json(sensors_data.temperature);
    out["magnetometer"] = MagnetometerData2Json(sensors_data.magnetometer);
    out["imu"] = IMUData2Json(sensors_data.imu);
    out["camera_moving_state"] = sl::toString(sensors_data.camera_moving_state).c_str();
    out["image_sync_trigger"] = sensors_data.image_sync_trigger;
    return out;
}

nlohmann::json BarometerData2Json(sl::SensorsData::BarometerData barometer_data)
{
    nlohmann::json out;
    out["is_available"] = barometer_data.is_available;
    out["timestamp"] = barometer_data.timestamp.getNanoseconds();
    out["pressure"] = barometer_data.pressure;
    out["relative_altitude"] = barometer_data.relative_altitude;
    out["effective_rate"] = barometer_data.effective_rate;
    return out;
}

nlohmann::json IMUData2Json(sl::SensorsData::IMUData imu_data)
{
    nlohmann::json out;
    out["is_available"] = imu_data.is_available;
    out["timestamp"] = imu_data.timestamp.getNanoseconds();
    out["pose"] = nlohmann::json();
    out["pose"]["translation"] = std::array<double, 3>();
    out["pose"]["translation"][0] = imu_data.pose.getTranslation().tx;
    out["pose"]["translation"][1] = imu_data.pose.getTranslation().ty;
    out["pose"]["translation"][2] = imu_data.pose.getTranslation().tz;
    out["pose"]["quaternion"] = std::array<double, 4>();
    out["pose"]["quaternion"][0] = imu_data.pose.getOrientation()[0];
    out["pose"]["quaternion"][1] = imu_data.pose.getOrientation()[1];
    out["pose"]["quaternion"][2] = imu_data.pose.getOrientation()[2];
    out["pose"]["quaternion"][3] = imu_data.pose.getOrientation()[3];
    out["pose_covariance"] = std::array<double, 9>();
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            out["pose_covariance"][i * 3 + j] = imu_data.pose_covariance(i, j);

    out["angular_velocity"] = std::array<double, 3>();
    out["angular_velocity"][0] = imu_data.angular_velocity[0];
    out["angular_velocity"][1] = imu_data.angular_velocity[1];
    out["angular_velocity"][2] = imu_data.angular_velocity[2];

    out["linear_acceleration"] = std::array<double, 3>();
    out["linear_acceleration"][0] = imu_data.linear_acceleration[0];
    out["linear_acceleration"][1] = imu_data.linear_acceleration[1];
    out["linear_acceleration"][2] = imu_data.linear_acceleration[2];

    out["angular_velocity_uncalibrated"] = std::array<double, 3>();
    out["angular_velocity_uncalibrated"][0] = imu_data.angular_velocity_uncalibrated[0];
    out["angular_velocity_uncalibrated"][1] = imu_data.angular_velocity_uncalibrated[1];
    out["angular_velocity_uncalibrated"][2] = imu_data.angular_velocity_uncalibrated[2];

    out["linear_acceleration_uncalibrated"] = std::array<double, 3>();
    out["linear_acceleration_uncalibrated"][0] = imu_data.linear_acceleration_uncalibrated[0];
    out["linear_acceleration_uncalibrated"][1] = imu_data.linear_acceleration_uncalibrated[1];
    out["linear_acceleration_uncalibrated"][2] = imu_data.linear_acceleration_uncalibrated[2];

    out["angular_velocity_covariance"] = std::array<double, 9>();
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            out["angular_velocity_covariance"][i * 3 + j] = imu_data.angular_velocity_covariance(i, j);

    out["linear_acceleration_covariance"] = std::array<double, 9>();
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            out["linear_acceleration_covariance"][i * 3 + j] = imu_data.linear_acceleration_covariance(i, j);

    out["effective_rate"] = imu_data.effective_rate;
    return out;
}

nlohmann::json MagnetometerData2Json(sl::SensorsData::MagnetometerData magnetometer_data)
{
    nlohmann::json out;
    out["is_available"] = magnetometer_data.is_available;
    out["timestamp"] = magnetometer_data.timestamp.getNanoseconds();

    out["magnetic_field_uncalibrated"] = std::array<double, 3>();
    out["magnetic_field_uncalibrated"][0] = magnetometer_data.magnetic_field_uncalibrated[0];
    out["magnetic_field_uncalibrated"][1] = magnetometer_data.magnetic_field_uncalibrated[1];
    out["magnetic_field_uncalibrated"][2] = magnetometer_data.magnetic_field_uncalibrated[2];

    out["magnetic_field_calibrated"] = std::array<double, 3>();
    out["magnetic_field_calibrated"][0] = magnetometer_data.magnetic_field_calibrated[0];
    out["magnetic_field_calibrated"][1] = magnetometer_data.magnetic_field_calibrated[1];
    out["magnetic_field_calibrated"][2] = magnetometer_data.magnetic_field_calibrated[2];

    out["magnetic_heading"] = magnetometer_data.magnetic_heading;
    out["magnetic_heading_state"] = sl::toString(magnetometer_data.magnetic_heading_state).c_str();
    out["magnetic_heading_accuracy"] = magnetometer_data.magnetic_heading_accuracy;
    out["effective_rate"] = magnetometer_data.effective_rate;
    return out;
}

nlohmann::json TemperatureData2Json(sl::SensorsData::TemperatureData temperature_data)
{
    nlohmann::json out;
    for (auto &kv : temperature_data.temperature_map)
    {
        out[sl::toString(kv.first).c_str()] = kv.second;
    }
    return out;
}
