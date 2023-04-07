import pyzed.sl as sl
import json


def IMUDataToJSON(imu_data):
    out = {}
    out["is_available"] = imu_data.is_available
    out["timestamp"] = imu_data.timestamp.get_nanoseconds()
    out["pose"] = {}
    pose = sl.Transform()
    imu_data.get_pose(pose)
    out["pose"]["translation"] = [0, 0, 0]
    out["pose"]["translation"][0] = pose.get_translation().get()[0]
    out["pose"]["translation"][1] = pose.get_translation().get()[1]
    out["pose"]["translation"][2] = pose.get_translation().get()[2]
    out["pose"]["quaternion"] = [0, 0, 0, 0]
    out["pose"]["quaternion"][0] = pose.get_orientation().get()[0]
    out["pose"]["quaternion"][1] = pose.get_orientation().get()[1]
    out["pose"]["quaternion"][2] = pose.get_orientation().get()[2]
    out["pose"]["quaternion"][3] = pose.get_orientation().get()[3]
    out["pose_covariance"] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(3):
        for j in range(3):
            out["pose_covariance"][i * 3 + j] = imu_data.get_pose_covariance().r[i][j] 

    out["angular_velocity"] = [0, 0, 0]
    out["angular_velocity"][0] = imu_data.get_angular_velocity()[0]
    out["angular_velocity"][1] = imu_data.get_angular_velocity()[1]
    out["angular_velocity"][2] = imu_data.get_angular_velocity()[2]

    out["linear_acceleration"] = [0, 0, 0]
    out["linear_acceleration"][0] = imu_data.get_linear_acceleration()[0]
    out["linear_acceleration"][1] = imu_data.get_linear_acceleration()[1]
    out["linear_acceleration"][2] = imu_data.get_linear_acceleration()[2]

    out["angular_velocity_uncalibrated"] = [0, 0, 0]
    out["angular_velocity_uncalibrated"][0] = imu_data.get_angular_velocity_uncalibrated()[
        0]
    out["angular_velocity_uncalibrated"][1] = imu_data.get_angular_velocity_uncalibrated()[
        1]
    out["angular_velocity_uncalibrated"][2] = imu_data.get_angular_velocity_uncalibrated()[
        2]

    out["linear_acceleration_uncalibrated"] = [0, 0, 0]
    out["linear_acceleration_uncalibrated"][0] = imu_data.get_linear_acceleration_uncalibrated()[
        0]
    out["linear_acceleration_uncalibrated"][1] = imu_data.get_linear_acceleration_uncalibrated()[
        1]
    out["linear_acceleration_uncalibrated"][2] = imu_data.get_linear_acceleration_uncalibrated()[
        2]

    out["angular_velocity_covariance"] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(3):
        for j in range(3):
            out["angular_velocity_covariance"][i * 3 +j] = imu_data.get_angular_velocity_covariance().r[i][j]

    out["linear_acceleration_covariance"] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(3):
        for j in range(3):
            out["linear_acceleration_covariance"][i * 3 +
                                                  j] = imu_data.get_linear_acceleration_covariance().r[i][j]

    out["effective_rate"] = imu_data.effective_rate
    return out


def BarometerDataToJSON(barometer_data):
    out = {}
    out["is_available"] = barometer_data.is_available
    out["timestamp"] = barometer_data.timestamp.get_nanoseconds()
    out["pressure"] = barometer_data.pressure
    out["relative_altitude"] = barometer_data.relative_altitude
    out["effective_rate"] = barometer_data.effective_rate
    return out


def TemperatureDataToJSON(temperature_data):
    out = {}
    sensors_location = [sl.SENSOR_LOCATION.IMU, sl.SENSOR_LOCATION.BAROMETER,
                        sl.SENSOR_LOCATION.ONBOARD_LEFT, sl.SENSOR_LOCATION.ONBOARD_RIGHT]
    sensors_location_str = ["IMU", "BAROMETER",
                            "ONBOARD_LEFT", "ONBOARD_RIGHT"]
    for sensor_str, sensor in zip(sensors_location_str, sensors_location):
        out[sensor_str] = temperature_data.get(sensor)
    return out


def MagnetometerDataToJSON(magnetometer_data):
    out = {}
    out["is_available"] = magnetometer_data.is_available
    out["timestamp"] = magnetometer_data.timestamp.get_nanoseconds()

    out["magnetic_field_uncalibrated"] = [0, 0, 0]
    out["magnetic_field_uncalibrated"][0] = magnetometer_data.get_magnetic_field_uncalibrated()[
        0]
    out["magnetic_field_uncalibrated"][1] = magnetometer_data.get_magnetic_field_uncalibrated()[
        1]
    out["magnetic_field_uncalibrated"][2] = magnetometer_data.get_magnetic_field_uncalibrated()[
        2]

    out["magnetic_field_calibrated"] = [0, 0, 0]
    out["magnetic_field_calibrated"][0] = magnetometer_data.get_magnetic_field_calibrated()[
        0]
    out["magnetic_field_calibrated"][1] = magnetometer_data.get_magnetic_field_calibrated()[
        1]
    out["magnetic_field_calibrated"][2] = magnetometer_data.get_magnetic_field_calibrated()[
        2]

    out["magnetic_heading"] = magnetometer_data.magnetic_heading
    if(magnetometer_data.magnetic_heading_state == sl.HEADING_STATE.GOOD):
        out["magnetic_heading_state"] ="GOOD"
    if(magnetometer_data.magnetic_heading_state == sl.HEADING_STATE.OK):
        out["magnetic_heading_state"] ="OK"
    if(magnetometer_data.magnetic_heading_state == sl.HEADING_STATE.NOT_GOOD):
        out["magnetic_heading_state"] ="NOT_GOOD"
    if(magnetometer_data.magnetic_heading_state == sl.HEADING_STATE.NOT_CALIBRATED):
        out["magnetic_heading_state"] ="NOT_CALIBRATED"
    if(magnetometer_data.magnetic_heading_state == sl.HEADING_STATE.MAG_NOT_AVAILABLE):
        out["magnetic_heading_state"] ="MAG_NOT_AVAILABLE"
    out["magnetic_heading_accuracy"] = magnetometer_data.magnetic_heading_accuracy
    out["effective_rate"] = magnetometer_data.effective_rate
    return out


def SensorsDataToJSON(sensors_data):
    out = {}
    out["barometer"] = BarometerDataToJSON(sensors_data.get_barometer_data())
    out["temperature"] = TemperatureDataToJSON(
        sensors_data.get_temperature_data())
    out["magnetometer"] = MagnetometerDataToJSON(
        sensors_data.get_magnetometer_data())
    out["imu"] = IMUDataToJSON(sensors_data.get_imu_data())
    if(sensors_data.camera_moving_state == sl.CAMERA_MOTION_STATE.STATIC):
        out["camera_moving_state"] = "STATIC"
    if(sensors_data.camera_moving_state == sl.CAMERA_MOTION_STATE.MOVING):
        out["camera_moving_state"] = "MOVING"
    if(sensors_data.camera_moving_state == sl.CAMERA_MOTION_STATE.FALLING):
        out["camera_moving_state"] = "FALLING"
    out["image_sync_trigger"] = sensors_data.image_sync_trigger
    return out


if (__name__ == "__main__"):
    # Open camera:
    zed = sl.Camera()
    zed.open()

    # Grab sensors values
    all_sensors_data_serialized = []
    imu_data_grabbed_number = 0
    old_imu_timestamp = 0
    print("Start grabbing 4000 IMU data")
    while (imu_data_grabbed_number < 4000):
        sensor_data = sl.SensorsData()
        if (zed.get_sensors_data(sensor_data, sl.TIME_REFERENCE.CURRENT)):
            if (old_imu_timestamp != sensor_data.get_imu_data().timestamp):
                old_imu_timestamp = sensor_data.get_imu_data().timestamp
                imu_data_grabbed_number += 1
                sensors_data_serialized = SensorsDataToJSON(sensor_data)
                all_sensors_data_serialized.append(sensors_data_serialized)
    zed.close()
    final_json = {}
    final_json["Sensors"] = all_sensors_data_serialized
    output_file = open("Sensors.json", 'w')
    json.dump(final_json, output_file, indent=4)
    output_file.close()
    print("Data were saved into ./Sensors.json")
