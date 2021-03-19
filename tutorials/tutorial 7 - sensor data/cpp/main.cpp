///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////


#include <sl/Camera.hpp>

using namespace std;
using namespace sl;

// Basic structure to compare timestamps of a sensor. Determines if a specific sensor data has been updated or not.
struct TimestampHandler {

    // Compare the new timestamp to the last valid one. If it is higher, save it as new reference.
    inline bool isNew(Timestamp& ts_curr, Timestamp& ts_ref) {
        bool new_ = ts_curr > ts_ref;
        if (new_) ts_ref = ts_curr;
        return new_;
    }
    // Specific function for IMUData.
    inline bool isNew(SensorsData::IMUData& imu_data) {
        return isNew(imu_data.timestamp, ts_imu);
    }
    // Specific function for MagnetometerData.
    inline bool isNew(SensorsData::MagnetometerData& mag_data) {
        return isNew(mag_data.timestamp, ts_mag);
    }
    // Specific function for BarometerData.
    inline bool isNew(SensorsData::BarometerData& baro_data) {
        return isNew(baro_data.timestamp, ts_baro);
    }

    Timestamp ts_imu = 0, ts_baro = 0, ts_mag = 0; // Initial values
};


// Function to display sensor parameters.
void printSensorConfiguration(SensorParameters& sensor_parameters) {
    if (sensor_parameters.isAvailable) {
        cout << "*****************************" << endl;
        cout << "Sensor Type: " << sensor_parameters.type << endl;
        cout << "Max Rate: "    << sensor_parameters.sampling_rate << SENSORS_UNIT::HERTZ << endl;
        cout << "Range: ["      << sensor_parameters.range << "] " << sensor_parameters.sensor_unit << endl;
        cout << "Resolution: "  << sensor_parameters.resolution << " " << sensor_parameters.sensor_unit << endl;
        if (isfinite(sensor_parameters.noise_density)) cout << "Noise Density: " << sensor_parameters.noise_density <<" "<< sensor_parameters.sensor_unit<<"/√Hz"<<endl;
        if (isfinite(sensor_parameters.random_walk)) cout << "Random Walk: " << sensor_parameters.random_walk <<" "<< sensor_parameters.sensor_unit<<"/s/√Hz"<<endl;
    }
}


int main(int argc, char **argv) {

    // Create a ZED camera object.
    Camera zed;

    // Set configuration parameters.
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::NONE; // No depth computation required here.

    // Open the camera.
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error " << returned_state << ", exit program.\n";
        return EXIT_FAILURE;
    }

    // Check camera model.
    auto info = zed.getCameraInformation();
    MODEL cam_model =info.camera_model;
    if (cam_model == MODEL::ZED) {
        cout << "This tutorial only works with ZED 2 and ZED-M cameras. ZED does not have additional sensors.\n"<<endl;
        return EXIT_FAILURE;
    }

    // Display camera information (model, serial number, firmware versions).
    cout << "Camera Model: " << cam_model << endl;
    cout << "Serial Number: " << info.serial_number << endl;
    cout << "Camera Firmware: " << info.camera_configuration.firmware_version << endl;
    cout << "Sensors Firmware: " << info.sensors_configuration.firmware_version << endl;

    // Display sensors configuration (imu, barometer, magnetometer).
    printSensorConfiguration(info.sensors_configuration.accelerometer_parameters);
    printSensorConfiguration(info.sensors_configuration.gyroscope_parameters);
    printSensorConfiguration(info.sensors_configuration.magnetometer_parameters);
    printSensorConfiguration(info.sensors_configuration.barometer_parameters);

    // Used to store sensors data.
    SensorsData sensors_data;

    // Used to store sensors timestamps and check if new data is available.
    TimestampHandler ts;

    // Retrieve sensors data during 5 seconds.
    auto start_time = std::chrono::high_resolution_clock::now();
    int count = 0;
    double elapse_time = 0;

    while (elapse_time < 5000) {

        // Depending on your camera model, different sensors are available.
        // They do not run at the same rate: therefore, to not miss any new samples we iterate as fast as possible
        // and compare timestamps to determine when a given sensor's data has been updated.
        // NOTE: There is no need to acquire images with grab(). getSensorsData runs in a separate internal capture thread.
        if (zed.getSensorsData(sensors_data, TIME_REFERENCE::CURRENT) == ERROR_CODE::SUCCESS) {

            // Check if a new IMU sample is available. IMU is the sensor with the highest update frequency.
            if (ts.isNew(sensors_data.imu)) {
                cout << "Sample " << count++ << "\n";
                cout << " - IMU:\n";
                cout << " \t Orientation: {" << sensors_data.imu.pose.getOrientation() << "}\n";
                cout << " \t Acceleration: {" << sensors_data.imu.linear_acceleration << "} [m/sec^2]\n";
                cout << " \t Angular Velocitiy: {" << sensors_data.imu.angular_velocity << "} [deg/sec]\n";

                // Check if Magnetometer data has been updated.
                if (ts.isNew(sensors_data.magnetometer))
                    cout << " - Magnetometer\n \t Magnetic Field: {" << sensors_data.magnetometer.magnetic_field_calibrated << "} [uT]\n";

                // Check if Barometer data has been updated.
                if (ts.isNew(sensors_data.barometer))
                    cout << " - Barometer\n \t Atmospheric pressure:" << sensors_data.barometer.pressure << " [hPa]\n";
            }
        }

        // Compute the elapsed time since the beginning of the main loop.
        elapse_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
    }

    // Close camera
    zed.close();
    return EXIT_SUCCESS;
}
