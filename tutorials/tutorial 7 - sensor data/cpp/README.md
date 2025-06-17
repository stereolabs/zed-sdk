# Tutorial 7: Getting sensors data from ZED Mini and ZED2

This tutorial shows how to retrieve sensors data from ZED Mini and ZED2.
Contrary to other samples, this one does not focus on images or depth information but on embedded sensors. It will loop for 5 seconds, printing the retrieved sensors values on console.<br/>
We assume that you have followed previous tutorials.

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

## Build the program

Download the sample and follow the instructions below: [More](https://www.stereolabs.com/docs/getting-started/application-development/)

#### Build for Windows

- Create a "build" folder in the source folder
- Open cmake-gui and select the source and build folders
- Generate the Visual Studio `Win64` solution
- Open the resulting solution and change configuration to `Release`
- Build solution

#### Build for Linux

Open a terminal in the sample directory and execute the following command:

    mkdir build
    cd build
    cmake ..
    make
	
# Code overview
## Create a camera

As in previous tutorials, we create, configure and open the ZED camera, as we do not need depth information we can disable its computation to save process power.

```
    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_parameters;
    // no depth computation required here
    init_parameters.depth_mode = DEPTH_MODE::NONE;

    // Open the camera
    ERROR_CODE err = zed.open(init_parameters);
    if (err != ERROR_CODE::SUCCESS) {
        cout << "Error " << err << ", exit program.\n";
        return -1;
    }
```

## Sensors data capture
Depending on your camera model, different sensors may send informations.
To simplify the retrieve process we have a global class, `SensorsData`, that encapsulates all sensors data.

```
    SensorsData sensors_data;
    double elapse_time = 0;
    while (elapse_time < 5000)
    {

        if (zed.getSensorsData(sensors_data, TIME_REFERENCE::CURRENT) == ERROR_CODE::SUCCESS) 
        {

        [...]

        }
    }        
```

## Process data
As previously said, sensors have different frequencies and they are stored in a global class which means between two `getSensorsData` call, some sensors may not have newer data to provide.
To handle this, each sensor sends the timestamp of its data, by checking if the given timestamp is newer than the previous we know if the data is a new one or not.

In this sample we use a basic class `TimestampHandler` to store timestamp and check for data update.

```
 TimestampHandler ts;
  if (ts.isNew(sensors_data.imu)) {
      // sensors_data.imu contains new data
  }
```
If the data are udpated we display them:
```
    cout << " - IMU:\n";
    // Filtered orientation quaternion
    cout << " \t Orientation: {" << sensors_data.imu.pose.getOrientation() << "}\n";

    // Filtered acceleration
    cout << " \t Acceleration: {" << sensors_data.imu.linear_acceleration << "} [m/sec^2]\n";

    // Filtered angular velocities
    cout << " \t Angular Velocities: {" << sensors_data.imu.angular_velocity << "} [deg/sec]\n";

    // Check if Magnetometer data has been updated 
    if (ts.isNew(sensors_data.magnetometer))
        // Filtered magnetic fields
        cout << " - Magnetometer\n \t Magnetic Field: {" << sensors_data.magnetometer.magnetic_field_calibrated << "} [uT]\n";

    // Check if Barometer data has been updated 
    if (ts.isNew(sensors_data.barometer))
        // Atmospheric pressure
        cout << " - Barometer\n \t Atmospheric pressure:" << sensors_data.barometer.pressure << " [hPa]\n";
```

You do not have to care about your camera model to acces sensors fields, if the sensors is not available its data will contains `NAN` values and its timestamp will be `0`.

Depending on your camera model and firmware, different sensors can send their temperature.
To access it you can iterate over sensors and check if the data is available:

```
    cout << " - Temperature\n";
    float temperature;
    for (int s = 0; s < SensorsData::TemperatureData::SENSOR_LOCATION::LAST; s++) {
        auto sensor_loc = static_cast<SensorsData::TemperatureData::SENSOR_LOCATION>(s);
        if (sensors_data.temperature.get(sensor_loc, temperature) == ERROR_CODE::SUCCESS)
            cout << " \t " << sensor_loc << ": " << temperature << "C\n";
}
```

## Close camera and exit

Once the data are extracted, don't forget to close the camera before exiting the program.<br/>

```
    // Close camera
    zed.close();
    return 0;
```

And this is it!<br/>

You can now get all the sensor data from ZED-M and ZED2 cameras.
