# Tutorial 6: Getting sensor data from ZED Mini and ZED2

This tutorial shows how to use retrieve sensor data from ZED Mini and ZED2. It will loop until 800 data samples are grabbed, printing the updated values on console.<br/>
We assume that you have followed previous tutorials.

### Prerequisites

- Windows 10, Ubuntu LTS
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

As in previous tutorials, we create, configure and open the ZED. In this example, we choose to have a right-handed coordinate system  with Y axis up, since it is the most common system chosen in 3D viewing software.

```
// Create a ZED camera object
Camera zed;

// Set configuration parameters
InitParameters init_params;
init_params.camera_resolution = RESOLUTION::HD720; // Use HD720 video mode (default fps: 60)
init_params.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // Use a right-handed Y-up coordinate system
init_params.coordinate_units = UNIT::METER; // Set units in meters

// Open the camera
ERROR_CODE err = zed.open(init_params);
if (err != ERROR_CODE::SUCCESS) {
    printf( "Error opening the camera: %s\n", toString(err).c_str() );
    exit(-1);
}
```

## Capture data

The sensor data can be retrieved in two ways: synchronized or not synchronized to the image frames<br/>
In this tutorial, we grab 800 not synchronized data, equal to 2 seconds.

```
    int i=0;
    Timestamp first_ts=0,prev_imu_ts=0, prev_baro_ts=0, prev_mag_ts=0;
    SensorsData data;

    while ( i<800 ) {
        // Get Sensor Data not synced with image frames
        if( zed.getSensorsData( data, TIME_REFERENCE::CURRENT ) != ERROR_CODE::SUCCESS ) {
            printf( "Error retrieving Sensor Data\n");
            exit(-1);
        }

        [...]
        
```

## Process data

To be sure that data are updated, since they are not synchronized to camera frames, we must check that the
timestamp as changed from the previous retrieved values. We use the timestamp of the IMU sensor as main value
since it's the sensor that runs at higher frequency. <br/>

```
// Check if Sensor Data are updated
        if( prev_imu_ts==imu_ts ) {
            continue;
        }
        prev_imu_ts = imu_ts;
```

If the data are updated we can extract and use them:

```
        printf("*** Sample #%d\n", i);

        printf(" * Relative timestamp: %g sec\n", static_cast<double>(data.imu.timestamp-first_ts)/1e9 );

        // Filtered orientation quaternion
        printf(" * IMU Orientation: Ox: %.3f, Oy: %.3f, Oz: %.3f, Ow: %.3f\n",
               data.imu.pose.getOrientation().ox, data.imu.pose.getOrientation().oy,
               data.imu.pose.getOrientation().oz, data.imu.pose.getOrientation().ow);
        // Filtered acceleration
        printf(" * IMU Acceleration [m/sec^2]: x: %.3f, y: %.3f, z: %.3f\n",
               data.imu.linear_acceleration.x, data.imu.linear_acceleration.y, data.imu.linear_acceleration.z);
        // Filtered angular velocities
        printf(" * IMU angular velocities [deg/sec]: x: %.3f, y: %.3f, z: %.3f\n",
               data.imu.angular_velocity.x, data.imu.angular_velocity.y, data.imu.angular_velocity.z);
```

If we are using a ZED2 we have more sensor data to be acquired and elaborated:

```
if(cam_model == MODEL::ZED2) {
```

IMU Temperature:
```
    // IMU temperature
    float imu_temp;
    data.temperature.get( SensorsData::TemperatureData::SENSOR_LOCATION::IMU, imu_temp );
    printf(" * IMU temperature: %g C\n", imu_temp );
```

Magnetic fields:
```
    // Check if Magnetometer Data are updated
    Timestamp mag_ts = data.magnetometer.timestamp;
    if( prev_mag_ts!=mag_ts ) {
        prev_mag_ts = mag_ts;

        // Filtered magnetic fields
        printf(" * Magnetic Fields [uT]: x: %.3f, y: %.3f, z: %.3f\n",
               data.magnetometer.magnetic_field_calibrated.x, data.magnetometer.magnetic_field_calibrated.y, data.magnetometer.magnetic_field_calibrated.z);
    }
```

Barometer data:
```
    // Check if Barometer Data are updated
    Timestamp baro_ts = data.barometer.timestamp;
    if( prev_baro_ts!=baro_ts ) {
        prev_baro_ts = baro_ts;

        // Atmospheric pressure
        printf(" * Atmospheric pressure [hPa]: %g\n", data.barometer.pressure);

        // Barometer temperature
        float baro_temp;
        data.temperature.get( SensorsData::TemperatureData::SENSOR_LOCATION::BAROMETER, baro_temp );
        printf(" * Barometer temperature: %g\n", baro_temp);

        // Camera temperatures
        float left_temp,right_temp;
        data.temperature.get( SensorsData::TemperatureData::SENSOR_LOCATION::ONBOARD_LEFT, left_temp );
        data.temperature.get( SensorsData::TemperatureData::SENSOR_LOCATION::ONBOARD_RIGHT, right_temp );
        printf(" * Camera left temperature: %g C\n", left_temp);
        printf(" * Camera right temperature: %g C\n", right_temp);
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


