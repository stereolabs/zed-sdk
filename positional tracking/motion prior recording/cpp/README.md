# ZED SDK - Motion Prior Recording

This sample records an SVO that carries a **motion prior** next to the images: the motion measured by a
source other than the camera images, which the positional tracking can use through
`Camera::ingestMotionPrior()`. Replaying that SVO with the [Positional Tracking](../../positional%20tracking/cpp)
sample hands each sample to the tracking at the same point it reached the application live, so a run
recorded once can be replayed and compared as often as needed.

The motion source used here is the **rotation of the camera between two images, pre-integrated from the raw
gyroscope of its IMU**, so the sample runs with nothing but a ZED camera. It stands in for your own source
(wheel odometry, an external VIO): only `integrateGyro()` in `main.cpp` changes.

## Getting Started

- Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
- Check the [Documentation](https://docs.stereolabs.com/docs/)

## Build the program

- Build for [Windows](https://docs.stereolabs.com/docs/app-development/cpp/windows/)
- Build for [Linux/Jetson](https://docs.stereolabs.com/docs/app-development/cpp/linux/)

## Run the program

```bash
# Record the camera, stop with Ctrl-C
./ZED_Motion_Prior_Recording recording_with_prior.svo2

# Or re-record an existing SVO2, its own IMU becoming the motion prior
./ZED_Motion_Prior_Recording recording_with_prior.svo2 recording.svo2

# Replay it: the Positional Tracking sample uses the motion prior on its own
./ZED_Positional_Tracking --svo recording_with_prior.svo2
```

The camera needs an IMU: every model but the first-generation ZED.

## How it works

```cpp
zed.open(initParameters);
zed.enableRecording(recordingParameters);
while (recording) {
    zed.read();                                   // the next image, not processed yet
    integrateGyro(zed, gyro, sample);             // the rotation since the previous image
    channel.writeSample(zed, sample, ...);        // into the SVO, with sl::SVOData
    zed.grab();                                   // the image goes into the SVO
}
zed.close();
```

`read()` before `grab()` is what keeps the timing right: in an application running the positional tracking,
the samples are handed to `ingestMotionPrior()` at that point, before the `grab()` that processes the image
they cover. The recording keeps the last image read when each sample arrived, and the replay hands it over
at the same point.

### The gyroscope as a motion source

Right after `read()`, `Camera::getSensorsDataBatch()` returns every IMU measurement taken since the previous
image (about 400 Hz on a ZED 2i, so a dozen or so per image). `integrateGyro()` integrates their calibrated
angular velocity (`SensorsData::IMUData::angular_velocity`, in deg/s) into one rotation per image:

- **Integration**: each step uses the rate averaged over its two ends, composed on the right so it turns
  about the axes the IMU had at the start of the step. The next sample starts at the last measurement of
  this one, so the samples chain without a gap. A hole of more than 100 ms in the measurements restarts the
  interval after it, rather than inventing the motion across it.

- **Axes**: the rotation is along the axes of the IMU, so the sensor transform recorded with it is
  `SensorsConfiguration::camera_imu_transform`, where the IMU sits in the camera. On replay it becomes
  the `RelativePoseData::sensor_transform` of every sample.
- **Measured axes**: rotation only (`RX`, `RY`, `RZ`). A gyroscope says nothing about how far the camera
  moved, so the translation is left to the images.
- **Uncertainty**: 0.01 rad (about half a degree) per axis and per sample. The integrated gyroscope and the images
  usually agree within a quarter of a degree between two images; much lower values get every disagreement rejected.

The tracking already fuses this same IMU (`PositionalTrackingParameters::enable_imu_fusion`), so this
recording demonstrates the recording path rather than improving the tracking.

### Using your own motion source

Replace `integrateGyro()` with a function filling `sl::RelativePoseData` from your source, and adapt
what goes with it:

| What | Where |
| --- | --- |
| where the source sits and where its axes point | the sensor transform given to `writeConfig()` |
| what the source measures | `measured_axes` of each sample, e.g. `TX`, `TY` and `RZ` for planar wheel odometry |
| how much to trust it | `covariance` of each sample |

Timestamps are nanoseconds on the same clock as `getTimestamp(TIME_REFERENCE::IMAGE)`, and each sample
starts where the previous one ended.

### The SVO data channels

Two `sl::SVOData` channels, both JSON, described in `MotionPriorChannel.hpp`:

- `motion_prior_config`, once: the coordinate system and the sensor transform.
- `motion_prior_samples`, one message per sample: the `sl::RelativePoseData` as given to `ingestMotionPrior()`
  (covariance and measured axes included), with the last image read when it arrived.

The format belongs to the samples, not to the ZED SDK API.

## Support

If you need assistance go to our Community site at https://community.stereolabs.com/
