# ZED Sensors API Sample - Python

This sample demonstrates how to use the `pyzed.sl.Sensors` API to manage multiple sensors (ZED cameras, ZED One cameras, and LiDAR) with a unified interface.

## Features

- **Multi-sensor streaming**: Manage multiple ZED cameras, ZED One cameras, and LiDAR devices simultaneously
- **Live and playback modes**: Stream from connected devices or replay from recorded files
- **Multi-file playback**: Play back multiple SVO/OSF files synchronized to a common timestamp
- **Recording**: Record data from all sensors to SVO2 (cameras, H.265) and OSF (LiDAR) files
- **Object detection**: 3D bounding box detection on ZED cameras via the Sensors API (`--od`)
- **Real-time visualization**: Display camera images and LiDAR intensity maps using OpenCV
- **3D point cloud viewer**: Optional OpenGL viewer for point clouds (`--opengl`)
- **Headless mode**: Run without display for data processing or recording

## Requirements

- ZED SDK 5.2 and pyzed
- CUDA Toolkit

```bash
pip install opencv-python numpy

# Optional, for 3D point cloud viewer:
pip install PyOpenGL PyOpenGL_accelerate
```

## Usage

### Auto-detect all connected sensors
```bash
python sensors_api_sample.py
```

### Play back a single recording
```bash
python sensors_api_sample.py camera.svo2
python sensors_api_sample.py lidar.osf
```

### Play back multiple recordings (synchronized)
```bash
python sensors_api_sample.py camera1.svo2 camera2.svo2 lidar.osf
```

### Use JSON configuration
```bash
python sensors_api_sample.py config.json
```

### Real-time playback mode
By default, playback reads files as fast as possible. Use `--realtime` to pace playback based on recorded timestamps:
```bash
python sensors_api_sample.py recording.svo2 --realtime
```

### Object detection
```bash
python sensors_api_sample.py --od
```

### 3D point cloud viewer (OpenGL)
```bash
python sensors_api_sample.py --opengl
```

### Headless mode (no display)
```bash
python sensors_api_sample.py --headless
```

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--headless` | | Run without display (data processing only) |
| `--opengl` | `--gl` | Enable 3D point cloud viewer (requires PyOpenGL) |
| `--od` | | Enable object detection on ZED cameras (3D bounding boxes) |
| `--realtime` | `-rt` | Use real-time playback mode (pace by timestamps) |
| `--help` | `-h` | Show help message |

## Controls

During runtime:
- **R**: Toggle recording on/off
- **Q** or **ESC**: Quit the application

## JSON Configuration Format

You can configure multiple sensors with their poses using a JSON file:

```json
{
  "zeds": [
    {
      "serial": 12345678,
      "pose": [1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1]
    },
    {
      "svo": "path/to/recording.svo2"
    }
  ],
  "lidars": [
    {
      "ip": "192.168.1.100"
    },
    {
      "osf": "path/to/recording.osf",
      "pose": [1, 0, 0, 1000,
               0, 1, 0, 0,
               0, 0, 1, 500,
               0, 0, 0, 1]
    }
  ]
}
```

The pose is a 4x4 transformation matrix (row-major) that defines the sensor's position and orientation relative to a common reference frame.

LiDARs also support a `svo` field as an alternative to `osf` for playback files.

## Output Files

When recording is enabled (press **R**), files are automatically generated with timestamps:
- **ZED cameras**: `zed_<serial>_<timestamp>.svo2`
- **ZED One cameras**: `zedone_<serial>_<timestamp>.svo2` *(if using C++ sample; Python does not yet auto-detect ZED One for recording names)*
- **LiDAR**: `lidar_<name>_<timestamp>.osf`

## API Overview

The sample demonstrates the following key Sensors API concepts:

1. **Initialization**: Create a `sl.Sensors()` instance and initialize with `sl.InitSensorsParameters`
2. **Adding devices**: Call `sensors.add()` with `sl.InitParameters` (ZED), `sl.InitParametersOne` (ZED One), or `sl.InitLidarParameters` (LiDAR) to get a `sl.SensorDeviceIdentifier`
3. **Setting poses**: Use `sensors.set_sensor_pose()` to position sensors relative to a common frame
4. **Synchronization**: Use `sensors.sync_svo()` to align playback files to a common timestamp
5. **Processing**: Call `sensors.process()` to update all sensors
6. **Retrieval**: Use `sensors.retrieve_image()` and `sensors.retrieve_measure()` per device ID to get image/point cloud data
7. **Object detection**: Enable with `sensors.enable_object_detection()`, retrieve with `sensors.retrieve_objects()`
8. **Recording**: Enable/disable recording with `sensors.enable_recording()` and `sensors.disable_recording()`

## See Also

- [C++ Sensors API Sample](../cpp/README.md) - Full-featured version with OpenGL/Rerun visualization, body tracking, and reference frame modes
- [ZED SDK Documentation](https://www.stereolabs.com/docs/)
- [Sensors API Reference](https://www.stereolabs.com/docs/api/sensors/)
