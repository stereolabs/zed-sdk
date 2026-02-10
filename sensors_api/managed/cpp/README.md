# Sensors API Sample

This sample demonstrates how to use the **Sensors API** (`sl::Sensors`) to manage multiple ZED cameras, ZED One cameras, and LiDARs with a unified interface. It features 3D visualization with OpenGL or Rerun, and supports live streaming, file playback/recording, object detection, and body tracking.

## Features

- **Unified Multi-Sensor Management**: Use `sl::Sensors` to manage ZED cameras, ZED One cameras, and LiDARs together
- **Live Streaming**: Auto-detect and stream from all connected devices
- **File Playback**: Play back SVO/SVO2 files (cameras) and OSF files (LiDAR), with optional real-time pacing
- **Multi-File Playback**: Play back multiple recordings simultaneously, synchronized to a common timestamp
- **Recording**: Record to SVO2 (cameras, H.265) and OSF (LiDAR) files during live streaming
- **3D Visualization**: OpenGL viewer with point clouds and camera navigation
- **Rerun Integration**: Optional Rerun viewer for advanced visualization (remote connect or save to file)
- **Object Detection**: 3D bounding box detection on ZED cameras via the Sensors API
- **Body Tracking**: Skeleton tracking on ZED cameras via the Sensors API
- **Reference Frame Modes**: Sensor (raw), Baselink (posed), or World (positional tracking)
- **JSON Configuration**: Configure multiple sensors with poses, detection settings via JSON file
- **Headless Mode**: Run without any display for data processing or recording

## Building

### Prerequisites

- ZED SDK 5.2
- CUDA Toolkit
- OpenCV
- OpenGL, GLUT, GLEW
- (Optional) Rerun SDK for Rerun viewer support

### Build Commands

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)

# With Rerun support:
cmake .. -DWITH_RERUN=ON
make -j$(nproc)
```

## Usage

### Auto-detect Connected Sensors
```bash
./SensorsAPISample
```

### Use JSON Configuration
```bash
./SensorsAPISample sensors_config.json
```

### Play Back Recording Files
```bash
./SensorsAPISample recording.svo2              # Single camera
./SensorsAPISample recording.osf               # Single LiDAR
./SensorsAPISample cam.svo2 lidar.osf          # Multiple files (synchronized)
./SensorsAPISample *.svo2 *.osf --realtime     # Play back at recorded frame rate
```

### Reference Frame Modes
```bash
./SensorsAPISample --baselink     # Apply sensor poses (default)
./SensorsAPISample --world        # Enable positional tracking
./SensorsAPISample --sensor       # Raw sensor data, no transform
```

### Object Detection & Body Tracking
```bash
./SensorsAPISample --object-detection          # Enable object detection on ZED cameras
./SensorsAPISample --body-tracking             # Enable body tracking on ZED cameras
./SensorsAPISample -od -bt                     # Both at once
```

### Use Rerun Viewer
```bash
./SensorsAPISample --rerun                             # Spawn Rerun viewer
./SensorsAPISample --rerun --connect 192.168.1.100:9876  # Connect to remote viewer
./SensorsAPISample --rerun --save recording.rrd          # Save to file
```

### Headless Mode
```bash
./SensorsAPISample --headless
```

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--baselink` | `-b` | Use BASELINK reference frame (default, sensor poses applied) |
| `--world` | `-w` | Use WORLD reference frame (positional tracking enabled) |
| `--sensor` | `-s` | Use SENSOR reference frame (raw data, no transform) |
| `--realtime` | `-rt` | Play back at recorded frame rate instead of as fast as possible |
| `--object-detection` | `-od` | Enable object detection on ZED cameras |
| `--body-tracking` | `-bt` | Enable body tracking on ZED cameras |
| `--rerun` | | Use Rerun viewer instead of OpenGL |
| `--connect <host:port>` | | Connect Rerun to a remote viewer |
| `--save <file.rrd>` | | Save Rerun data to file for offline viewing |
| `--headless` | | Run without display (data processing only) |
| `--help` | `-h` | Show help message |

## Controls (OpenGL Viewer)

| Key | Action |
|-----|--------|
| **R** | Toggle recording (SVO2 for cameras, OSF for LiDAR) |
| **W/A/S/D** | Move camera in 3D view |
| **Space** | Move up |
| **C** | Move down |
| **Mouse left-drag** | Rotate view |
| **Mouse right-drag** | Pan view |
| **Scroll** | Zoom in/out |
| **Q/ESC** | Quit |

## JSON Configuration Format

The JSON configuration allows you to specify multiple sensors with their settings and poses:

```json
{
  "zeds": [
    {
      "serial": 12345678,
      "pose": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
      "object_detection": true,
      "body_tracking": false
    },
    {
      "svo": "recording.svo2"
    }
  ],
  "lidars": [
    {
      "ip": "192.168.1.100",
      "pose": [1,0,0,1000, 0,1,0,0, 0,0,1,500, 0,0,0,1]
    },
    {
      "osf": "lidar_recording.osf"
    }
  ],
  "object_detection": {
    "enabled": true,
    "confidence": 50,
    "model": "accurate"
  },
  "body_tracking": {
    "enabled": true,
    "confidence": 50,
    "model": "accurate",
    "format": "38"
  }
}
```

### Configuration Fields

#### ZED Cameras (`zeds` array)
| Field | Type | Description |
|-------|------|-------------|
| `serial` | int | Camera serial number (for live streaming) |
| `svo` | string | Path to SVO/SVO2 file (for playback) |
| `pose` | [float x16] | 4x4 transformation matrix (row-major) |
| `rotation` | [float x3] | Rodrigues rotation vector (alternative to `pose`) |
| `translation` | [float x3] | Translation vector in mm (alternative to `pose`) |
| `object_detection` | bool | Enable object detection for this camera |
| `body_tracking` | bool | Enable body tracking for this camera |

#### LiDARs (`lidars` array)
| Field | Type | Description |
|-------|------|-------------|
| `ip` | string | LiDAR IP address (for live streaming) |
| `osf` | string | Path to OSF file (for playback) |
| `svo` | string | Path to SVO file (for playback, alternative to `osf`) |
| `pose` | [float x16] | 4x4 transformation matrix (row-major) |
| `rotation` | [float x3] | Rodrigues rotation vector (alternative to `pose`) |
| `translation` | [float x3] | Translation vector in mm (alternative to `pose`) |

#### Global Detection Settings
| Field | Type | Description |
|-------|------|-------------|
| `object_detection.enabled` | bool | Enable object detection globally |
| `object_detection.confidence` | int | Detection confidence threshold (0–100, default: 50) |
| `object_detection.model` | string | `"accurate"`, `"medium"`, or `"fast"` |
| `body_tracking.enabled` | bool | Enable body tracking globally |
| `body_tracking.confidence` | int | Tracking confidence threshold (0–100, default: 50) |
| `body_tracking.model` | string | `"accurate"`, `"medium"`, or `"fast"` |
| `body_tracking.format` | string | `"18"`, `"34"`, or `"38"` (default: `"38"`) |

## Recording

Press **R** during runtime to start/stop recording:
- ZED cameras → `.svo2` files (H.265 compressed)
- ZED One cameras → `.svo2` files (H.265 compressed)
- LiDARs → `.osf` files (Ouster Sensor Format)

Recording files are saved in the current directory with timestamps:
- `zed_<serial>_<timestamp>.svo2`
- `zedone_<serial>_<timestamp>.svo2`
- `lidar_<name>_<timestamp>.osf`

## API Overview

The sample uses the `sl::Sensors` class to manage multiple sensors with a unified interface:

```cpp
sl::Sensors sensors;

// Initialize the Sensors manager
sl::InitSensorsParameters init_params;
init_params.coordinate_units = sl::UNIT::METER;
init_params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
sensors.init(init_params);

// Add a ZED camera
sl::SensorDeviceIdentifier zed_id;
sl::InitParameters zedParams;
zedParams.input.setFromSerialNumber(serial);
zedParams.depth_mode = sl::DEPTH_MODE::NEURAL;
sensors.add(zedParams, zed_id);
sensors.setSensorPose(pose, zed_id);

// Add a LiDAR
sl::SensorDeviceIdentifier lidar_id;
sl::InitLidarParameters lidarParams;
lidarParams.input.setFromStream("192.168.1.100");
sensors.add(lidarParams, lidar_id);

// Configure reference frame
sl::RuntimeSensorsParameters rt_params;
rt_params.reference_frame = sl::SENSORS_REFERENCE_FRAME::BASELINK;
sensors.setRuntimeSensorsParameters(rt_params);

// Synchronize playback files (if using recorded files)
sensors.syncSVO();

// Main loop: process and retrieve batched data
sl::BatchedData<sl::Mat> batched_measures, batched_images;
while (running) {
    if (sensors.process() == sl::SENSORS_ERROR_CODE::SUCCESS) {
        sensors.retrieveMeasure(batched_measures, sl::MEASURE::XYZRGBA);
        sensors.retrieveImage(batched_images, sl::VIEW::LEFT);
        // Access per-device data via batched_measures[device_id], batched_images[device_id]
    }
}
```

## Troubleshooting

### No devices found
- Ensure ZED cameras and LiDARs are properly connected
- Check USB permissions on Linux: `sudo chmod 666 /dev/ttyUSB*`
- For LiDARs, ensure network connectivity and correct IP configuration

### OpenGL errors
- Update graphics drivers
- Ensure GLEW is properly initialized before GL calls

### CUDA errors
- Verify CUDA toolkit version matches ZED SDK requirements
- Check GPU memory availability
