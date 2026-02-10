# Sensor Placer

A Qt-based GUI tool for visually positioning and orienting multiple ZED cameras and LiDARs relative to each other. It displays live point clouds from all connected sensors in a shared 3D view, letting you interactively adjust each sensor's pose (rotation & translation) using dials, sliders, and spin boxes. Once satisfied, export the result as a JSON configuration file compatible with the Sensors API managed samples.

## Features

- **Interactive 3D Placement**: Adjust each sensor's rotation (Euler angles via dials/spin boxes) and translation (scrollbars/spin boxes) while seeing live point clouds update in real time
- **ZED Camera Support**: Open cameras by serial number or play back SVO/SVO2 files
- **LiDAR Support**: Connect to LiDARs by IP address or play back OSF files
- **Auto-Detect**: Discover and open all connected ZED cameras and LiDARs automatically
- **Floor Plane Detection**: Align a ZED camera's pose to the detected floor plane with one click
- **IMU Gravity Alignment**: Apply pitch/roll correction from IMU gravity data
- **JSON Config I/O**: Load a `sensors_config.json` to pre-populate devices and poses, then export the adjusted poses back to JSON
- **Per-Sensor Coloring**: Each sensor gets a distinct color in the 3D view for easy identification
- **Rendering Options**: Toggle edge-only mode, colored/monochrome point clouds, and adjust point size
- **Hide/Remove Sensors**: Temporarily hide or permanently remove sensors from the scene

## Building

### Prerequisites

- ZED SDK 5.x
- CUDA Toolkit
- Qt 5 (Widgets + OpenGL modules)
- OpenGL

### Build Commands

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Launch with Auto-Detect
```bash
./SensorPlacer --auto
```

### Load an Existing Configuration
```bash
./SensorPlacer sensors_config.json
```

### Specify an Output File
```bash
./SensorPlacer sensors_config.json -o sensors_placed.json
```

When `-o` is provided, the configuration is automatically saved to that path when the window is closed.

### Command-Line Options

| Option | Description |
|--------|-------------|
| `sensors_config.json` | Load sensors and poses from a JSON configuration file |
| `-o`, `--output FILE` | Output file path for the placed config (default: `<input>_placed.json`) |
| `--auto` | Auto-detect all connected sensors on startup |
| `--help`, `-h` | Show help message |

## GUI Controls

### Sensor Management
- **Serial** field + **Add** button: Open a ZED camera by serial number
- **IP** field + **Add** button: Open a LiDAR by IP address
- **SVO/OSF** field + **Add** button: Open a recorded file (SVO/SVO2 for ZED, OSF for LiDAR)
- **Auto-Detect** button: Discover and add all connected sensors
- **Remove** button: Remove the selected sensor
- **Hide** checkbox: Toggle visibility of the selected sensor

### Pose Adjustment
- **Rotation dials** (X/Y/Z): Rotate the selected sensor using dial widgets
- **Rotation spin boxes** (RX/RY/RZ): Set rotation values numerically (degrees)
- **Translation scrollbars** (TX/TY/TZ): Translate the selected sensor using scrollbars
- **Translation spin boxes** (TX/TY/TZ): Set translation values numerically (meters)

### Assisted Placement
- **Find Plane** button: Detect the floor plane and align the selected ZED camera to it (ZED only)
- **IMU Pose** button: Apply gravity-based pitch/roll correction from the IMU (ZED only)

### 3D View
- **Mouse left-drag**: Rotate the view
- **Mouse right-drag**: Pan the view
- **Scroll**: Zoom in/out
- **Point size slider**: Adjust point cloud rendering size
- **Edges** checkbox: Toggle edge-only depth rendering (ZED only)
- **Colors** checkbox: Toggle colored vs. monochrome point clouds

### File Operations
- **Load** button: Open a JSON configuration file via file dialog
- **Export** button: Save the current sensor layout to a JSON file via file dialog

## JSON Configuration Format

The tool reads and writes the same JSON format used by the [Sensors API samples](../../managed/cpp/README.md):

```json
{
  "zeds": [
    {
      "serial": 12345678,
      "rotation": [0.0, 0.1, 0.0],
      "translation": [0.0, 1.2, -0.5]
    },
    {
      "svo": "recording.svo2"
    }
  ],
  "lidars": [
    {
      "ip": "192.168.1.100",
      "pose": [1,0,0,0.5, 0,1,0,0, 0,0,1,0.2, 0,0,0,1]
    },
    {
      "osf": "lidar_recording.osf"
    }
  ]
}
```

### Pose Formats

The tool supports two pose representations:

- **Rotation + Translation** (ZED export default): `"rotation": [rx, ry, rz]` (Rodrigues vector, radians) and `"translation": [tx, ty, tz]` (meters)
- **4×4 Matrix** (LiDAR export default, also accepted for ZEDs): `"pose": [m0..m15]` — 16-element row-major transformation matrix

Both formats are accepted on import for either device type.

## Workflow

1. **Add sensors** — Use auto-detect, serial numbers, IP addresses, or recorded files
2. **Adjust poses** — Use the dials, scrollbars, or spin boxes to align each sensor's point cloud in the shared 3D view. Use "Find Plane" or "IMU Pose" for initial orientation
3. **Export** — Save the configuration with adjusted poses to a JSON file
4. **Use in Sensors API** — Pass the exported JSON to the [managed](../../managed/cpp/README.md) or [threaded](../../threaded/cpp/) sensors API sample:
   ```bash
   ./SensorsAPISample sensors_placed.json
   ```

## See Also

- [ZED SDK Documentation](https://www.stereolabs.com/docs/)
