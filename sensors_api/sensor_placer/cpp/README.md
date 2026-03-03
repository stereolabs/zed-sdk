# Sensor Placer

A GUI tool for positioning and orienting multiple ZED cameras and LiDARs relative to each other. It displays live point clouds from all connected sensors in a shared 3D view, letting you interactively adjust each sensor's pose. Once aligned, export the result as a JSON configuration file for the Sensors API.

## Building

Requires **ZED SDK 5.2+**, **CUDA**, **Qt 5** (Widgets + OpenGL), and **OpenGL**.

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
# Auto-detect all connected sensors
./SensorPlacer --auto

# Load an existing configuration
./SensorPlacer sensors_config.json

# Specify an output file (auto-saved on close)
./SensorPlacer sensors_config.json -o sensors_placed.json
```

| Option | Description |
|--------|-------------|
| `sensors_config.json` | Load sensors and poses from a JSON file |
| `-o`, `--output FILE` | Output path (default: `<input>_placed.json`) |
| `--auto` | Auto-detect all connected sensors |

## Multi-Camera Calibration Protocol

Follow these steps to calibrate a multi-sensor rig:

### 1. Physical Setup

Mount all cameras and LiDARs in their final positions. Ensure sufficient overlap between adjacent sensors' fields of view — at least 15% overlap is recommended for reliable alignment.

### 2. Gravity Alignment (Initial Orientation)

For each ZED camera, click **Gravity Align** to apply gravity-based pitch and roll correction from the IMU. This removes tilt and gives each camera a level starting orientation. In most case prefer **Floor Align**.

### 3. Floor Plane Detection

Select one ZED camera as the **reference** (origin). Click **Floor Align** to detect the floor and snap the camera to it. This defines the ground plane and sets the vertical reference for the entire rig.

### 4. Point Cloud Alignment

With the reference camera anchored, adjust each remaining sensor's pose to align its point cloud with the reference:

- Use the **rotation dials** (RX/RY/RZ) to match orientation
- Use the **translation scrollbars** (TX/TY/TZ) to match position
- Focus on overlapping areas — walls, corners, and floor edges are good alignment features
- Toggle **Edges only** mode for ZED cameras to see depth discontinuities more clearly
- Use **Real Colors** mode to visually match texture features

### 5. Iterative Refinement

Cycle through all sensors and fine-tune. Check alignment from multiple viewpoints by rotating the 3D view. A well-calibrated rig will show seamless point clouds in overlap regions with no visible offsets.

### 6. Export

Click **Export** (or use the `-o` flag) to save the final configuration. Use the exported JSON with the Sensors API:

```bash
./SensorsAPISample sensors_placed.json
```

### Tips

- Calibrate in a structured environment (rooms with flat walls and right angles) for easier alignment
- Avoid calibrating in open/featureless spaces
- Align sensors with floor plane before any transformation, then Shift+click the same phsical feature on both point cloud in order to align them. Pick matching points on the 2 sensors in the same order.

## JSON Configuration Format

```json
{
  "zeds": [
    {
      "serial": 12345678,
      "rotation": [0.0, 0.1, 0.0],
      "translation": [0.0, 1.2, -0.5]
    }
  ],
  "lidars": [
    {
      "ip": "192.168.1.100",
      "pose": [1,0,0,0.5, 0,1,0,0, 0,0,1,0.2, 0,0,0,1]
    }
  ]
}
```

Two pose formats are supported:

- **Rotation + Translation**: `"rotation": [rx, ry, rz]` (Rodrigues, radians) + `"translation": [tx, ty, tz]` (meters)
- **4×4 Matrix**: `"pose": [m0..m15]` — 16-element row-major transformation matrix

Both formats are accepted for either device type on import.
