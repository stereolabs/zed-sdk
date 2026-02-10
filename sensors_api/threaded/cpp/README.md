# Threaded Multi-Sensor Sample

This sample demonstrates how to use the **individual** `sl::Camera` and `sl::Lidar` APIs to manage multiple sensors with **dedicated threads** for each device. Unlike the managed sample that uses the unified `sl::Sensors` wrapper, this sample gives you direct, per-device control over grab loops, recording, and data retrieval.

## Features

- **Per-Device Worker Threads**: Each ZED camera and LiDAR runs in its own thread with independent grab/retrieve loops
- **Lock-Free Data Exchange**: Triple-buffered data path from worker threads to the main rendering thread (zero-copy, no mutex on the hot path)
- **Cross-Device Synchronization** (`--sync`): Optional timestamp-based synchronization that bundles frames from different sensors by matching timestamps within a configurable tolerance window
- **Live Streaming**: Auto-detect and stream from all connected ZED cameras and LiDARs
- **File Playback**: Play back SVO/SVO2 files (cameras) and OSF files (LiDAR)
- **Synchronized Playback Start**: All device threads wait at a barrier until every device is open, ensuring aligned playback start
- **Recording**: Record to SVO2 (cameras, H.265) and OSF (LiDAR) with runtime toggle
- **3D Visualization**: OpenGL viewer with merged point clouds, images, bounding boxes, and skeletons
- **Object Detection**: 3D bounding box detection on ZED cameras
- **Body Tracking**: Skeleton tracking on ZED cameras
- **Reference Frame Modes**: Sensor (raw), Baselink (sensor poses applied), or World (positional tracking)
- **JSON Configuration**: Configure multiple sensors with poses and detection settings
- **Headless Mode**: Run without display for data processing or recording

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  main thread                                                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Main Loop                                             │ │
│  │  • poll triple buffers (default) or syncNext (--sync)  │ │
│  │  • apply baselink/world transforms (GPU)               │ │
│  │  • update GLViewer                                     │ │
│  └──────────┬─────────────────────────────┬───────────────┘ │
│             │                             │                 │
│    TripleBuffer / ingest()       TripleBuffer / ingest()    │
│             │                             │                 │
│  ┌──────────┴──────────┐   ┌──────────────┴──────────────┐  │
│  │  zedWorkerThread    │   │  lidarWorkerThread           │  │
│  │  • sl::Camera       │   │  • sl::Lidar                 │  │
│  │  • grab + retrieve  │   │  • grab + retrieve           │  │
│  │  • OD / BT          │   │  • point cloud + intensity   │  │
│  │  • recording        │   │  • recording                 │  │
│  └─────────────────────┘   └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Default Mode (Triple Buffer)

Each worker thread writes into a lock-free triple buffer. The main thread polls each device's buffer independently, rendering whichever data is newest. This gives the lowest possible latency and works well when you don't need cross-device frame alignment.

### Sync Mode (`--sync`)

When `--sync` is passed, worker threads instead `ingest()` packets into an `AsyncPacketSynchronizer`. The main loop calls `syncNext()` which returns a `Bundle` — a set of packets from different devices whose timestamps fall within a tolerance window (~15 ms). ZED cameras are registered as **Required** streams and LiDARs as **Optional**, so bundles are emitted at the camera frame rate with LiDAR data attached when available.

## Building

### Prerequisites

- ZED SDK 5.x
- CUDA Toolkit
- OpenCV
- OpenGL, GLUT, GLEW

### Build Commands

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Auto-detect Connected Sensors
```bash
./SensorsAPISample
```

### Play Back Recordings
```bash
# Single camera
./SensorsAPISample camera.svo2

# Camera + LiDAR together
./SensorsAPISample camera.svo2 lidar.osf

# Multiple files at recorded frame rate
./SensorsAPISample cam1.svo2 cam2.svo2 lidar.osf --realtime
```

### Synchronized Playback
```bash
# Enable cross-device timestamp synchronization
./SensorsAPISample camera.svo2 lidar.osf --sync
```

### Use JSON Configuration
```bash
./SensorsAPISample sensors_config.json
```

### Enable AI Features
```bash
# Object detection
./SensorsAPISample -od

# Body tracking
./SensorsAPISample -bt

# Both
./SensorsAPISample -od -bt
```

### Reference Frame Modes
```bash
# Baselink (default) — apply sensor extrinsic poses
./SensorsAPISample -b

# World — enable positional tracking, transform to world frame
./SensorsAPISample -w

# Sensor — raw data, no transform applied
./SensorsAPISample -s
```

### Headless Mode
```bash
./SensorsAPISample --headless camera.svo2
```

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--baselink` | `-b` | Use BASELINK reference frame (default, sensor poses applied) |
| `--world` | `-w` | Use WORLD reference frame (enable positional tracking) |
| `--sensor` | `-s` | Use SENSOR reference frame (raw data, no transform) |
| `--realtime` | `-rt` | Play back at recorded frame rate |
| `--object-detection` | `-od` | Enable object detection on ZED cameras |
| `--body-tracking` | `-bt` | Enable body tracking on ZED cameras |
| `--headless` | | Run without display (data processing only) |
| `--sync` | | Enable timestamp-based cross-device synchronization |
| `--combined-layout` | `-cl` | Use combined viewer layout (all sensors in one 3D view) |
| `--help` | `-h` | Show help message |

## Runtime Controls

| Key | Action |
|-----|--------|
| `R` | Toggle recording (SVO2 for cameras, OSF for LiDAR) |
| `W/A/S/D` | Move camera in 3D view |
| Mouse drag | Rotate view |
| `Q` / `ESC` | Quit |

## JSON Configuration

The sample accepts a JSON configuration file to define multiple sensors with poses and detection settings:

```json
{
  "zeds": [
    {"serial": 12345678, "object_detection": true, "body_tracking": false},
    {"svo": "recording.svo2"}
  ],
  "lidars": [
    {"ip": "192.168.1.100"},
    {"osf": "recording.osf", "pose": [1,0,0,1000, 0,1,0,0, 0,0,1,500, 0,0,0,1]}
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

### JSON Fields

**Sensors:**
- `serial` — ZED camera serial number (live)
- `svo` — SVO/SVO2 file path (playback)
- `ip` — LiDAR IP address (live)
- `osf` — OSF file path (LiDAR playback)
- `pose` — 16-element column-major 4×4 transform matrix
- `rotation` — Rodrigues rotation vector `[rx, ry, rz]` (alternative to `pose`)
- `translation` — Translation vector `[tx, ty, tz]` in meters (alternative to `pose`)
- `object_detection` — Per-camera OD enable (boolean)
- `body_tracking` — Per-camera BT enable (boolean)

**Detection settings:**
- `model` — `"accurate"`, `"medium"`, or `"fast"`
- `confidence` — Detection confidence threshold (0–100)
- `format` — Body format: `"18"`, `"34"`, or `"38"`

## Key Differences from Managed Sample

| | Threaded (this sample) | Managed (`sl::Sensors`) |
|---|---|---|
| **API** | `sl::Camera` + `sl::Lidar` directly | `sl::Sensors` unified wrapper |
| **Threading** | Manual per-device threads | Managed internally by the SDK |
| **Data flow** | Triple buffer or async sync | SDK callback / polling |
| **Control** | Full grab loop control per device | High-level API |
| **ZED One** | Not directly supported | Supported via `sl::Sensors` |
| **Use case** | Custom pipelines, low-latency, sync experiments | Quick prototyping, standard workflows |
