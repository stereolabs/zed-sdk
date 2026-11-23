# ZED SDK - Multi Camera Positional Tracking

This sample tracks a rig of several ZED cameras mounted on one rigid body (a robot, a vehicle, a
helmet) with the Sensors API. The cameras are tracked together, as that one body: they feed a single
positional tracking and build a single shared map, so a camera looking backward or sideways keeps the
rig tracked when the front camera sees nothing useful, and a place seen by one camera can be
recognized later by another.

The 3D viewer shows every camera of the rig at its pose, the path of the rig, the map (landmarks and
keyframes, colored by the camera that took them), the tracking status, and the image of any camera
of the rig.

## Getting Started

- Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
- Check the [Documentation](https://docs.stereolabs.com/docs/)

## Build the program

- Build for [Windows](https://docs.stereolabs.com/docs/app-development/cpp/windows/)
- Build for [Linux/Jetson](https://docs.stereolabs.com/docs/app-development/cpp/linux/)

## Run the program

### Describe your rig

The sample needs to know which cameras make up the rig and where each one is mounted on it. Both go
in a YAML file; `rig.yaml` next to this README is a commented example to start from:

```yaml
global:
  resolution: "AUTO"
  fps: 30
  tracking_camera: "front"   # the ZED stereo camera whose IMU is used

cameras:
  - label: "front"
    input: { type: "live", serial_number: 40000001 }

  - label: "rear"
    input: { type: "live", serial_number: 40000002 }
    T_camera_baselink: [
      -1.0, 0.0,  0.0,  0.0,
       0.0, 1.0,  0.0,  0.0,
       0.0, 0.0, -1.0, -0.4,
       0.0, 0.0,  0.0,  1.0 ]
```

- `input` is a live camera (`serial_number`), a recording (`svo`, with a `path` absolute or relative
  to the rig file) or a network stream (`stream`, with an `ip` and an optional `port`).
- `T_camera_baselink` places the camera on the rig: a 4x4 matrix, row by row, taking a point from the
  camera frame into the rig frame. The camera frame is the one of its left sensor (X right, Y down,
  Z forward) and translations are in meters, the convention most calibration tools use. A camera
  without it sits at the origin of the rig frame. The accuracy of the tracking depends on these
  placements: measure them with care.

The rig mixes ZED stereo cameras and ZED One cameras (ZED X One), described the same way: the sample
recognizes a ZED One when it opens it. A ZED stereo camera drives the tracking, so the rig needs at
least one; `tracking_camera` must name one of them, and without it the first ZED stereo camera listed
is used. Each ZED One adds its images to the map and gets its pose from the rig.

All the cameras must run at the same resolution and frame rate, with synchronized timestamps.
Recordings of the same run are aligned in time automatically when the sample starts.

### First run

```bash
./ZED_Positional_Tracking_Multi_Camera rig.yaml
```

A window opens with the rig in 3D and the image of the camera that drives the tracking. Move the rig
around: its path is drawn as you go, every camera follows at its place on the rig, and the tracking
status is displayed in the corner. Press `n` to show the image of the next camera, `esc` to quit, and
see [Keyboard shortcuts](#keyboard-shortcuts) for what you can toggle while it runs.

Recordings play at the speed they were recorded at. The **Pause** button next to the pose card (or
`k`) freezes the replay and the tracking with it, while the 3D view stays live to look around the map.

### How it works

The ZED API flow is in `main.cpp`, condensed here:

```cpp
sensors.init(initSensorsParameters);
for (each camera of the rig) {
    sensors.add(initParameters, cameraId);          // start the camera
    sensors.setSensorPose(cameraToRig, cameraId);   // where it sits on the rig
}
trackingParameters.multi_sensor_mode = MULTI_SENSOR_TRACKING_MODE::SINGLE_BODY;
sensors.enablePositionalTracking(trackingParameters);
while (view.isOpen()) {
    sensors.process();                              // one frame from every camera
    sensors.getPosition(pose, SENSORS_REFERENCE_FRAME::WORLD, cameraId);
}
sensors.close();
```

`MULTI_SENSOR_TRACKING_MODE::SINGLE_BODY` is what makes the cameras one rig. Without it, each camera
runs its own tracking with its own map, unaware of the others.

`main.cpp` also holds the functions that build the `InitParameters` of a camera, its sensor pose and the
`PositionalTrackingSensorsParameters`, the configuration worth reading before you copy anything into
your own application. The cameras compute no depth (`DEPTH_MODE::NONE`): the tracking runs on the
images themselves, and with several cameras that saves a lot of GPU time.

Everything that is not the ZED API lives beside `main.cpp`, so the flow above stays readable:

| File | What it holds |
| --- | --- |
| `src/main.cpp` | the tracking flow and its configuration |
| `RigConfig` | the rig file: the cameras, their inputs and where they sit on the rig |
| `SampleArguments` | command line |
| `SampleConsole` | console output |
| `MapIO` | area files, on disk |
| `TrajectoryRecorder` | trajectory (TUM) and per-frame status (CSV) output |
| `LandmarkOverlay` | drawing the tracked landmarks on the image |
| `SvoPlaybackClock` | replaying the SVOs at the speed they were recorded at |
| `RigView` | the OpenGL viewer |

### Options

```
Usage: ./ZED_Positional_Tracking_Multi_Camera <rig.yaml> [options]

  <rig.yaml>                  The cameras of the rig, their inputs and where they sit on the rig.
                                        See rig.yaml next to this sample.

Options:
  --help                      Shows usage information.
  -i <input_area_file>        Optional. Input area file used in explore mode (default) or --map mode
  --map -o <output_area_file> Optional. Map mode creates or updates an .area file.
                                        Requires -o <output_area_filename> for generated map
  --export-tum [file]         Optional. Export the rig trajectory in TUM format (default: out.tum),
                                        and the trajectory of every camera next to it
                                        (out_<label>.tum)
  --no-gui                    Optional. Run without the viewer, ending on the end of the recordings
                                        instead of on the window being closed. For a batch replay
                                        scored from --export-tum / --status-log.
  --export-keyframes [file]   Optional. Export the final pose of the rig at every keyframe group of
                                        the map in TUM format, written on exit
                                        (default file: keyframes.tum)
  --status-log <file.csv>     Optional. Write the tracking status of every frame to a CSV
                                        (odometry / spatial memory / fusion status and the pose
                                        confidence).
```

### Examples

```bash
# Map an area with the rig and save it
./ZED_Positional_Tracking_Multi_Camera rig.yaml --map -o new_map.area

# Relocalize the rig in a map saved earlier
./ZED_Positional_Tracking_Multi_Camera rig.yaml -i new_map.area

# Score a replay: trajectories and per-frame status to disk, no window
./ZED_Positional_Tracking_Multi_Camera rig_recordings.yaml --no-gui --export-tum --status-log status.csv
```

## Features

### One rig, one pose

The tracking estimates the pose of the rig itself (its BASELINK frame, the origin of the rig file).
Every camera then gets its own pose from `getPosition()`: the pose of the rig moved by where the
camera sits on it. The world frame is the rig frame when the tracking started, aligned with gravity
through the IMU of the camera that drives the tracking.

### Spatial memory and area files

The rig builds a single map, saved to and reloaded from one `.area` file, exactly like a single
camera:

- **Explore** (default): track, and optionally start from a known area with `-i <file>.area`.
- **Map** (`--map -o <file>.area`): same, and export the map on exit. `--map` requires `-o`.

An area file holds the map of the rig that saved it: reload it with the same rig.

### Recording a run

- `--export-tum [file.tum]` writes the trajectory of the rig in TUM format
  (`timestamp tx ty tz qx qy qz qw`), the input most trajectory-error tools expect, and the trajectory
  of every camera next to it, named after its label (`out_front.tum`, `out_rear.tum`, ...).
- `--status-log <file.csv>` writes one row per frame: frame index, timestamp, odometry / spatial
  memory / tracking fusion status, and the pose confidence. Use it to score a replay for robustness
  (tracking losses, relocalizations) rather than only for trajectory error.

Both are flushed per frame, so an interrupted replay still leaves a usable file.

- `--export-keyframes [file.tum]` writes the pose of the rig at every keyframe group of the map (the
  instants at which the cameras of the rig created keyframes together, which the tracking keeps and
  optimizes) in the same TUM format, ordered by timestamp. It is written once when the run ends
  (default `keyframes.tum`), so the poses include every correction the last loop closure applied.

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `space` | toggle camera view visibility |
| `n` | show the image of the next camera of the rig |
| `d` | switch background color from dark to light |
| `l` | enable / disable landmark display |
| `c` | show / hide the rig path |
| `f` | follow the rig |
| `z` | reset the view |
| `k` | pause / resume an SVO replay (same as the Pause button) |
| `ctrl` + drag | rotate |
| `esc` | exit |

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `Failed to enable positional tracking` with `INVALID_FUNCTION_PARAMETERS` | The cameras do not run at the same resolution and frame rate, or a ZED One has no calibration. |
| `tracking_camera '...' is a ZED One` | A ZED One cannot drive the tracking: name a ZED stereo camera, or leave `tracking_camera` out. |
| `The rig needs at least one ZED stereo camera` | Every camera of the rig file is a ZED One: add a ZED stereo camera to drive the tracking. |
| The cameras of the rig point the wrong way in the viewer | A `T_camera_baselink` is inverted (it must take a point from the camera into the rig frame), or written on other axes than X right, Y down, Z forward. |
| `Cannot parse the rig file` | A YAML syntax error. Keep the closing bracket of a matrix on its last row, as in `rig.yaml`. |
| `Option --map requires -o <output_area_file>` | `--map` needs a destination: `--map -o session.area`. |

## Support

If you need assistance go to our Community site at https://community.stereolabs.com/
