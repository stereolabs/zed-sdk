# ZED SDK - Positional Tracking

This sample tracks the pose of the camera in a world frame of reference (where it is and which
way it points, image after image) and shows it live in a 3D viewer alongside the point cloud, the
landmarks the tracking relies on, and its current status.

Beyond the basics it demonstrates the features built around tracking: saving and reloading a mapped
area, and feeding the tracking motion measured by an external source such as wheel odometry.

## Getting Started

- Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
- Check the [Documentation](https://docs.stereolabs.com/docs/)

## Build the program

- Build for [Windows](https://docs.stereolabs.com/docs/app-development/cpp/windows/)
- Build for [Linux/Jetson](https://docs.stereolabs.com/docs/app-development/cpp/linux/)

## Run the program

### First run

With a camera connected, no options are needed:

```bash
./ZED_Positional_Tracking
```

A window opens with the camera image and a 3D view. Move the camera around: the trajectory is drawn
as you go, and the tracking status is displayed in the corner. Press `esc` to quit, and see
[Keyboard shortcuts](#keyboard-shortcuts) for what you can toggle while it runs.

No camera at hand? Any recording works:

```bash
./ZED_Positional_Tracking --svo recording.svo2
```

A recording plays at the speed it was recorded at. The **Pause** button next to the pose card (or `k`) freezes
the replay and the tracking with it, while the 3D view stays live to look around the map.

### How it works

The whole ZED API flow is in `main()`, and it is short:

```cpp
zed.open(initParameters);                       // start the camera
zed.enablePositionalTracking(trackingParameters);
while (view.isOpen()) {
    zed.grab(runtimeParameters);                // process one image
    zed.getPosition(pose);                      // where the camera is now
}
zed.close();
```

The real loop calls `zed.read()` before `grab()`, which splits "fetch the next image" from "process
it", which is what leaves room to ingest external motion for an image before the tracking consumes
it (see [Motion prior](#motion-prior-help-from-an-external-source)). If you do not use a motion
prior, `grab()` alone is all you need.

`main()` also holds the two functions that build `InitParameters` and `PositionalTrackingParameters`,
the configuration worth reading before you copy anything into your own application. This sample
runs the tracking in `POSITIONAL_TRACKING_MODE::GEN_3` with `DEPTH_MODE::NEURAL`, area memory and
IMU fusion enabled.

Everything that is not the ZED API lives beside `main.cpp`, so the flow above stays readable:

| File | What it holds |
| --- | --- |
| `src/main.cpp` | the tracking flow and its configuration |
| `SampleArguments` | command line |
| `SampleConsole` | console output |
| `MotionPriorFile` | reading and replaying externally measured motion |
| `MotionPriorChannel` | the same motion, recorded in the SVO next to the images |
| `MapIO` | area files, on disk |
| `TrajectoryRecorder` | trajectory (TUM) and per-image status (CSV) output |
| `LandmarkOverlay` | drawing the tracked landmarks on the image |
| `SvoPlaybackClock` | replaying an SVO at the speed it was recorded at |
| `SLAMView`, `MotionPriorView` | the OpenGL viewer, and what the motion prior did drawn over it |

### Options

```
Usage: ./ZED_Positional_Tracking [options]

Options:
  --help                      Shows usage information.
  --resolution <mode>         Optional. Resolution options: (HD2K | HD1200 | HD1080 | HD720 | SVGA | VGA)
  --svo <filename.svo>        Optional. Use SVO file input. Mutually exclusive with --stream
  --stream <ip[:port]>        Optional. Use network streaming input. Mutually exclusive with --svo
  -i <input_area_file>        Optional. Input area file used in explore mode (default) or --map mode
  --map -o <output_area_file> Optional. Map mode creates or updates an .area file.
                                        Requires -o <output_area_filename> for generated map
  --roi <roi_filepath>        Optional. Region of interest image mask to ignore a static area
  --custom-initial-pose       Optional. Use custom initial pose (see code comments for more detail)
  --2d-ground-mode            Optional. Enable 2D ground mode
  --export-tum                Optional. Export camera trajectory to out.tum file in TUM format
  --export-keyframes [file]   Optional. Export the final pose of every keyframe of the map in TUM
                                        format, written on exit (default file: keyframes.tum)
  --status-log <file.csv>     Optional. Write the tracking status of every image to a CSV
                                        (odometry / spatial memory / fusion / motion prior
                                        status and the pose confidence).
  --voxel                     Optional. Use voxel-decimated point cloud (retrieveVoxelMeasure)
                                        instead of the organized grid (retrieveMeasure).
  --motion-prior <file.json>  Optional. Enable the motion prior and replay the motion measured by
                                        an external source (wheel odometry, external VIO, GNSS)
                                        from a JSON file. See the file format in the sample README.
                                        Requires the GEN_3 positional tracking mode.
                                        An SVO that carries a motion prior uses it on its own;
                                        this file wins over it.
  --no-motion-prior           Optional. Ignore the motion prior recorded in the SVO.
  --motion-prior-delay <ms>   Optional. Hand every motion prior sample to the tracking this
                                        much later than it arrived, to test a late source.
```

### Examples

```bash
# Map an area and save it
./ZED_Positional_Tracking --map -o new_map.area

# Replay a recording, relocalizing in a map saved earlier
./ZED_Positional_Tracking --svo recording.svo2 -i map.area

# Replay a recording with motion from an external source
./ZED_Positional_Tracking --svo recording.svo2 --motion-prior odometry.json

# Replay a recording that carries its own motion prior, then without it, then as if it were late
./ZED_Positional_Tracking --svo recording_with_prior.svo2
./ZED_Positional_Tracking --svo recording_with_prior.svo2 --no-motion-prior
./ZED_Positional_Tracking --svo recording_with_prior.svo2 --motion-prior-delay 250

# Score a replay: trajectory and per-image status to disk
./ZED_Positional_Tracking --svo recording.svo2 --export-tum --status-log status.csv
```

## Features

### Spatial memory and area files

The tracking remembers places it has seen. That memory can be saved to an `.area` file and reloaded
later, which is what lets a new session recognize a known place and give poses in **that area's**
frame instead of starting a fresh one at the origin. The sample has two modes:

- **Explore** (default): track, and optionally start from a known area with `-i <file>.area`.
- **Map** (`--map -o <file>.area`): same, and export the spatial memory on exit. `--map` requires
  `-o`.

Relocalization is not instant: the camera has to see enough of the mapped area again. Watch the
spatial memory status in the viewer: it stays `INITIALIZING` while the camera has not recognized
the loaded area yet, reports `LOOP_CLOSED` at the moment it does, and settles on `KNOWN_MAP` once
localized. `MAP_UPDATE` means it is mapping new ground.

`--roi <mask.png>` masks out part of the image, typically the vehicle or robot the camera is
mounted on. Those pixels move with the camera while looking perfectly static, and without a mask
the tracking may trust them as fixed landmarks.

`--custom-initial-pose` starts the tracking from a hardcoded pose instead of the origin, which is
useful when the camera starts at a known place. See `buildTrackingParameters()` in `main.cpp` for the
matrix and how to change it.

### Motion prior: help from an external source

Images alone struggle in fast motion, on low-texture surfaces, or when a moving object fills the
view. If your platform already measures its own motion (wheel odometry, an external VIO, a
GNSS-derived displacement), the tracking can use it. `--motion-prior <file.json>`, or an SVO
recorded with one (see [below](#recording-the-motion-prior-in-the-svo)), makes the sample push,
just before each `grab()`, every sample that had reached the application by then with
`Camera::ingestMotionPrior()`. There is nothing to enable: with `POSITIONAL_TRACKING_MODE::GEN_3`
the tracking uses the motion prior from the first sample on.

Two things to carry into your own application:

- **Order matters.** Samples must be ingested *before* the `grab()` that processes the image they
  cover; anything later only applies to the next image.
- **The file is a stand-in.** A real application calls `ingestMotionPrior()` from the thread reading
  its own source, as the measurements arrive. The JSON replay exists so the feature can be exercised
  on a recording.

The sample prints the status of the prior whenever it changes, which is the fastest way to tell
whether your samples are actually being used:

| Status | Meaning |
| --- | --- |
| `OK` | the external motion was used for this image |
| `MISSING` | nothing covered this image; occasional values are expected and harmless |
| `OUT_OF_SYNC` | motion was available but did not cover this image's interval; check that the source and the camera share a time reference |
| `REJECTED` | the motion disagreed with the images and was discarded; persistent values usually mean a wrong mounting transform, a wrong sign convention, or an uncertainty set too low |

#### File format

Translations are in the coordinate units selected at init, rotations in radians.

```json
{
  "coordinate_system": "IMAGE",
  "sensor_transform": {
    "translation": [0.0, -0.2, 0.1],
    "rotation_vector": [0.0, 0.0, 0.0]
  },
  "measured_axes": ["TX", "TY", "RZ"],
  "samples": [
    {
      "end_timestamp": 1712345678901234567,
      "translation": [0.02, 0.0, 0.0],
      "rotation_vector": [0.0, 0.0, 0.004],
      "translation_std": [0.01, 0.01, 0.01],
      "rotation_std": [0.001, 0.001, 0.002]
    }
  ]
}
```

| Field | Scope | Meaning |
| --- | --- | --- |
| `coordinate_system` | file, optional | The `sl::COORDINATE_SYSTEM` `sensor_transform` is written in, by its enumerator name (`IMAGE`, `RIGHT_HANDED_Y_UP`, `RIGHT_HANDED_Z_UP_X_FWD`, ...). Default: `IMAGE` (X right, Y down, Z forward). |
| `sensor_transform` | file, optional | Where the source sits on the rig and where its axes point, in the `coordinate_system` above. Passed to the SDK as `RelativePoseData::sensor_transform` of every sample, which is what lets the tracking move the measurements into the camera frame. The SDK reads it in the coordinate system chosen at init (`RIGHT_HANDED_Y_UP` in this sample), so the sample converts it on load; the samples themselves stay along the axes of the source. A file in a coordinate system of the opposite handedness is refused. Omitting it means the samples are already the motion of the camera itself. |
| `measured_axes` | file or sample, optional | Which degrees of freedom the source actually measures; the camera keeps authority over the rest. Planar wheel odometry is `["TX", "TY", "RZ"]`. Default: all six. |
| `end_timestamp` | sample, **required** | Nanoseconds, same reference as `getTimestamp(TIME_REFERENCE::IMAGE)`. |
| `arrival_timestamp` | sample, optional | When the sample reached the application, same clock. The replay hands it over then instead of at `end_timestamp`, so a late source stays late. |
| `start_timestamp` | sample, optional after the first | Start of the interval this sample covers. Defaults to the previous sample, so a chained trace needs no gaps. The first sample has nothing to chain onto and must set it. |
| `translation` / `rotation_vector` | sample | The motion over the interval. `"quaternion": [x, y, z, w]` is accepted in place of `rotation_vector`. |
| `translation_std` / `rotation_std` | sample | Per-axis uncertainty, the shorthand for an uncorrelated source. |
| `covariance` | sample, optional | Full 36-value row-major matrix, ordered `[tx, ty, tz, rx, ry, rz]`. Takes precedence over the `*_std` shorthand. |
| `is_stationary` | sample, optional | Marks the platform as known to be still over the interval. |

On load the sample prints how many samples it read, the time window they span, their interval range,
the total distance they describe and how late they arrived. Check these against your recording
before blaming the tracking.

#### Recording the motion prior in the SVO

A motion prior can travel inside the SVO, next to the images it was used with: the replay then
needs no extra file, and hands each sample to the tracking at the same point it reached the
application live. A sample that arrived late live is late in the replay too, which a file of
measurements alone cannot reproduce.

`MotionPriorChannel.hpp` holds the format and a small recorder. In a live application recording an
SVO (`Camera::enableRecording()`):

```cpp
MotionPriorRecorder recorder;
recorder.writeConfig(zed, sensorTransform, measuredAxes);      // once

// Wherever the samples from your source are handed to the tracking:
zed.ingestMotionPrior(sample);
recorder.write(zed, {sample}, lastGrabbedImageTimestamp);      // same moment
```

It writes two `sl::SVOData` channels, both JSON: `motion_prior_config` once (`coordinate_system`,
the one the camera was opened with, `sensor_transform`, `measured_axes`) and `motion_prior_samples`, one message per arrival with the samples in the file
format above and `last_image_timestamp`, the last image grabbed when they arrived. The replay hands
them over before the image that follows it. The format belongs to this sample, not to the SDK API.

When `--svo` carries the channel, the sample uses it on its own and says so on start;
`--no-motion-prior` ignores it and `--motion-prior <file.json>` takes precedence over it.
`--motion-prior-delay <ms>` holds every sample back by that much on either source, to see what a
late source does to the tracking.

The [Motion Prior Recording](../../motion%20prior%20recording/cpp) sample records such an SVO end to end,
with the camera rotation integrated from its gyroscope as the source; it is the quickest way to get one.

#### What the viewer shows

With a motion prior, the viewer draws what became of it:

- a short arrow per sample, from the pose where its interval starts to where the sample says the
  camera went;
- a straight line per keyframe link reported by `Camera::getPositionalTrackingMotionPriorLinks()`
  (polled once a second), between the two keyframes the motion prior holds together. It starts on the
  from keyframe and ends with an arrowhead on the to keyframe. Click a line to name both ends in the view
  and see the link as the SDK reports it next to the motion the samples it spans recorded;
- a panel on the right: where the samples come from, how each one was used, how late they arrived;
- a timeline along the bottom: the last 2 s of images, keyframes and samples, each sample as a bar
  over the interval it measures with a whisker to the moment it reached the application.

| Color | Meaning |
| --- | --- |
| teal | used for the pose of the image it ends at |
| orange, dashed | used by the map only: it arrived too late for its image |
| red | rejected: it disagreed with what the camera sees |
| gray | unused, or not processed yet |

### Point cloud display

`--voxel` renders the point cloud with `retrieveVoxelMeasure()` instead of `retrieveMeasure()`: the
SDK returns a voxel-decimated, unorganized cloud (one point per occupied voxel) rather than one
point per depth pixel. It is far lighter to display and to pass downstream. The point count varies
from image to image, so the viewer uploads this cloud from CPU memory instead of through CUDA-OpenGL
interop.

### Recording a run

- `--export-tum` writes the trajectory to `out.tum` in TUM format (`timestamp tx ty tz qx qy qz qw`),
  the input most trajectory-error tools expect.
- `--status-log <file.csv>` writes one row per image: frame index, timestamp, odometry / spatial
  memory / tracking fusion / relative pose fusion status, and the pose confidence. Use it to score a replay for
  robustness (tracking losses, relocalizations, how the prior was used) rather than only for
  trajectory error.

Both are flushed per image, so an interrupted replay still leaves a usable file.

- `--export-keyframes [file.tum]` writes the keyframes of the map (the poses the tracking keeps and
  optimizes, not the per-image trajectory) in the same TUM format, ordered by timestamp. It is
  written once when the run ends (default `keyframes.tum`), so the poses include every correction
  the last loop closure applied. Comparing it to `out.tum` shows what the map optimization changed.

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `space` | toggle camera view visibility |
| `d` | switch background color from dark to light |
| `p` | enable / disable the live point cloud |
| `l` | enable / disable landmark display |
| `c` | show / hide the camera path (hidden while the motion prior is shown) |
| `f` | follow the camera |
| `z` | reset the view |
| `m` | show / hide the motion prior arrows and keyframe links; the camera path does the opposite |
| `t` | show / hide the motion prior timeline |
| `k` | pause / resume an SVO replay (same as the Pause button) |
| click | select a keyframe link (motion prior) |
| `ctrl` + drag | rotate |
| `esc` | exit |

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Relative pose fusion status stays `OUT_OF_SYNC` | The sample timestamps do not share a reference with the images. They must be nanoseconds on the same clock as `TIME_REFERENCE::IMAGE`. |
| Relative pose fusion status stays `REJECTED` | Usually a wrong `sensor_transform`, an inverted sign convention, or `*_std` values that claim more confidence than the source deserves. |
| Relative pose fusion status stays `MISSING` | The trace does not overlap the images at all. The sample reports the two time windows on the first image, compare them. |
| `Option --map requires -o <output_area_file>` | `--map` needs a destination: `--map -o session.area`. |

## Support

If you need assistance go to our Community site at https://community.stereolabs.com/
