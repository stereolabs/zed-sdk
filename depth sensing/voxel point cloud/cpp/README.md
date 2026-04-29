# Voxel Point Cloud — C++

This sample demonstrates voxel-decimated 3D point cloud capture with the ZED SDK. It displays an interactive OpenGL viewer where you can adjust voxel parameters in real time.

## Features

- Live voxel-decimated point cloud with 3D viewer
- Toggle between voxel and full point cloud
- Interactive control of all `VoxelMeasureParameters` fields
- Three voxelization modes: FIXED, STEREO_UNCERTAINTY, LINEAR
- Adjustable voxel size and resolution scale
- Toggle between centroid and grid center positioning
- SVO playback with pause and seek
- Save decimated point cloud to `.ply`

## Controls

| Key | Action |
|-----|--------|
| `+`/`-` | Increase / decrease voxel size |
| `1`/`2`/`3` | Mode: FIXED / STEREO_UNCERTAINTY / LINEAR |
| `,`/`.` | Decrease / increase resolution scale |
| `c` | Toggle centroid / grid center |
| `p`/`P` | Decrease / increase point size |
| `d`/`D` | Decrease / increase depth confidence threshold |
| `v` | Toggle voxel / full point cloud |
| `r` | Reset all parameters to defaults |
| `s` | Save point cloud to `VoxelPointcloud.ply` |
| `Space` | Pause / resume (SVO) |
| `Left`/`Right` | Rewind / fast-forward 200 frames (SVO) |
| `q` / `Esc` | Quit |
| Mouse left + drag | Rotate camera |
| Mouse right + drag | Pan camera |
| Mouse wheel | Zoom |

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Run

```bash
./ZED_Voxel_Point_Cloud             # Live camera
./ZED_Voxel_Point_Cloud path.svo    # SVO file
./ZED_Voxel_Point_Cloud 192.168.1.10:30000  # Stream
```
