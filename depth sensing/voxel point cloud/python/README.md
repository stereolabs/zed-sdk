# Voxel Point Cloud — Python

This sample demonstrates voxel-decimated 3D point cloud capture with the ZED SDK Python API. It displays an interactive OpenGL viewer where you can adjust voxel parameters in real time.

## Features

- Live voxel-decimated point cloud with 3D viewer
- Toggle between voxel and full point cloud
- Interactive control of all `VoxelMeasureParameters` fields
- Three voxelization modes: FIXED, STEREO_UNCERTAINTY, LINEAR
- SVO playback with pause and seek
- Optional GPU data transfer via CuPy

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

## Run

```bash
python voxel_point_cloud.py                            # Live camera
python voxel_point_cloud.py --input_svo_file path.svo  # SVO file
python voxel_point_cloud.py --ip_address 192.168.1.10  # Stream
```
