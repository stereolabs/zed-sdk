# AGENTS.md

Guidance for AI coding agents (and new contributors) working in a local clone of this
repository. Human-oriented overview lives in [README.md](README.md); a machine-readable
catalog of every sample is in [samples.json](samples.json); a link index for retrieval is
in [llms.txt](llms.txt).

## What this repository is

This repo contains **tutorials and code samples** for the Stereolabs **ZED SDK**. It is
**not** the SDK itself — the ZED SDK is a separate binary library that must be installed
first (see Prerequisites). Each sample shows how to use one SDK feature and is provided in
**C++**, **Python**, and/or **C#**.

## Repository layout

Top-level directories are SDK modules; each contains one or more samples, and each sample
contains `cpp/`, `python/` and/or `csharp/` subfolders with its source and a `README.md`.

| Directory | Module |
| --- | --- |
| `tutorials/` | Progressive, minimal tutorials for every module (C++, Python, C#, C) — start here. |
| `camera control/` | Capture images and adjust camera video settings. |
| `camera streaming/` | Stream the video feed over the network and receive it. |
| `depth sensing/` | Point clouds and depth maps (fusion, ROI, voxel, refocus, export). |
| `positional tracking/` | 6-DoF camera pose tracking. |
| `global localization/` | Fuse tracking with GNSS/GPS for geo-referenced positioning. |
| `spatial mapping/` | 3D meshes and fused point clouds. |
| `object detection/` | 3D object detection, custom YOLO detectors, multi-camera. |
| `body tracking/` | Human body/skeleton tracking, FBX/JSON export, integrations. |
| `plane detection/` | Plane and floor-plane detection. |
| `recording/` | Record/playback SVO/SVO2 files (encrypted, multi-camera, external data). |
| `sensors_api/` | IMU/sensor data and multi-sensor (camera + LiDAR) management. |
| `virtual stereo/` | Virtual stereo pair (C++ only). |
| `zed one/` | ZED One mono-camera samples (live, streaming, SVO, custom inference). |

`tools/` holds the `samples.json` generator. `samples.json`, `llms.txt`, `AGENTS.md`,
`CLAUDE.md`, `README.md` and `CMakeLists.txt` live at the root.

## Prerequisites

- **ZED SDK 5.3** or later — see
  [Getting Started](https://docs.stereolabs.com/docs/development/zed-sdk.md) and install on
  [Windows](https://docs.stereolabs.com/docs/development/zed-sdk/windows.md),
  [Linux](https://docs.stereolabs.com/docs/development/zed-sdk/linux.md) or
  [Jetson](https://docs.stereolabs.com/docs/development/zed-sdk/linux/work-with-nvidia-jetson.md).
- An **NVIDIA GPU** with **Compute Capability > 5**, plus **CUDA** (installed with the SDK).
- A ZED camera to run against, or an `.svo`/`.svo2` recording to run without hardware (see
  the `recording/` samples).

## Build & run

The exact command for each sample is in **that sample's `README.md`** — treat it as the
source of truth. General patterns:

**C++** — build per sample with CMake:
```bash
cd "<sample>/cpp"
mkdir build && cd build
cmake ..            # Windows: cmake .. then cmake --build . --config Release
make -j$(nproc)
./<Executable>      # e.g. ./ZED_Camera_Control
```
The **root `CMakeLists.txt`** builds a curated subset of samples at once. It builds either
C++ or C# (not both): `cmake -DBUILD_CPP=ON ..` (default) or `-DBUILD_CPP=OFF` for C#.
With `INSTALL_SAMPLES=ON` (default) binaries are deployed to `./bin`.

**Python** — install the SDK and the
[`pyzed` package](https://docs.stereolabs.com/docs/development/api-languages/python.md), then:
```bash
cd "<sample>/python"
pip install -r requirements.txt   # only where a requirements.txt is present
python <sample>.py
```

**C#** — Windows only; NuGet packages download automatically. Open the project and build,
or use the .NET CLI. See the sample's `csharp/README.md`.

## Conventions & gotchas

- **Folder names contain spaces** (e.g. `depth sensing/`). Always quote paths in shells and
  `%20`-encode them in URLs.
- **Language coverage varies per sample** — not every sample has all three languages (e.g.
  Virtual Stereo is C++ only). Check `languages` in `samples.json` rather than assuming.
- **Don't duplicate the API reference or install guides here** — they live on
  https://docs.stereolabs.com (append `.md` to any page URL for clean Markdown, or use the
  docs MCP server at https://docs.stereolabs.com/_mcp/server).

## Maintaining the agent-readiness files

- After **adding, moving, or removing a sample**, regenerate the manifest:
  ```bash
  python tools/generate_index.py
  ```
  CI runs `python tools/generate_index.py --check` and fails if `samples.json` is stale.
- After **adding a new top-level module**, also add a line to `llms.txt` (curated by hand)
  and a row to the table above. Bump the SDK version reference in `README.md`; the generator
  reads it automatically into `samples.json`.
