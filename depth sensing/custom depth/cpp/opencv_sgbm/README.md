# Custom Depth — OpenCV SGBM

Minimal `DEPTH_MODE::CUSTOM` sample: the disparity is computed on CPU with `cv::StereoSGBM` at **half resolution** and ingested back into the SDK, which then produces all its usual depth outputs from it.

What it demonstrates:

- the `read()` → retrieve rectified images → compute → `ingestCustomDepth()` → `grab()` sequence
- CPU map ingest (the SDK uploads internally; the `cv::Mat` is reusable as soon as the call returns)
- any-resolution ingest: disparity values are expressed in pixels at the map's own resolution
- `scale` usage: SGBM outputs 16-bit fixed point (disparity ×16) → `scale = 1/16`
- invalid handling for free: SGBM marks invalid pixels with negative values, which the ingest classifies as invalid — no cleanup pass needed
- the SDK outputs generated from the ingested disparity: depth view (OpenCV window) and 3D colored point cloud (OpenGL viewer)

## Build & run

```bash
mkdir build && cd build
cmake ..
make
./ZED_Custom_Depth_SGBM            # live camera
./ZED_Custom_Depth_SGBM file.svo2  # SVO playback (works identically)
```

Expect modest quality/speed — CPU SGBM is the simplest possible external matcher and is meant as plumbing demonstration, not as a depth reference. See the `tensorrt_stereo_onnx` sample for a GPU network-based source.
