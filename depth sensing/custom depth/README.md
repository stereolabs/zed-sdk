# Custom Depth Ingest

These samples show how to feed the ZED SDK with your **own disparity/depth estimation** instead of the built-in modes, using `sl::DEPTH_MODE::CUSTOM` and `sl::Camera::ingestCustomDepth()`.

The per-frame sequence is:

```cpp
zed.read();                                   // 1. acquisition only (also feeds recording/streaming)
zed.retrieveImage(left,  sl::VIEW::LEFT);     // 2. rectified images (computed on demand)
zed.retrieveImage(right, sl::VIEW::RIGHT);
/* run your own stereo matcher / network */   // 3. external compute, any resolution
zed.ingestCustomDepth(data);                  // 4. must happen between read() and grab()
zed.grab(rt);                                 // 5. full pipeline runs on YOUR depth
```

After `grab()`, every depth-dependent feature works as usual: point cloud and measures, spatial mapping, object detection 3D boxes, plane detection, etc. Positional tracking is supported with `sl::POSITIONAL_TRACKING_MODE::GEN_3` (depth-free visual-inertial odometry).

## Samples

| Sample | External depth source | Input memory |
|---|---|---|
| [cpp/opencv_sgbm](cpp/opencv_sgbm) | OpenCV `cv::StereoSGBM` (CPU semi-global matching) | CPU |
| [cpp/tensorrt_stereo_onnx](cpp/tensorrt_stereo_onnx) | Any stereo ONNX network ran with TensorRT | GPU (zero extra copy) |
| [python](python) | OpenCV `cv2.StereoSGBM` (Python) | CPU (numpy view, no copy) |

## Key points

- **Map content**: `CUSTOM_DEPTH_FORMAT::DISPARITY` (pixels at the map's own resolution, positive = closer) or `CUSTOM_DEPTH_FORMAT::DEPTH` (metric, in `InitParameters::coordinate_units`).
- **`scale`**: one multiplier applied before interpretation. Use it for fixed-point outputs (SGBM gives disparity ×16 → `scale = 1/16`), normalized outputs (`scale = max_disparity`), negative conventions (`scale = -1`) or unit mismatches.
- **Invalid pixels**: NaN, negative disparities, 0/negative depths and ±Inf are classified at ingest — no need to clean the map first.
- **Confidence** is optional: without it, a validity-derived confidence is used. Provide one (`PROBABILITY` [0,1] or `ZED` [0,100] convention) to get graded filtering and better stabilization/mapping weighting.
- **grab() without ingest fails** by design (the frame is not consumed: you can still ingest and retry). To explicitly skip depth for a frame, set `RuntimeParameters::enable_depth = false`. A `read()`-only loop (no grab) is valid for recording/streaming.
