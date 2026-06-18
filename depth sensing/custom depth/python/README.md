# Custom Depth — Python (OpenCV SGBM)

Python version of the `DEPTH_MODE.CUSTOM` flow: disparity computed externally with `cv2.StereoSGBM` at half resolution and ingested back with `Camera.ingest_custom_depth()`, the SDK pipeline running on it.

What it demonstrates:

- the `read()` → retrieve rectified images → compute → `ingest_custom_depth()` → `grab()` sequence
- zero-copy numpy → `sl.Mat` input: `get_data(deep_copy=False)` returns a writable view over the Mat memory, so the SGBM result is written straight into the pre-allocated ingest Mat
- `scale` usage (SGBM 16-bit fixed point → `scale = 1/16`) and free invalid handling (SGBM negatives are classified invalid at ingest)
- graded texture-based confidence via the `PROBABILITY` convention, with an interactive `confidence_threshold` trackbar
- center-pixel metric depth printout as a scale sanity check

## Run

```bash
python3 custom_depth_sgbm.py             # live camera
python3 custom_depth_sgbm.py file.svo2   # SVO playback
```

Requires a pyzed build that includes the custom depth API (SDK ≥ 5.4).
