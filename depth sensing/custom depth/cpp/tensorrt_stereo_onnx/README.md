# Custom Depth — TensorRT stereo network (ONNX)

`DEPTH_MODE::CUSTOM` with a GPU source: a stereo disparity network runs with TensorRT and its **raw GPU output buffer is ingested directly** (`sl::Mat` wrapping the device pointer — no device-to-host copy of the disparity). The structure mirrors the custom object detection TensorRT samples: pass an `.onnx`, the engine is built and cached next to it on first run.

What it demonstrates, on top of the SGBM sample:

- **fully GPU-resident pipeline**: `sl::Camera::retrieveTensor(left, right, params)` pre-processes both rectified views (resize + RGB + NCHW + normalization) in one fused SDK pass into densely packed GPU tensors, bound directly as TensorRT inputs — the images and the disparity never touch the CPU, and the whole chain runs on the SDK CUDA stream with zero host synchronization
- **derived graded confidence**: a one-pass CUDA kernel turns the disparity itself into a confidence map (low at depth edges via the disparity gradient, zero on geometric occlusions `x − d < 0` and invalid pixels), ingested with the `PROBABILITY` convention — the runtime `confidence_threshold` then sweeps density as with the built-in modes
- GPU map ingest (zero extra copy of the network output)
- GEN_3 positional tracking running alongside custom depth (and feeding the depth stabilizer pose)
- network-resolution ingest: the disparity stays at the network size, the SDK rescales values internally
- the SDK outputs generated from the network disparity: depth view (OpenCV window) and 3D colored point cloud (OpenGL viewer)

## Quick start

```bash
mkdir build && cd build && cmake .. && make && cd ..

python3 get_model.py                  # downloads the default model (CREStereo, Apache-2.0)
./build/ZED_Custom_Depth_TensorRT     # finds the default model automatically; live camera
./build/ZED_Custom_Depth_TensorRT file.svo2   # same, on an SVO
```

The default model is **CREStereo** (`init` export, 2 inputs, Apache-2.0 — redistributable), fetched by `get_model.py`. Variants: `--variant iter2|iter5|iter10` (speed↔quality), `--resolution 240x320|480x640|720x1280`. The download URL can be redirected to another host with `--url` or `SL_CUSTOM_DEPTH_MODEL_URL`.

Input normalization is **auto-selected from the model filename** (`crestereo` → raw 0-255 RGB; anything else → ImageNet `(pixel/255 − mean)/std`), overridable with `--raw` / `--imagenet`.

## SOTA option: Fast-FoundationStereo

[Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) (NVlabs, CVPR 2026, ~14 ms at 640×480 on an RTX 3090 in TensorRT fp16) matches this sample's layout exactly — inputs `left_image`/`right_image` (1,3,H,W), output `disparity`, ImageNet normalization. The authors publish **prebuilt single-file ONNX exports** (320×736 and 576×960, 4 or 8 refinement iterations), which the script downloads directly — no repo clone, no checkpoint, no export step:

```bash
python3 get_model.py --model fast-foundation-stereo                 # 576x960, 8 iters
./build/ZED_Custom_Depth_TensorRT models/fast_foundation_stereo_576x960_iter8.onnx
```

For another input size, `--export` falls back to building the ONNX from source (clones the repo, downloads the checkpoint — requires `pip install gdown` — and runs their export script): `python3 get_model.py --model fast-foundation-stereo --export --height 480 --width 640`.

> ⚠ The Fast-FoundationStereo weights are released under a **non-commercial (research-only) license** — fine for evaluating this sample, not for products, and not redistributable (the script asks for confirmation). For a deployable model, use the CREStereo default, another permissively-licensed network, or your own.

## Bring your own ONNX

Any stereo network exported with:

- **two image inputs** (left then right, declaration order), NCHW float32, fixed size, 1 or 3 channels
- **one disparity output**, `(1[,1|2],H,W)` float32, disparity in plane 0, in pixels at the network input resolution, positive values

Other public models (RAFT-Stereo, IGEV-Stereo, ...) can be exported to this layout with their repository's ONNX export script using a fixed input size. The pre-processing is configured through `sl::TensorParameters` in `main.cpp` — adjust `scale`/`mean`/`std` if your export embeds a different normalization, and adapt `CustomDepthData::scale` if it outputs normalized (`scale = max_disparity`) or negative (`scale = -1`) disparities.

Relative/affine-invariant monocular models (MiDaS-style) are **not** usable: the pipeline is metric.

## Usage

```
./build/ZED_Custom_Depth_TensorRT [model.onnx | model.engine] [file.svo | file.svo2] [--raw | --imagenet]
```

All arguments are optional and recognized by extension/flag, in any order: no model → the `get_model.py` default is searched; no SVO → live camera.
