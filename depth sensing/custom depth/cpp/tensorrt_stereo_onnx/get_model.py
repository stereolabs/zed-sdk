#!/usr/bin/env python3
"""
Fetch or build a stereo disparity ONNX for the ZED custom depth TensorRT sample.

Default (no arguments): download CREStereo (Apache-2.0), ready to run:
    python3 get_model.py
    ./build/ZED_Custom_Depth_TensorRT models/crestereo_init_iter5_480x640.onnx

SOTA option: Fast-FoundationStereo (NVlabs, research-only license) — downloads the
authors' prebuilt single-file ONNX directly (no repo clone, no export, no gdown):
    python3 get_model.py --model fast-foundation-stereo            # 576x960, 8 iters
    python3 get_model.py --model fast-foundation-stereo --ffs-resolution 320x736 --iters 4
Custom sizes still possible with --export (clones the repo, downloads the checkpoint,
runs their export script):
    python3 get_model.py --model fast-foundation-stereo --export --height 480 --width 640

The download base URL can be overridden (e.g. Stereolabs-hosted bucket) with --url or the
SL_CUSTOM_DEPTH_MODEL_URL environment variable.
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

# CREStereo ONNX exports (PINTO0309 conversion, Apache-2.0), hosted on stereodemo's GitHub
# releases. Use the "init" variants only: 2 inputs (left, right), raw 0-255 RGB input,
# output (1,2,H,W) with disparity in plane 0 — matching the sample contract.
CRESTEREO_BASE_URL = os.environ.get(
    "SL_CUSTOM_DEPTH_MODEL_URL",
    "https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo")

FFS_REPO = "https://github.com/NVlabs/Fast-FoundationStereo.git"
FFS_GDRIVE_FOLDER = "1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap"  # checkpoint folder from their readme

# Pre-exported single-file ONNX published by the authors in the same Drive folder
# (weights/onnx/23_36_37/<res>/): downloading one of these is much lighter than
# cloning the repo + checkpoint + exporting. Resolution is HxW.
FFS_PREBUILT_ONNX = {
    ("320x736", 4): "1p9vgRh_8R1FXA79l8VdnN28dDQ3NEEhf",
    ("320x736", 8): "19vkjOQlgmHDDoJ14QjroXn2mCPm64hRY",
    ("576x960", 4): "1zgNEsa6DAg6NgaH0GCtr-riVUPAyXRWw",
    ("576x960", 8): "1sgH9SwmRT45NnNixHdDfEh4Xs0ZEkQ5Q",
}


def download(url, dest):
    print(f"Downloading {url}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"

    last_done = [-1]

    def report(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size * 100 // total, 100)
            if done != last_done[0]:  # one update per percent, plays nice when piped
                last_done[0] = done
                sys.stdout.write(f"\r  {done}% of {total // (1024 * 1024)} MB")
                sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=report)
    print()
    os.replace(tmp, dest)
    return dest


def get_crestereo(args):
    name = f"crestereo_init_{args.variant}_{args.resolution}.onnx"
    dest = os.path.join(MODELS_DIR, name)
    if os.path.isfile(dest):
        print(f"Already present: {dest}")
        return dest
    return download(f"{CRESTEREO_BASE_URL}/{name}", dest)


def run(cmd, cwd=None):
    print("+ " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def get_fast_foundation_stereo(args):
    print("Fast-FoundationStereo weights are released under a NON-COMMERCIAL (research-only) "
          "NVIDIA license.\nFine for evaluation, not for products or redistribution.")
    if input("Continue? [y/N] ").strip().lower() != "y":
        sys.exit(1)

    if not args.export:
        # Fast path: the authors publish ready-made single-file ONNX exports
        key = (args.ffs_resolution, args.iters)
        if key not in FFS_PREBUILT_ONNX:
            sys.exit(f"No prebuilt export for resolution={args.ffs_resolution} iters={args.iters}. "
                     f"Available: {sorted(set(k[0] for k in FFS_PREBUILT_ONNX))} x iters {sorted(set(k[1] for k in FFS_PREBUILT_ONNX))}, "
                     "or use --export for a custom size.")
        name = f"fast_foundation_stereo_{args.ffs_resolution}_iter{args.iters}.onnx"
        dest = os.path.join(MODELS_DIR, name)
        if os.path.isfile(dest):
            print(f"Already present: {dest}")
            return dest
        # Direct download endpoint: works on public Drive files where gdown's uc?id= flow fails
        return download("https://drive.usercontent.google.com/download"
                        f"?id={FFS_PREBUILT_ONNX[key]}&export=download&confirm=t", dest)

    # --export: full legwork (clone + checkpoint + ONNX export), for custom resolutions
    if shutil.which("gdown") is None:
        sys.exit("The checkpoint is hosted on Google Drive: please `pip install gdown` and re-run.")
    repo_dir = os.path.join(SCRIPT_DIR, "third_party", "Fast-FoundationStereo")
    if not os.path.isdir(repo_dir):
        run(["git", "clone", "--depth", "1", FFS_REPO, repo_dir])

    weights_dir = os.path.join(repo_dir, "weights")
    have_ckpt = os.path.isdir(weights_dir) and \
        any(f.endswith(".pth") for _, _, fs in os.walk(weights_dir) for f in fs)
    if not have_ckpt:
        if shutil.which("gdown") is None:
            sys.exit("The checkpoint is hosted on Google Drive: please `pip install gdown` and re-run.")
        run(["gdown", "--folder", FFS_GDRIVE_FOLDER, "-O", weights_dir])

    ckpt = None
    for root, _, files in os.walk(weights_dir):
        for f in files:
            if f.endswith(".pth"):
                ckpt = os.path.join(root, f)
    if ckpt is None:
        sys.exit(f"No .pth checkpoint found under {weights_dir}")

    out_dir = os.path.join(repo_dir, "output")
    run([sys.executable, "scripts/make_single_onnx.py",
         "--model_dir", ckpt, "--save_path", out_dir,
         "--height", str(args.height), "--width", str(args.width),
         "--valid_iters", str(args.iters), "--max_disp", "192"], cwd=repo_dir)

    exported = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".onnx")]
    if not exported:
        sys.exit(f"Export did not produce an .onnx in {out_dir} — check the script output above "
                 "(their requirements.txt must be installed: pip install -r third_party/Fast-FoundationStereo/requirements.txt)")
    exported.sort(key=os.path.getmtime)
    name = f"fast_foundation_stereo_{args.height}x{args.width}_iter{args.iters}.onnx"
    dest = os.path.join(MODELS_DIR, name)
    os.makedirs(MODELS_DIR, exist_ok=True)
    shutil.copy2(exported[-1], dest)
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["crestereo", "fast-foundation-stereo"], default="crestereo")
    p.add_argument("--variant", default="iter5", choices=["iter2", "iter5", "iter10", "iter20"],
                   help="CREStereo refinement iterations baked in the export (default: iter5)")
    p.add_argument("--resolution", default="480x640", choices=["240x320", "480x640", "720x1280"],
                   help="CREStereo input size HxW (default: 480x640)")
    p.add_argument("--ffs-resolution", default="576x960", choices=["320x736", "576x960"],
                   help="Fast-FoundationStereo prebuilt export size HxW (default: 576x960)")
    p.add_argument("--iters", type=int, default=8, choices=[4, 8],
                   help="Fast-FoundationStereo refinement iterations (default: 8)")
    p.add_argument("--export", action="store_true",
                   help="Build the Fast-FoundationStereo ONNX from source instead of downloading the prebuilt one (custom --height/--width)")
    p.add_argument("--height", type=int, default=480, help="export height with --export (multiple of 32)")
    p.add_argument("--width", type=int, default=640, help="export width with --export (multiple of 32)")
    p.add_argument("--url", default=None, help="Override the CREStereo download base URL")
    args = p.parse_args()
    if args.url:
        global CRESTEREO_BASE_URL
        CRESTEREO_BASE_URL = args.url

    dest = get_crestereo(args) if args.model == "crestereo" else get_fast_foundation_stereo(args)
    print(f"\nModel ready: {dest}")
    print(f"Run: ./build/ZED_Custom_Depth_TensorRT \"{dest}\"")


if __name__ == "__main__":
    main()
