#!/usr/bin/env python3
"""Generate samples.json: a machine-readable catalog of every ZED SDK sample.

A "sample" is the directory that *contains* one or more language folders
(``cpp`` / ``python`` / ``csharp`` / ``c``). This captures every granularity in
the repo, from module-level samples (e.g. ``camera control``) to deeply nested
ones (e.g. ``recording/export/svo``).

The output is deterministic (stable sort, no timestamps) so a CI job can run this
script and fail on ``git diff`` if the committed ``samples.json`` is out of date.

Usage (from the repo root):
    python tools/generate_index.py            # write samples.json
    python tools/generate_index.py --check     # exit 1 if samples.json is stale

Standard library only; no third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "samples.json"

LANG_DIRS = {"cpp": "cpp", "python": "python", "csharp": "csharp", "c": "c"}
SKIP_TOP = {".git", ".github", "build", "tools"}

# Per-module display name + canonical documentation page (the current Fern docs URLs,
# in `.md` form per the docs site's own llms.txt convention). Keep these in sync with
# the documentation site; every URL here is verified to resolve.
DOCS_ROOT = "https://docs.stereolabs.com/docs.md"
_M = "https://docs.stereolabs.com/docs/development/zed-sdk/modules/"
MODULE_INFO = {
    "body tracking":       ("Body Tracking",       _M + "body-tracking.md"),
    "camera control":      ("Camera Control",      _M + "camera/camera-controls.md"),
    "camera streaming":    ("Camera Streaming",    _M + "camera/local-network-streaming.md"),
    "depth sensing":       ("Depth Sensing",       _M + "depth-sensing.md"),
    "global localization": ("Global Localization", _M + "global-localization.md"),
    "object detection":    ("Object Detection",    _M + "object-detection.md"),
    "plane detection":     ("Plane Detection",     _M + "spatial-mapping/plane-detection.md"),
    "positional tracking": ("Positional Tracking", _M + "positional-tracking.md"),
    "recording":           ("Recording",           _M + "camera/recording.md"),
    "sensors_api":         ("Sensors API",         _M + "sensors.md"),
    "spatial mapping":     ("Spatial Mapping",     _M + "spatial-mapping.md"),
    "tutorials":           ("Tutorials",           "https://docs.stereolabs.com/docs/development/zed-sdk/modules.md"),
    "virtual stereo":      ("Virtual Stereo",      "https://docs.stereolabs.com/docs/products/cameras/zedxone/dual-camera-stereo-vision.md"),
    "zed one":             ("ZED One",             "https://docs.stereolabs.com/docs/products/cameras/zedxone.md"),
}


def detect_sdk_version() -> str:
    """Read 'ZED SDK X.Y' from the root README so the manifest tracks releases."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"ZED SDK\s+(\d+\.\d+)", readme)
    return match.group(1) if match else "unknown"


def find_sample_dirs() -> dict[Path, set[str]]:
    """Map each sample directory -> set of languages it provides."""
    samples: dict[Path, set[str]] = {}
    for lang_dir in REPO_ROOT.rglob("*"):
        if not lang_dir.is_dir() or lang_dir.name not in LANG_DIRS:
            continue
        rel = lang_dir.relative_to(REPO_ROOT)
        if rel.parts[0] in SKIP_TOP:
            continue
        sample_dir = lang_dir.parent
        samples.setdefault(sample_dir, set()).add(LANG_DIRS[lang_dir.name])
    return samples


def first_paragraph(readme: Path) -> str:
    """Extract a one-line description: the first prose paragraph of a README.

    Skips HTML/badges/images/comments and collapses the first real sentence(s)
    into a single trimmed line.
    """
    if not readme.is_file():
        return ""
    text = readme.read_text(encoding="utf-8", errors="ignore")
    # Drop HTML comments and tags, then walk line by line.
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    para: list[str] = []
    saw_bullet = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if para:
                break
            continue
        if line.startswith("#"):            # heading
            continue
        if line.startswith(("<", "![", "|", ">", "---", "===")):  # html/image/table/quote/rule
            continue
        if re.fullmatch(r"\[!\[.*\]\(.*\)\]\(.*\)", line):         # badge
            continue
        bullet = re.match(r"[-*]\s+(.*)", line)
        if bullet:
            saw_bullet = True
            line = bullet.group(1)
        para.append(line)
    blob = "; ".join(para) if saw_bullet else " ".join(para)
    blob = re.sub(r"<[^>]+>", "", blob)                  # stray inline html
    blob = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", blob)     # inline images
    blob = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", blob)  # links -> text
    blob = re.sub(r"[*`_]", "", blob)                    # emphasis/code marks
    blob = re.sub(r"\s+", " ", blob).strip()
    if len(blob) > 280:
        blob = blob[:277].rsplit(" ", 1)[0] + "..."
    return blob


def describe(sample_dir: Path, languages: list[str], module_key: str) -> str:
    """Find the best README for a description, searching sample -> lang -> module."""
    candidates = [sample_dir / "README.md"]
    candidates += [sample_dir / lang / "README.md" for lang in languages]
    candidates.append(REPO_ROOT / module_key / "README.md")
    for readme in candidates:
        desc = first_paragraph(readme)
        if desc:
            return desc
    # No README anywhere: synthesize a minimal, non-fabricated description.
    display = MODULE_INFO.get(module_key, (module_key.title(), ""))[0]
    name = sample_dir.relative_to(REPO_ROOT / module_key).as_posix()
    return f"{display} sample." if name == "." else f"{name} — {display} sample."


def best_readme(sample_dir: Path, languages: list[str]) -> str:
    candidates = [sample_dir / "README.md"]
    candidates += [sample_dir / lang / "README.md" for lang in languages]
    for readme in candidates:
        if readme.is_file():
            return readme.relative_to(REPO_ROOT).as_posix()
    return ""


def slugify(path_str: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path_str.lower()).strip("-")


def build_index() -> dict:
    samples = []
    for sample_dir, lang_set in find_sample_dirs().items():
        rel = sample_dir.relative_to(REPO_ROOT)
        module_key = rel.parts[0]
        display, docs = MODULE_INFO.get(module_key, (module_key.title(), DOCS_ROOT))
        languages = sorted(lang_set)
        sub = rel.relative_to(module_key).as_posix()  # "." when sample == module
        name = display if sub == "." else sub
        samples.append({
            "id": slugify(rel.as_posix()),
            "module": display,
            "name": name,
            "path": rel.as_posix(),
            "languages": languages,
            "description": describe(sample_dir, languages, module_key),
            "docs": docs,
            "readme": best_readme(sample_dir, languages),
        })
    samples.sort(key=lambda s: s["path"])
    return {
        "sdk_version": detect_sdk_version(),
        "repository": "https://github.com/stereolabs/zed-sdk",
        "count": len(samples),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate samples.json")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if samples.json is out of date (for CI)")
    args = parser.parse_args()

    index = build_index()
    rendered = json.dumps(index, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != rendered:
            print("samples.json is out of date. Run: python tools/generate_index.py",
                  file=sys.stderr)
            return 1
        print(f"samples.json is up to date ({index['count']} samples).")
        return 0

    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({index['count']} samples).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
