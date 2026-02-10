# ZED SDK - ZED One SVO Playback

This sample shows how to read and play back SVO files recorded with a ZED One camera.

## Features

- Load and read SVO files
- Display video frames using OpenCV
- Control playback (forward, backward, save frame)
- Show playback progress

## Usage

### C++

```bash
cd build
./ZED_One_SVO_Playback <path_to_svo_file>
```

### Python

```bash
python svo_playback.py --input_svo_file <path_to_svo_file>
```

## Controls

| Key | Action |
|-----|--------|
| `s` | Save current frame as PNG |
| `f` | Jump forward 1 second |
| `b` | Jump backward 1 second |
| `q` | Quit |
