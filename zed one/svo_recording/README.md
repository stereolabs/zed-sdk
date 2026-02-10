# ZED SDK - ZED One SVO Recording

This sample shows how to record video from a ZED One camera to SVO format.

## Features

- Record video to SVO format using H.265 compression
- Display recording status (frames recorded, compression ratio)
- Graceful stop with Ctrl+C

## Usage

### C++

```bash
cd build
./ZED_One_SVO_Recording <output_svo_file>
```

### Python

```bash
python svo_recording.py --output_svo_file <output_svo_file>
```

## Example

```bash
# C++
./ZED_One_SVO_Recording my_recording.svo2

# Python
python svo_recording.py --output_svo_file my_recording.svo2
```

Press `Ctrl+C` to stop recording.
