# ZED SDK - ZED One Streaming Receiver

This sample shows how to receive and display a video stream from a remote ZED One camera.

## Features

- Receive video stream over the network
- Display video using OpenCV
- Camera settings control via keyboard
- Exposure ROI selection via mouse

## Usage

### C++

```bash
cd build
./ZED_One_Streaming_Receiver <ip:port>
```

### Python

```bash
python streaming_receiver.py --ip_address <ip:port>
```

## Example

```bash
# C++
./ZED_One_Streaming_Receiver 192.168.1.100:30000

# Python
python streaming_receiver.py --ip_address 192.168.1.100:30000
```

## Controls

| Key | Action |
|-----|--------|
| `+` | Increase current setting value |
| `-` | Decrease current setting value |
| `s` | Switch to next camera setting |
| `l` | Toggle LED on/off |
| `r` | Reset all settings to default |
| `a` | Apply exposure ROI (after mouse selection) |
| `f` | Reset exposure ROI to full image |
| `q` | Quit |

## Mouse Controls

- **Left click + drag**: Select exposure ROI area
- **Right click**: Clear ROI selection
