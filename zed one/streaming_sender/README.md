# ZED SDK - ZED One Streaming Sender

This sample shows how to stream live video from a ZED One camera over the network.

## Features

- Stream video using H.265 codec
- Configurable port and resolution
- Low-latency network transmission

## Usage

### C++

```bash
cd build
./ZED_One_Streaming_Sender [port]
```

### Python

```bash
python streaming_sender.py [--resolution <RESOLUTION>]
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `port` | Network port for streaming | 30000 |
| `resolution` | Video resolution (HD2K, HD1200, HD1080, HD720, SVGA, VGA) | AUTO |

## Example

```bash
# Stream on default port
./ZED_One_Streaming_Sender

# Stream on custom port
./ZED_One_Streaming_Sender 30001

# Python with resolution
python streaming_sender.py --resolution HD1080
```

Press `Ctrl+C` to stop streaming.
