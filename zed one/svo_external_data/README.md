# ZED SDK - ZED One SVO External Data

This sample shows how to embed and retrieve custom data within SVO files recorded with a ZED One camera.

## Features

- Embed custom data into SVO files during recording
- Retrieve custom data from SVO files during playback
- Timestamp-synchronized data storage

## Samples

### Recording with External Data
Record video while embedding custom data at each frame.

```bash
# C++
./ZED_One_SVO_External_Data_Recording <output_svo_file>

# Python (not available yet)
```

### Playback with External Data
Play back an SVO file and retrieve the embedded custom data.

```bash
# C++
./ZED_One_SVO_External_Data_Playback <input_svo_file>

# Python (not available yet)
```

## Use Cases

- Embed sensor data (GPS, IMU, etc.) synchronized with video frames
- Store metadata or annotations within the SVO file
- Create synchronized multi-modal datasets
