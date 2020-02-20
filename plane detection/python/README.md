# Stereolabs ZED - Plane detection

This sample shows how to retrieves the floor plane from the scene and the plane at a specific image coordinate with the ZED SDK.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

### Prerequisites

- Windows 10, Ubuntu LTS
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

### Plane Detection

Plane Detection sample is searching for the floor in a video and extracts it into a mesh if it found it.

```
python "plane_example.py" svo_file.svo
```
