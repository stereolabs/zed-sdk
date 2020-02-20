# Stereolabs ZED - SVO Export

This sample demonstrates how to read a SVO file and convert it into an AVI file (LEFT + RIGHT) or (LEFT + DEPTH_VIEW).

It can also convert a SVO in the following png image sequences: LEFT+RIGHT, LEFT+DEPTH_VIEW, and LEFT+DEPTH_16Bit.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).
  
### Prerequisites

- Windows 10, Ubuntu LTS
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [ZED Python API](https://www.stereolabs.com/docs/app-development/python/)
- OpenCV python (`python -m pip install opencv-python`)

## Run the example

```
python "export_svo.py" svo_file.svo
```

Usage:

```

export_svo.py A B C

Please use the following parameters from the command line:
 A - SVO file path (input) : "path/to/file.svo"
 B - AVI file path (output) or image sequence folder(output) : "path/to/output/file.avi" or "path/to/output/folder/"
 C - Export mode:  0=Export LEFT+RIGHT AVI.
				   1=Export LEFT+DEPTH_VIEW AVI.
				   2=Export LEFT+RIGHT image sequence.
				   3=Export LEFT+DEPTH_VIEW image sequence.
				   4=Export LEFT+DEPTH_16Bit image sequence.
 A and B need to end with '/' or '\'

Examples:
  (AVI LEFT+RIGHT)              export_svo.py "path/to/file.svo" "path/to/output/file.avi" 0
  (AVI LEFT+DEPTH)              export_svo.py "path/to/file.svo" "path/to/output/file.avi" 1
  (SEQUENCE LEFT+RIGHT)         export_svo.py "path/to/file.svo" "path/to/output/folder/" 2
  (SEQUENCE LEFT+DEPTH)         export_svo.py "path/to/file.svo" "path/to/output/folder/" 3
  (SEQUENCE LEFT+DEPTH_16Bit)   export_svo.py "path/to/file.svo" "path/to/output/folder/" 4
```