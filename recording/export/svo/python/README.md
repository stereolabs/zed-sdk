# Stereolabs ZED - SVO Export

This sample demonstrates how to read a SVO file and convert it into an AVI file (LEFT + RIGHT) or (LEFT + DEPTH_VIEW).

It can also convert a SVO in the following png image sequences: LEFT+RIGHT, LEFT+DEPTH_VIEW, and LEFT+DEPTH_16Bit.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/) and [pyZED Package](https://www.stereolabs.com/docs/app-development/python/install/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)
 
## Run the program

To run the program, use the following command in your terminal:
```bash
python export_svo.py --mode <mode> --input_svo_file <input_svo_file> --output_avi_file <output_avi_file> --output_path_dir <output_path_dir>
```

Arguments: 
  - --mode Mode 0 is to export LEFT+RIGHT AVI. <br /> Mode 1 is to export LEFT+DEPTH_VIEW Avi. <br /> Mode 2 is to export LEFT+RIGHT image sequence. <br /> Mode 3 is to export LEFT+DEPTH_View image sequence. <br /> Mode 4 is to export LEFT+DEPTH_16BIT image sequence.
  - --input_svo_file Path to an existing .svo file 
  - --output_avi_file Path to a .avi file that will be created
  - --output_path_dir Path to an existing folder where .png will be saved
### Features
 - Export .svo file to LEFT+RIGHT .avi
 - Export .svo file to LEFT+DEPTH_VIEW .avi
 - Export .svo file to LEFT+RIGHT image sequence
 - Export .svo file to LEFT+DEPTH_View image sequence
 - Export .svo file to LEFT+DEPTH_16BIT image sequence
Examples : 
```
python export_svo.py --mode 0 --input_svo_file <input_svo_file> --output_avi_file <output_avi_file> 
python export_svo.py --mode 1 --input_svo_file <input_svo_file> --output_avi_file <output_avi_file> 
python export_svo.py --mode 2 --input_svo_file <input_svo_file> --output_path_dir <output_path_dir> 
python export_svo.py --mode 3 --input_svo_file <input_svo_file> --output_path_dir <output_path_dir> 
python export_svo.py --mode 4 --input_svo_file <input_svo_file> --output_path_dir <output_path_dir> 
```

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/