# ZED SDK - Body Tracking

This sample shows how to detect and track human bodies in space.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
 - Check the [Documentation](https://docs.stereolabs.com/docs/)

## Build the program
 - Build for [Windows](https://docs.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://docs.stereolabs.com/docs/app-development/cpp/linux/)

## Run the program
*NOTE: The ZED v1 is not compatible with this module*
- Navigate to the build directory and launch the executable
- Or open a terminal in the build directory and run the sample :

      ./ZED_Body_Tracking

With no argument the sample opens a live camera with the accurate model. An SVO file or a
network stream can be given as the first argument, and the body tracking configuration can be
changed from the command line. Run `./ZED_Body_Tracking --help` for the full list.

| Option | Values | Description |
|---|---|---|
| `--input <src>` | `file.svo2`, `ip[:port]` | Input source (also accepted as the first positional argument) |
| `--resolution <res>` | `HD2K` `HD1200` `HD1080` `HD720` `SVGA` `XVGA` `TXGA` `VGA` `AUTO` | Camera resolution |
| `--depth <mode>` | `neural_light` `neural` `neural_plus` | Depth mode |
| `--model <m>` | `fast` `medium` `accurate` | Body tracking model |
| `--generation <g>` | `1` `2` | Body tracking network generation |
| `--format <f>` | `18` `34` `38` | Keypoint format |
| `--confidence <n>` | `0`-`100` | Detection confidence threshold |
| `--smoothing <n>` | `0`-`1` | Skeleton smoothing |
| `--fitting` | | Enable body fitting (implied by `--format 34`) |
| `--no-segmentation` | | Disable the body segmentation mask |

For example, to compare both network generations on the same recording:

      ./ZED_Body_Tracking recording.svo2 --model accurate --generation 1
      ./ZED_Body_Tracking recording.svo2 --model accurate --generation 2

## Features
 - Display bodies bounding boxes by pressing the `b` key.
 - Press `q` to quit and `m` to pause / resume.

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/