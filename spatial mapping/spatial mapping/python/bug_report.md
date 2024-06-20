## Steps to Reproduce
```commandline
git clone git@github.com:stereolabs/zed-sdk.git
cd "depth sensing/automatic region of interest/python"
python3 automatic_region_of_interest.py
```

## Expected Result
should not raise error just after starting the script.

## Actual Result

```commandline
python3 automatic_region_of_interest.py 
[Sample] Using default resolution
[2024-06-20 09:32:21 UTC][ZED][INFO] Logging level INFO
[2024-06-20 09:32:22 UTC][ZED][INFO] Using USB input... Switched to default resolution HD720
[2024-06-20 09:32:22 UTC][ZED][INFO] [Init]  Depth mode: NEURAL
[2024-06-20 09:32:23 UTC][ZED][INFO] [Init]  Camera successfully opened.
[2024-06-20 09:32:23 UTC][ZED][INFO] [Init]  Camera FW version: 1523
[2024-06-20 09:32:23 UTC][ZED][INFO] [Init]  Video mode: HD720@60
[2024-06-20 09:32:23 UTC][ZED][INFO] [Init]  Serial Number: S/N 32045770
Press 'a' to apply the ROI
Press 'r' to reset the ROI
Press 's' to save the ROI as image file to reload it later
Press 'l' to load the ROI from an image file
Traceback (most recent call last):
  File "automatic_region_of_interest.py", line 173, in <module>
    main()
  File "automatic_region_of_interest.py", line 108, in main
    roi_param.auto_apply = True
AttributeError: 'pyzed.sl.RegionOfInterestParameters' object has no attribute 'auto_apply'

```

## ZED Camera model
ZED2i

## Environment
Jetson AGX orin
Ubuntu 20.04 LTS
JetPack 5.1
python3.8.10
SDK version: 4.1.1

## installer
ZED_SDK_Tegra_L4T35.2_v4.1.1.zstd.run
