build failed in depth sensing/automatic region of interest/cpp

## Steps to Reproduce
```commandline
git clone git@github.com:stereolabs/zed-sdk.git
cd "depth sensing/automatic region of interest/cpp"
mkdir build
cd build
cmake ../CMakeLists.txt
make
```

## Expected Result
make should be successful.
## Actual Result

```commandline
someone@someone-orin:~/github/zed-sdk/depth sensing/automatic region of interest/cpp/build$ make
[ 50%] Building CXX object CMakeFiles/ZED_Auto_Sensing_ROI.dir/src/main.cpp.o
/home/someone/github/zed-sdk/depth sensing/automatic region of interest/cpp/src/main.cpp: In function ‘int main(int, char**)’:
/home/someone/github/zed-sdk/depth sensing/automatic region of interest/cpp/src/main.cpp:88:15: error: ‘struct sl::RegionOfInterestParameters’ has no member named ‘auto_apply’
   88 |     roi_param.auto_apply = true;
      |               ^~~~~~~~~~
make[2]: *** [CMakeFiles/ZED_Auto_Sensing_ROI.dir/build.make:76: CMakeFiles/ZED_Auto_Sensing_ROI.dir/src/main.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/ZED_Auto_Sensing_ROI.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
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
