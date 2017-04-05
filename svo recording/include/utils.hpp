#ifndef __UTILS_SVO_RECORDING_INCLUDE__
#define __UTILS_SVO_RECORDING_INCLUDE__

#include <cstdio>
#include <cstring>
#include <signal.h>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <mutex>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>

#include <sl/Camera.hpp>

struct InfoOption {
    std::string svo_path;
    bool recordingMode = false;
    std::string output_path;
    bool computeDisparity = false;
    bool videoMode = false;
};

inline bool testFileExist(std::string &filename) {
#ifdef _WIN32
    struct _stat64 buffer;
    return (_stat64(filename.c_str(), &buffer) == 0);
#else
	struct stat64 buffer;
	return (stat64(filename.c_str(), &buffer) == 0);
#endif
}

void parse_args(int argc, char **argv, InfoOption &info);

void recordVideo(sl::Mat &image);
void recordImages(sl::Mat &image);

void initActions(sl::Camera *zed, InfoOption &modes);
void manageActions(sl::Camera *zed, char &key, InfoOption &modes);
void exitActions();

void generateImageToRecord(sl::Camera *zed, InfoOption &modes, sl::Mat &out);

#endif