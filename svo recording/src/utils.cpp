#include "utils.hpp"

using namespace std;

cv::VideoWriter *video_writer = NULL;
std::string image_name;
int im_count = 0;
sl::Mat buff;

void parse_args(int argc, char **argv, InfoOption &info) {

    //*  OpenCV4Tegra (2.4) and OpenCV 3.1 handles parameters in a different ways.
    //*  In the following lines, we show how to handle both ways by checking if we are on the Jetson (_SL_JETSON_) or not to take "OpenCV2.4" or "OpenCV3.1" style
#ifdef _SL_JETSON_
    const cv::String keys = {
        "{ h | help       || print help message }"
        "{ f | filename   |  | SVO filename to record (if svo doesn't exist) or to playback (if svo does exist) (ex : -f=test.svo  or --filename=test.svo)  }"
        "{ o | output      |  | Name of the output converted video file (*.avi), Left+ VIEW_DEPTH with -z option, Left+Right otherwise. }"
        "{ z | depth  || Compute depth}"
    };
    cv::CommandLineParser parser(argc, argv, keys.c_str());
#else
    const cv::String keys =
            "{help h usage ?|  | print help message. (ex : ZED_SVO_Recording -h}"
            "{filename f    |  | SVO filename to record (if svo doesn't exist) or to playback (if svo does exist) (ex : ZED_SVO_Recording -f=test.svo  or ZED_SVO_Recording --filename=test.svo) }"
            "{output o       |  |  Name of the output converted video file (*.avi) from a svo given by -f, Left+ VIEW_DEPTH with -z option, Left+Right otherwise}"
            "{depth z   |  | Compute depth}";

    sl::String version = sl::Camera::getSDKVersion();

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Sample from ZED SDK " + std::string(version)); //about is not available under OpenCV2.4
#endif

#ifdef _SL_JETSON_
    if (parser.get<bool>("help")) {
        parser.printParams();
        exit(0);
    }
#else
    if (parser.has("help")) {
        parser.printMessage();
        exit(0);
    }
#endif

    info.svo_path = parser.get<std::string>("filename");

    if (info.svo_path.empty()) {
        cout << "  !!!! You must specify at least the SVO filename  -- see allowed options !!!!\n" << std::endl;
#ifdef _SL_JETSON_
        parser.printParams();
#else
        parser.printMessage();
#endif
        exit(0);
    }

    info.output_path = parser.get<std::string>("output");
    if (!info.output_path.empty() && testFileExist(info.svo_path)) {
        if (info.output_path.find(".avi") != std::string::npos) {
            std::cout << "Converting svo file into video file" << std::endl;
            info.videoMode = true;
        } else
            std::cout << "Converting svo file into a sequence of images" << std::endl;
    } else if (!testFileExist(info.svo_path)) {
        // Detect if a ZED is connected to begin the recording
        info.recordingMode = sl::Camera::isZEDconnected();
        if (info.recordingMode) cout << "ZED Detected ! Recording mode enabled" << std::endl;
    } else {
        cout << "Please provides an output path, or replace the SVO path by a valid non existing one" << endl;
        exit(0);
    }

    if (!info.output_path.empty() && info.recordingMode) cout << "Output path given ignored" << endl;

#ifdef _SL_JETSON_
    if (parser.get<bool>("depth") && !info.output_path.empty()) {
#else
    if (parser.has("depth") && !info.output_path.empty()) {
#endif
        cout << "depth map enabled" << endl;
        info.computeDisparity = 1;
    }
}

void initActions(sl::Camera *zed, InfoOption &modes) {
    if (modes.videoMode && !video_writer) video_writer = new cv::VideoWriter(modes.output_path, CV_FOURCC('M', '4', 'S', '2'), 25,
            cv::Size(zed->getResolution().width * 2, zed->getResolution().height));
    else image_name = modes.output_path;

    buff.alloc(zed->getResolution().width * 2, zed->getResolution().height, sl::MAT_TYPE_8U_C3);
}

void exitActions() {
    if (video_writer) delete video_writer;
}

void recordVideo(sl::Mat &image) {
    
    video_writer->write(cv::Mat(image.getHeight(), image.getWidth(), CV_8UC4, image.getPtr<sl::uchar1>(sl::MEM_CPU)));
}

void recordImages(sl::Mat &image) {
    image.write((image_name + "_" + std::to_string(im_count) + ".png").c_str());
}

void generateImageToRecord(sl::Camera *zed, InfoOption &modes, sl::Mat &out) {

    sl::Mat leftIm, rightIm;
    cv::Size size(zed->getResolution().width * 2, zed->getResolution().height);
    cv::Mat sbsIm(size, CV_8UC4);

    // Left
    zed->retrieveImage(leftIm, sl::VIEW_LEFT);
    // Right (Disparity Map or Right image)
    if (modes.computeDisparity) zed->retrieveImage(rightIm, sl::VIEW_DEPTH);
    else zed->retrieveImage(rightIm, sl::VIEW_RIGHT);

    cv::Mat tmp_cv;
    tmp_cv = cv::Mat(leftIm.getHeight(), leftIm.getWidth(), CV_8UC4, leftIm.getPtr<sl::uchar1>(sl::MEM_CPU));
    tmp_cv.copyTo(sbsIm(cv::Rect(0, 0, zed->getResolution().width, zed->getResolution().height)));
    tmp_cv = cv::Mat(rightIm.getHeight(), rightIm.getWidth(), CV_8UC4, rightIm.getPtr<sl::uchar1>(sl::MEM_CPU));
    tmp_cv.copyTo(sbsIm(cv::Rect(zed->getResolution().width, 0, zed->getResolution().width, zed->getResolution().height)));
    tmp_cv = cv::Mat(out.getHeight(), out.getWidth(), CV_8UC4, out.getPtr<sl::uchar1>(sl::MEM_CPU));
    cv::cvtColor(sbsIm, tmp_cv, CV_RGBA2RGB);
}

void manageActions(sl::Camera *zed, char &key, InfoOption &modes) {

    generateImageToRecord(zed, modes, buff);

    if (modes.videoMode) recordVideo(buff);
    else recordImages(buff);

    switch (key) {
        case 'f':
        case 'F':
            if (!modes.recordingMode) {
                zed->setSVOPosition(zed->getSVOPosition() + 100);
                cout << "fast forward" << endl;
            }
            break;
        case 'r':
        case 'R':
            if (!modes.recordingMode) {
                zed->setSVOPosition(zed->getSVOPosition() - 100);
                cout << "fast rewind" << endl;
            }
            break;
    }
    im_count++;

}
