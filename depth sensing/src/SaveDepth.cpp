#include "SaveDepth.hpp"

using namespace std;

int count_save = 0;
int mode_PointCloud = 0;
int mode_Depth = 0;
sl::POINT_CLOUD_FORMAT PointCloud_format;
sl::DEPTH_FORMAT Depth_format;

std::string getPointCloudFormatName(sl::POINT_CLOUD_FORMAT f) {
    std::string str_;
    switch (f) {
        case sl::POINT_CLOUD_FORMAT_XYZ_ASCII:
            str_ = "XYZ";
            break;
        case sl::POINT_CLOUD_FORMAT_PCD_ASCII:
            str_ = "PCD";
            break;
        case sl::POINT_CLOUD_FORMAT_PLY_ASCII:
            str_ = "PLY";
            break;
        case sl::POINT_CLOUD_FORMAT_VTK_ASCII:
            str_ = "VTK";
            break;
        default:
            break;
    }
    return str_;
}

std::string getDepthFormatName(sl::DEPTH_FORMAT f) {
    std::string str_;
    switch (f) {
        case sl::DEPTH_FORMAT_PNG:
            str_ = "PNG";
            break;
        case sl::DEPTH_FORMAT_PFM:
            str_ = "PFM";
            break;
        case sl::DEPTH_FORMAT_PGM:
            str_ = "PGM";
            break;
        default:
            break;
    }
    return str_;
}

void processKeyEvent(sl::Camera& zed, char &key) {

    switch (key) {

        case 'd':
        case 'D':
            saveDepth(zed, path + prefixDepth + to_string(count_save));

            break;

        case 'n': // Depth format
        case 'N':
        {
            mode_Depth++;
            Depth_format = static_cast<sl::DEPTH_FORMAT> (mode_Depth % 3);
            std::cout << "Depth format: " << getDepthFormatName(Depth_format) << std::endl;
        }
            break;


        case 'p':
        case 'P':
            savePointCloud(zed, path + prefixPointCloud + to_string(count_save));
            break;


        case 'm': // Point cloud format
        case 'M':
        {
            mode_PointCloud++;
            PointCloud_format = static_cast<sl::POINT_CLOUD_FORMAT> (mode_PointCloud % 4);
            std::cout << "Point Cloud format: " << getPointCloudFormatName(PointCloud_format) << std::endl;
        }
            break;

        case 'h': // Print help
        case 'H':
            cout << helpString << endl;
            break;

        case 's': // Save side by side image
            saveSbSImage(zed, std::string("ZED_image") + std::to_string(count_save) + std::string(".png"));
            break;
    }
    count_save++;
}

void savePointCloud(sl::Camera& zed, std::string filename) {
    std::cout << "Saving Point Cloud... " << flush;
    bool t = sl::savePointCloudAs(zed, PointCloud_format, filename.c_str(), true, false);
    if (t)
        std::cout << "Done" << endl;

    else
        std::cout << "Failed... Please check that you have permissions to write on disk" << endl;
}

void saveDepth(sl::Camera& zed, std::string filename) {
    float max_value = std::numeric_limits<unsigned short int>::max();
    float scale_factor = max_value / zed.getDepthMaxRangeValue();

    std::cout << "Saving Depth Map... " << flush;
    bool t = sl::saveDepthAs(zed, Depth_format, filename.c_str(), scale_factor);
    if (t)
        std::cout << "Done" << endl;

    else
        std::cout << "Failed... Please check that you have permissions to write on disk" << endl;
}

// Side by side images saving function using opencv

void saveSbSImage(sl::Camera& zed, std::string filename) {
    sl::Resolution image_resolution = zed.getResolution();

    cv::Mat sbs_image(image_resolution.height, image_resolution.width * 2, CV_8UC4);
    cv::Mat left_image(sbs_image, cv::Rect(0, 0, image_resolution.width, image_resolution.height));
    cv::Mat right_image(sbs_image, cv::Rect(image_resolution.width, 0, image_resolution.width, image_resolution.height));

    sl::Mat tmp_sl;
    cv::Mat tmp_cv;

    zed.retrieveImage(tmp_sl, sl::VIEW_LEFT);
    tmp_cv = cv::Mat(tmp_sl.getHeight(), tmp_sl.getWidth(), CV_8UC4, tmp_sl.getPtr<sl::uchar1>(sl::MEM_CPU));
    tmp_cv.copyTo(left_image);
    zed.retrieveImage(tmp_sl, sl::VIEW_RIGHT);
    tmp_cv = cv::Mat(tmp_sl.getHeight(), tmp_sl.getWidth(), CV_8UC4, tmp_sl.getPtr<sl::uchar1>(sl::MEM_CPU));
    tmp_cv.copyTo(right_image);

    cv::imshow("Image", sbs_image);
    cv::cvtColor(sbs_image, sbs_image, CV_RGBA2RGB);

    cv::imwrite(filename, sbs_image);
}
