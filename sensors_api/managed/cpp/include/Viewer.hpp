#ifndef VIEWER_HPP
#define VIEWER_HPP

#include <sl/Camera.hpp>
#include <sl/Lidar.hpp>

enum class ZedViewMode {
    Image,
    PointCloud
};

class Viewer {
public:
    virtual ~Viewer() = default;
    virtual void init(int argc, char** argv) = 0;
    virtual bool isAvailable() = 0;
    virtual bool isDone() = 0;
    virtual void updateLidar(const std::string& id, sl::Mat& point_cloud, sl::Mat& intensity_img) = 0;
    virtual void updateZed(const std::string& id, sl::Mat& image, sl::Mat* point_cloud = nullptr) = 0;
    virtual void updateZedOne(const std::string& id, sl::Mat& image) = 0;
    virtual void updateZedObjects(
        const std::string& id,
        const sl::Objects& objects,
        const sl::Matrix4f* transform = nullptr,
        sl::Timestamp timestamp = 0
    ) = 0;
    virtual void
    updateZedBodies(const std::string& id, const sl::Bodies& bodies, const sl::Matrix4f* transform = nullptr, sl::Timestamp timestamp = 0)
        = 0;
};

#endif // VIEWER_HPP
