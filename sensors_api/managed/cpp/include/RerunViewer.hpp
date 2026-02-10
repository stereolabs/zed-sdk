#ifndef RERUN_VIEWER_HPP
#define RERUN_VIEWER_HPP

#include "Viewer.hpp"

#ifdef WITH_RERUN
    #include <rerun.hpp>
#endif

#include <mutex>
#include <unordered_map>
#include <vector>

enum class RerunMode {
    Spawn,   // Default: spawn local viewer
    Connect, // Connect to remote viewer
    Save     // Save to .rrd file
};

struct RerunOptions {
    RerunMode mode = RerunMode::Spawn;
    std::string connect_addr; // For Connect mode: "host:port"
    std::string save_path;    // For Save mode: "output.rrd"
};

class RerunViewer : public Viewer {
public:
    RerunViewer();
    RerunViewer(const RerunOptions& options);
    ~RerunViewer() override = default;

    void init(int argc, char** argv) override;
    bool isAvailable() override;
    bool isDone() override;

    void updateLidar(const std::string& id, sl::Mat& point_cloud, sl::Mat& intensity_img) override;
    void updateZed(const std::string& id, sl::Mat& image, sl::Mat* point_cloud = nullptr) override;
    void updateZedOne(const std::string& id, sl::Mat& image) override;
    void updateZedObjects(
        const std::string& id,
        const sl::Objects& objects,
        const sl::Matrix4f* transform = nullptr,
        sl::Timestamp timestamp = 0
    ) override;
    void
    updateZedBodies(const std::string& id, const sl::Bodies& bodies, const sl::Matrix4f* transform = nullptr, sl::Timestamp timestamp = 0)
        override;

private:
    std::string sanitize(std::string str);

    std::mutex m_mutex;
    RerunOptions m_options;

#ifdef WITH_RERUN
    rerun::RecordingStream rec;

    struct Buffer {
        std::vector<rerun::Position3D> positions;
        std::vector<rerun::Color> colors;
        std::vector<uint8_t> img_data;
    };
    std::unordered_map<std::string, Buffer> m_buffers;
#endif
};

#endif // RERUN_VIEWER_HPP
