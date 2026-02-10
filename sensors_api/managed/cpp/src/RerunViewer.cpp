#include "RerunViewer.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>

#ifdef WITH_RERUN
RerunViewer::RerunViewer()
    : rec("sensors_api_sample") { }

RerunViewer::RerunViewer(const RerunOptions& options)
    : rec("sensors_api_sample")
    , m_options(options) { }

void RerunViewer::init(int argc, char** argv) {
    switch (m_options.mode) {
        case RerunMode::Connect:
            std::cout << "Connecting to Rerun viewer at: " << m_options.connect_addr << std::endl;
            rec.connect_grpc(m_options.connect_addr).exit_on_failure();
            break;
        case RerunMode::Save:
            std::cout << "Recording to Rerun file: " << m_options.save_path << std::endl;
            rec.save(m_options.save_path).exit_on_failure();
            break;
        case RerunMode::Spawn:
        default:
            auto opts = rerun::SpawnOptions();
            opts.memory_limit = "8000MB";
            rec.spawn(opts).exit_on_failure();
            break;
    }
    rec.log("world", rerun::ViewCoordinates::RIGHT_HAND_Y_UP);
}

bool RerunViewer::isAvailable() {
    return true;
}

bool RerunViewer::isDone() {
    return false; // Rerun doesn't control the loop
}

std::string RerunViewer::sanitize(std::string str) {
    std::string original = str;
    std::replace(str.begin(), str.end(), ' ', '_');
    std::replace(str.begin(), str.end(), '-', '_');
    std::replace(str.begin(), str.end(), ':', '_');
    std::replace(str.begin(), str.end(), '.', '_');
    return str;
}

void RerunViewer::updateLidar(const std::string& id, sl::Mat& point_cloud, sl::Mat& intensity_img) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string prefix = "world/lidar_" + sanitize(id);
    auto& buf = m_buffers[id];

    // Use sensor timestamp from point cloud, fall back to wall clock if not available
    int64_t ts_ns = point_cloud.timestamp.getNanoseconds();
    if (ts_ns == 0) {
        std::cout << "[RerunViewer] Warning: No timestamp in point_cloud for lidar " << id << ", using wall clock" << std::endl;
        ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // 1. Log Point Cloud
    size_t num_points = point_cloud.getWidth() * point_cloud.getHeight();
    buf.positions.clear();
    buf.colors.clear();
    buf.positions.reserve(num_points);
    buf.colors.reserve(num_points);

    sl::float4* data_ptr = point_cloud.getPtr<sl::float4>();
    for (size_t k = 0; k < num_points; k++) {
        sl::float4 pt = data_ptr[k];
        if (pt.x == 0 && pt.y == 0 && pt.z == 0)
            continue;

        buf.positions.push_back({pt.x, pt.y, pt.z});

        // Color is packed as ABGR: (A << 24) | (B << 16) | (G << 8) | R (matching ZED SDK RGBA mode)
        uint32_t color_uint;
        memcpy(&color_uint, &pt.w, sizeof(uint32_t));
        uint8_t r = (color_uint >> 0) & 0xFF;
        uint8_t g = (color_uint >> 8) & 0xFF;
        uint8_t b = (color_uint >> 16) & 0xFF;
        buf.colors.push_back({r, g, b, 255});
    }
    rec.set_time_nanos("timestamp", ts_ns);
    rec.log(prefix + "/points", rerun::Points3D(buf.positions).with_colors(buf.colors));

    // 2. Log Intensity Image
    int channels = intensity_img.getChannels();
    size_t img_size = intensity_img.getWidth() * intensity_img.getHeight() * 4;
    if (buf.img_data.size() < img_size)
        buf.img_data.resize(img_size);

    if (channels == 1) {
        // Convert 1-channel to RGBA (grayscale)
        uint8_t* src = (uint8_t*)intensity_img.getPtr<sl::uchar1>();
        size_t step = intensity_img.getStepBytes();
        for (int r = 0; r < intensity_img.getHeight(); r++) {
            uint8_t* row_src = src + r * step;
            uint8_t* row_dst = &buf.img_data[r * intensity_img.getWidth() * 4];
            for (int c = 0; c < intensity_img.getWidth(); c++) {
                uint8_t val = row_src[c];
                row_dst[c * 4 + 0] = val; // R
                row_dst[c * 4 + 1] = val; // G
                row_dst[c * 4 + 2] = val; // B
                row_dst[c * 4 + 3] = 255; // A
            }
        }
    } else if (channels == 4) {
        for (int r = 0; r < intensity_img.getHeight(); r++) {
            memcpy(
                &buf.img_data[r * intensity_img.getWidth() * 4],
                (uint8_t*)intensity_img.getPtr<sl::uchar1>() + r * intensity_img.getStepBytes(),
                intensity_img.getWidth() * 4
            );
        }
    }

    rec.set_time_nanos("timestamp", ts_ns);
    rec.log(
        prefix + "/image",
        rerun::Image(
            rerun::Collection<uint8_t>::borrow(buf.img_data.data(), img_size),
            rerun::WidthHeight(intensity_img.getWidth(), intensity_img.getHeight()),
            rerun::datatypes::ColorModel::RGBA
        )
    );
}

void RerunViewer::updateZed(const std::string& id, sl::Mat& image, sl::Mat* point_cloud) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string prefix = "world/zed_" + sanitize(id);
    auto& buf = m_buffers[id];

    // Use sensor timestamp from image, fall back to wall clock if not available
    int64_t ts_ns = image.timestamp.getNanoseconds();
    if (ts_ns == 0) {
        std::cout << "[RerunViewer] Warning: No timestamp in image for zed " << id << ", using wall clock" << std::endl;
        ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Image
    if (image.getWidth() > 0) {
        int channels = image.getChannels();
        size_t img_size = image.getWidth() * image.getHeight() * 4;
        if (buf.img_data.size() < img_size)
            buf.img_data.resize(img_size);

        if (channels == 4) {
            for (int r = 0; r < image.getHeight(); r++) {
                memcpy(
                    &buf.img_data[r * image.getWidth() * 4],
                    (uint8_t*)image.getPtr<sl::uchar1>() + r * image.getStepBytes(),
                    image.getWidth() * 4
                );
            }
            rec.set_time_nanos("timestamp", ts_ns);
            rec.log(
                prefix + "/image",
                rerun::Image(
                    rerun::Collection<uint8_t>::borrow(buf.img_data.data(), img_size),
                    rerun::WidthHeight(image.getWidth(), image.getHeight()),
                    rerun::datatypes::ColorModel::BGRA
                )
            );
        } else if (channels == 3) {
            // Convert BGR to BGRA
            uint8_t* src = (uint8_t*)image.getPtr<sl::uchar1>();
            size_t step = image.getStepBytes();
            for (int r = 0; r < image.getHeight(); r++) {
                uint8_t* row_src = src + r * step;
                uint8_t* row_dst = &buf.img_data[r * image.getWidth() * 4];
                for (int c = 0; c < image.getWidth(); c++) {
                    row_dst[c * 4 + 0] = row_src[c * 3 + 0]; // B
                    row_dst[c * 4 + 1] = row_src[c * 3 + 1]; // G
                    row_dst[c * 4 + 2] = row_src[c * 3 + 2]; // R
                    row_dst[c * 4 + 3] = 255;                // A
                }
            }
            rec.set_time_nanos("timestamp", ts_ns);
            rec.log(
                prefix + "/image",
                rerun::Image(
                    rerun::Collection<uint8_t>::borrow(buf.img_data.data(), img_size),
                    rerun::WidthHeight(image.getWidth(), image.getHeight()),
                    rerun::datatypes::ColorModel::BGRA
                )
            );
        }
    }

    // Cloud
    if (point_cloud && point_cloud->getWidth() > 0) {
        size_t num_points = point_cloud->getWidth() * point_cloud->getHeight();
        int step = 4;

        buf.positions.clear();
        buf.colors.clear();
        buf.positions.reserve(num_points / step);
        buf.colors.reserve(num_points / step);

        sl::float4* data_ptr = point_cloud->getPtr<sl::float4>();
        for (size_t k = 0; k < num_points; k += step) {
            sl::float4 pt = data_ptr[k];
            if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z)) {
                buf.positions.push_back({pt.x, pt.y, pt.z});

                uint32_t color_uint;
                memcpy(&color_uint, &pt.w, sizeof(uint32_t));
                uint8_t b = (color_uint >> 0) & 0xFF;
                uint8_t g = (color_uint >> 8) & 0xFF;
                uint8_t r = (color_uint >> 16) & 0xFF;
                buf.colors.push_back({b, g, r, 255});
            }
        }
        rec.set_time_nanos("timestamp", ts_ns);
        rec.log(prefix + "/points", rerun::Points3D(buf.positions).with_colors(buf.colors));
    }
}

void RerunViewer::updateZedOne(const std::string& id, sl::Mat& image) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string prefix = "world/zedone_" + sanitize(id);
    auto& buf = m_buffers[id];

    // Use sensor timestamp from image, fall back to wall clock if not available
    int64_t ts_ns = image.timestamp.getNanoseconds();
    if (ts_ns == 0) {
        std::cout << "[RerunViewer] Warning: No timestamp in image for zedone " << id << ", using wall clock" << std::endl;
        ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    if (image.getWidth() > 0) {
        int channels = image.getChannels();
        size_t img_size = image.getWidth() * image.getHeight() * 4;
        if (buf.img_data.size() < img_size)
            buf.img_data.resize(img_size);

        if (channels == 4) {
            for (int r = 0; r < image.getHeight(); r++) {
                memcpy(
                    &buf.img_data[r * image.getWidth() * 4],
                    (uint8_t*)image.getPtr<sl::uchar1>() + r * image.getStepBytes(),
                    image.getWidth() * 4
                );
            }
            rec.set_time_nanos("timestamp", ts_ns);
            rec.log(
                prefix + "/image",
                rerun::Image(
                    rerun::Collection<uint8_t>::borrow(buf.img_data.data(), img_size),
                    rerun::WidthHeight(image.getWidth(), image.getHeight()),
                    rerun::datatypes::ColorModel::BGRA
                )
            );
        } else if (channels == 3) {
            // Convert BGR to BGRA
            uint8_t* src = (uint8_t*)image.getPtr<sl::uchar1>();
            size_t step = image.getStepBytes();
            for (int r = 0; r < image.getHeight(); r++) {
                uint8_t* row_src = src + r * step;
                uint8_t* row_dst = &buf.img_data[r * image.getWidth() * 4];
                for (int c = 0; c < image.getWidth(); c++) {
                    row_dst[c * 4 + 0] = row_src[c * 3 + 0]; // B
                    row_dst[c * 4 + 1] = row_src[c * 3 + 1]; // G
                    row_dst[c * 4 + 2] = row_src[c * 3 + 2]; // R
                    row_dst[c * 4 + 3] = 255;                // A
                }
            }
            rec.set_time_nanos("timestamp", ts_ns);
            rec.log(
                prefix + "/image",
                rerun::Image(
                    rerun::Collection<uint8_t>::borrow(buf.img_data.data(), img_size),
                    rerun::WidthHeight(image.getWidth(), image.getHeight()),
                    rerun::datatypes::ColorModel::BGRA
                )
            );
        }
    }
}

void RerunViewer::updateZedObjects(
    const std::string& id,
    const sl::Objects& objects,
    const sl::Matrix4f* transform,
    sl::Timestamp timestamp
) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string prefix = "world/zed_" + sanitize(id) + "/objects";

    // Use sensor timestamp if available, otherwise fall back to wall clock
    int64_t ts_ns = timestamp.getNanoseconds();
    if (ts_ns == 0) {
        std::cout << "[RerunViewer] Warning: No timestamp provided for zed objects " << id << ", using wall clock" << std::endl;
        ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    std::vector<rerun::Position3D> centers;
    std::vector<rerun::HalfSize3D> half_sizes;
    std::vector<rerun::Color> colors;
    std::vector<rerun::Text> labels;

    for (const auto& obj : objects.object_list) {
        if (obj.bounding_box.size() != 8)
            continue;

        sl::float3 pos = obj.position;
        if (transform) {
            float x = pos.x * (*transform)(0, 0) + pos.y * (*transform)(0, 1) + pos.z * (*transform)(0, 2) + (*transform)(0, 3);
            float y = pos.x * (*transform)(1, 0) + pos.y * (*transform)(1, 1) + pos.z * (*transform)(1, 2) + (*transform)(1, 3);
            float z = pos.x * (*transform)(2, 0) + pos.y * (*transform)(2, 1) + pos.z * (*transform)(2, 2) + (*transform)(2, 3);
            pos = sl::float3(x, y, z);
        }

        centers.push_back({pos.x, pos.y, pos.z});
        half_sizes.push_back({obj.dimensions.x * 0.5f, obj.dimensions.y * 0.5f, obj.dimensions.z * 0.5f});

        // Color based on object ID
        float hue = fmod(obj.id * 0.618033988749895f, 1.0f);
        float s = 0.8f, v = 0.9f;
        float c = v * s;
        float x_hsv = c * (1 - fabs(fmod(hue * 6.0f, 2.0f) - 1));
        float m = v - c;
        float r, g, b;
        int hi = (int)(hue * 6.0f);
        switch (hi) {
            case 0:
                r = c;
                g = x_hsv;
                b = 0;
                break;
            case 1:
                r = x_hsv;
                g = c;
                b = 0;
                break;
            case 2:
                r = 0;
                g = c;
                b = x_hsv;
                break;
            case 3:
                r = 0;
                g = x_hsv;
                b = c;
                break;
            case 4:
                r = x_hsv;
                g = 0;
                b = c;
                break;
            default:
                r = c;
                g = 0;
                b = x_hsv;
                break;
        }
        colors.push_back({(uint8_t)((r + m) * 255), (uint8_t)((g + m) * 255), (uint8_t)((b + m) * 255), 200});
        labels.push_back(rerun::Text(std::to_string(obj.id)));
    }

    if (!centers.empty()) {
        rec.set_time_nanos("timestamp", ts_ns);
        rec.log(prefix, rerun::Boxes3D::from_centers_and_half_sizes(centers, half_sizes).with_colors(colors).with_labels(labels));
    }
}

void RerunViewer::updateZedBodies(const std::string& id, const sl::Bodies& bodies, const sl::Matrix4f* transform, sl::Timestamp timestamp) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string prefix = "world/zed_" + sanitize(id) + "/bodies";

    // Use sensor timestamp if available, otherwise fall back to wall clock
    int64_t ts_ns = timestamp.getNanoseconds();
    if (ts_ns == 0) {
        ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Body 38 bone connections
    static const std::vector<std::pair<int, int>> BODY_38_BONES = {
        {0,  1 },
        {1,  2 },
        {2,  3 },
        {3,  4 },
        {4,  5 },
        {5,  6 },
        {6,  7 },
        {5,  8 },
        {8,  9 },
        {3,  10},
        {10, 11},
        {11, 12},
        {12, 13},
        {3,  14},
        {14, 15},
        {15, 16},
        {16, 17},
        {0,  18},
        {18, 19},
        {19, 20},
        {20, 21},
        {21, 22},
        {0,  23},
        {23, 24},
        {24, 25},
        {25, 26},
        {26, 27}
    };

    std::vector<rerun::Position3D> joint_positions;
    std::vector<rerun::Color> joint_colors;
    std::vector<rerun::LineStrip3D> skeleton_lines;

    for (const auto& body : bodies.body_list) {
        if (body.tracking_state != sl::OBJECT_TRACKING_STATE::OK || body.keypoint.empty())
            continue;

        // Color based on body ID
        float hue = fmod(body.id * 0.618033988749895f, 1.0f);
        float s = 0.8f, v = 0.9f;
        float c = v * s;
        float x_hsv = c * (1 - fabs(fmod(hue * 6.0f, 2.0f) - 1));
        float m = v - c;
        float r, g, b;
        int hi = (int)(hue * 6.0f);
        switch (hi) {
            case 0:
                r = c;
                g = x_hsv;
                b = 0;
                break;
            case 1:
                r = x_hsv;
                g = c;
                b = 0;
                break;
            case 2:
                r = 0;
                g = c;
                b = x_hsv;
                break;
            case 3:
                r = 0;
                g = x_hsv;
                b = c;
                break;
            case 4:
                r = x_hsv;
                g = 0;
                b = c;
                break;
            default:
                r = c;
                g = 0;
                b = x_hsv;
                break;
        }
        rerun::Color body_color {(uint8_t)((r + m) * 255), (uint8_t)((g + m) * 255), (uint8_t)((b + m) * 255), 255};

        // Add joints
        for (const auto& kp : body.keypoint) {
            if (std::isfinite(kp.x) && std::isfinite(kp.y) && std::isfinite(kp.z)) {
                joint_positions.push_back({kp.x, kp.y, kp.z});
                joint_colors.push_back(body_color);
            }
        }

        // Add skeleton lines
        for (const auto& bone : BODY_38_BONES) {
            if (bone.first < static_cast<int>(body.keypoint.size()) && bone.second < static_cast<int>(body.keypoint.size())) {
                const sl::float3& kp1 = body.keypoint[bone.first];
                const sl::float3& kp2 = body.keypoint[bone.second];
                if (std::isfinite(kp1.x) && std::isfinite(kp1.y) && std::isfinite(kp1.z) && std::isfinite(kp2.x) && std::isfinite(kp2.y)
                    && std::isfinite(kp2.z)) {
                    skeleton_lines.push_back(rerun::LineStrip3D({
                        {kp1.x, kp1.y, kp1.z},
                        {kp2.x, kp2.y, kp2.z}
                    }));
                }
            }
        }
    }

    rec.set_time_nanos("timestamp", ts_ns);
    if (!joint_positions.empty()) {
        rec.log(prefix + "/joints", rerun::Points3D(joint_positions).with_colors(joint_colors).with_radii({0.03f}));
    }
    if (!skeleton_lines.empty()) {
        rec.log(prefix + "/skeleton", rerun::LineStrips3D(skeleton_lines));
    }
}

#else
// Stub implementation when Rerun is not available
RerunViewer::RerunViewer() { }
RerunViewer::RerunViewer(const RerunOptions& options)
    : m_options(options) { }
void RerunViewer::init(int argc, char** argv) { }
bool RerunViewer::isAvailable() {
    return false;
}
void RerunViewer::updateLidar(const std::string& id, sl::Mat& point_cloud, sl::Mat& intensity_img) { }
void RerunViewer::updateZed(const std::string& id, sl::Mat& image, sl::Mat* point_cloud) { }
void RerunViewer::updateZedOne(const std::string& id, sl::Mat& image) { }
void RerunViewer::updateZedObjects(
    const std::string& id,
    const sl::Objects& objects,
    const sl::Matrix4f* transform,
    sl::Timestamp timestamp
) { }
void RerunViewer::updateZedBodies(const std::string& id, const sl::Bodies& bodies, const sl::Matrix4f* transform, sl::Timestamp timestamp) {
}
bool RerunViewer::isDone() {
    return true;
}
#endif
