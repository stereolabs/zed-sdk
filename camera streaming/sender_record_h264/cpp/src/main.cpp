// Capture the SDK's outgoing streaming H264/H265 bitstream and save it to a
// standard Annex-B file that opens directly in VLC / ffplay / etc.
//
// Usage:
//   ZED_Sender_Record_H264 [out.h264] [seconds] [H264|H265]
//
// Notes:
//  - The SDK is still streaming over UDP as usual; this sample just taps the
//    same encoded bitstream and writes it to disk in parallel — no second
//    encoding session, no re-encoding loss.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <sl/Camera.hpp>

static std::atomic<bool> g_stop {false};
static void on_sigint(int) {
    g_stop.store(true);
}

int main(int argc, char** argv) {
    std::string out_path = (argc > 1) ? argv[1] : "out.h264";
    const int seconds = (argc > 2) ? std::atoi(argv[2]) : 10;
    const bool want_h264 = (argc > 3) && (std::string(argv[3]) == "H264" || std::string(argv[3]) == "h264");

    std::signal(SIGINT, on_sigint);

    sl::Camera zed;
    sl::InitParameters init;
    init.camera_resolution = sl::RESOLUTION::AUTO;
    init.camera_fps = 30;
    init.depth_mode = sl::DEPTH_MODE::NONE;
    init.sdk_verbose = 0;

    auto err = zed.open(init);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "open() failed: " << sl::toString(err) << "\n";
        return 1;
    }

    sl::StreamingParameters sp;
    sp.codec = want_h264 ? sl::STREAMING_CODEC::H264 : sl::STREAMING_CODEC::H265;
    sp.port = 30000;
    sp.bitrate = 8000;
    err = zed.enableStreaming(sp);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "enableStreaming() failed: " << sl::toString(err) << "\n";
        zed.close();
        return 1;
    }

    std::cout << "Streaming over UDP port " << sp.port << " in " << (want_h264 ? "H264" : "H265") << ", and writing encoded bitstream to "
              << out_path << " for ~" << seconds << "s (Ctrl-C to stop early)\n";

    // Adjust extension hint if user kept default but chose H265
    if (!want_h264 && out_path == "out.h264")
        out_path = "out.hevc";

    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "Cannot open " << out_path << " for writing\n";
        zed.disableStreaming();
        zed.close();
        return 1;
    }

    // Show what the SDK exposes right now — should report SENDING active.
    for (const auto& info : zed.getEncodedStreamsInfo()) {
        const char* src = (info.source == sl::ENCODED_STREAM_SOURCE::RECEIVING) ? "RECEIVING"
            : (info.source == sl::ENCODED_STREAM_SOURCE::SENDING)               ? "SENDING"
            : (info.source == sl::ENCODED_STREAM_SOURCE::RECORDING)             ? "RECORDING"
                                                                                : "?";
        std::cout << "  - " << src << " active=" << (info.active ? "yes" : "no")
                  << " codec=" << (info.codec == sl::STREAMING_CODEC::H264 ? "H264" : "H265") << " bitrate=" << info.bitrate_kbps
                  << " kbps\n";
    }

    const auto t_end = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    size_t total_bytes = 0;
    size_t frames_written = 0;
    size_t keyframes = 0;

    while (!g_stop.load() && std::chrono::steady_clock::now() < t_end) {
        if (zed.grab() != sl::ERROR_CODE::SUCCESS)
            continue;

        sl::EncodedStreamPacket pkt;
        auto r = zed.retrieveEncodedStreamPacket(pkt, sl::ENCODED_STREAM_SOURCE::SENDING);
        if (r != sl::ERROR_CODE::SUCCESS)
            continue;

        out.write(reinterpret_cast<const char*>(pkt.data), static_cast<std::streamsize>(pkt.size));
        total_bytes += pkt.size;
        frames_written++;
        if (pkt.is_keyframe)
            keyframes++;
    }

    out.close();
    zed.disableStreaming();
    zed.close();

    std::cout << "Wrote " << frames_written << " frames (" << keyframes << " keyframes), " << total_bytes << " bytes → " << out_path
              << "\n";
    std::cout << "Open with: ffplay " << out_path << "    (or VLC)\n";
    return 0;
}
