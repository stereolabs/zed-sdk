// Multi-source encoded-packet test. Exercises every combination of
// RECEIVING / SENDING / RECORDING simultaneously and writes one Annex-B
// file per active source.
//
// Modes:
//   --live [seconds] [H264|H265]
//       Open the LIVE camera; enable streaming and recording; tap
//       SENDING + RECORDING; write send.h264 and record.h264.
//
//   --stream <ip:port> [seconds]
//       Open from a network sender; tap RECEIVING; write recv.h264.
//
//   --all <ip:port> [seconds]
//       Open from a network sender, also enable RE-streaming on
//       port 30100 and recording. All three taps fire; expect
//       recv.h264 + send.h264 + record.h264.
//
// Each file is verified afterwards with `ffprobe <file>`.
//
// Note: the LIVE mode requires a connected ZED camera. The --stream
// modes require a running sender (e.g. the ZED_Streaming_Sender sample
// on the same machine or another).

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <sl/Camera.hpp>

static std::atomic<bool> g_stop {false};
static void on_sigint(int) {
    g_stop.store(true);
}

static const char* source_name(sl::ENCODED_STREAM_SOURCE s) {
    switch (s) {
        case sl::ENCODED_STREAM_SOURCE::RECEIVING:
            return "RECEIVING";
        case sl::ENCODED_STREAM_SOURCE::SENDING:
            return "SENDING";
        case sl::ENCODED_STREAM_SOURCE::RECORDING:
            return "RECORDING";
        default:
            return "?";
    }
}

struct SinkState {
    std::ofstream file;
    size_t bytes = 0;
    size_t frames = 0;
    size_t keyframes = 0;
};

static void print_status(sl::Camera& zed) {
    std::cout << "Active encoded sources:\n";
    for (const auto& info : zed.getEncodedStreamsInfo()) {
        std::cout << "  - " << source_name(info.source) << " active=" << (info.active ? "yes" : "no")
                  << " codec=" << (info.codec == sl::STREAMING_CODEC::H264 ? "H264" : "H265") << " bitrate=" << info.bitrate_kbps << " kbps"
                  << (info.is_lossless ? " (lossless)" : "") << "\n";
    }
}

static int run_loop(
    sl::Camera& zed,
    const std::vector<sl::ENCODED_STREAM_SOURCE>& taps,
    std::map<sl::ENCODED_STREAM_SOURCE, SinkState>& sinks,
    int seconds
) {
    const auto t_end = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (!g_stop.load() && std::chrono::steady_clock::now() < t_end) {
        if (zed.grab() != sl::ERROR_CODE::SUCCESS)
            continue;
        for (auto src : taps) {
            sl::EncodedStreamPacket pkt;
            if (zed.retrieveEncodedStreamPacket(pkt, src) != sl::ERROR_CODE::SUCCESS)
                continue;
            auto& s = sinks[src];
            if (!s.file.is_open())
                continue;
            s.file.write(reinterpret_cast<const char*>(pkt.data), static_cast<std::streamsize>(pkt.size));
            s.bytes += pkt.size;
            s.frames++;
            if (pkt.is_keyframe)
                s.keyframes++;
        }
    }
    for (auto& kv : sinks)
        kv.second.file.close();
    return 0;
}

static void
print_results(const std::map<sl::ENCODED_STREAM_SOURCE, SinkState>& sinks, const std::map<sl::ENCODED_STREAM_SOURCE, std::string>& paths) {
    for (const auto& kv : sinks) {
        const auto& p = paths.at(kv.first);
        std::cout << source_name(kv.first) << ": " << kv.second.frames << " frames (" << kv.second.keyframes << " keyframes), "
                  << kv.second.bytes << " bytes → " << p << "\n";
    }
}

static int mode_live(int seconds, bool want_h264) {
    sl::Camera zed;
    sl::InitParameters init;
    init.camera_resolution = sl::RESOLUTION::AUTO;
    init.camera_fps = 30;
    init.depth_mode = sl::DEPTH_MODE::NONE;
    init.sdk_verbose = 0;

    auto err = zed.open(init);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "open(): " << sl::toString(err) << "\n";
        return 1;
    }

    sl::StreamingParameters sp;
    sp.codec = want_h264 ? sl::STREAMING_CODEC::H264 : sl::STREAMING_CODEC::H265;
    sp.port = 30000;
    sp.bitrate = 6000;
    err = zed.enableStreaming(sp);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "enableStreaming: " << sl::toString(err) << "\n";
        zed.close();
        return 1;
    }

    sl::RecordingParameters rp;
    rp.video_filename = "live_record.svo2";
    rp.compression_mode = want_h264 ? sl::SVO_COMPRESSION_MODE::H264 : sl::SVO_COMPRESSION_MODE::H265;
    rp.bitrate = 8000;
    err = zed.enableRecording(rp);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "enableRecording: " << sl::toString(err) << "\n";
        zed.disableStreaming();
        zed.close();
        return 1;
    }

    print_status(zed);

    const std::string ext = want_h264 ? ".h264" : ".hevc";
    std::map<sl::ENCODED_STREAM_SOURCE, std::string> paths = {
        {sl::ENCODED_STREAM_SOURCE::SENDING,   "live_send" + ext  },
        {sl::ENCODED_STREAM_SOURCE::RECORDING, "live_record" + ext},
    };
    std::map<sl::ENCODED_STREAM_SOURCE, SinkState> sinks;
    for (auto& kv : paths)
        sinks[kv.first].file.open(kv.second, std::ios::binary | std::ios::trunc);

    run_loop(zed, {sl::ENCODED_STREAM_SOURCE::SENDING, sl::ENCODED_STREAM_SOURCE::RECORDING}, sinks, seconds);
    print_results(sinks, paths);

    zed.disableRecording();
    zed.disableStreaming();
    zed.close();
    return 0;
}

static int mode_stream(const std::string& ip, unsigned short port, int seconds) {
    sl::Camera zed;
    sl::InitParameters init;
    init.input.setFromStream(ip.c_str(), port);
    init.depth_mode = sl::DEPTH_MODE::NONE;
    init.sdk_verbose = 0;

    auto err = zed.open(init);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "open(): " << sl::toString(err) << "\n";
        return 1;
    }
    print_status(zed);

    std::string recv_ext = ".h264";
    for (const auto& i : zed.getEncodedStreamsInfo())
        if (i.source == sl::ENCODED_STREAM_SOURCE::RECEIVING && i.active)
            recv_ext = (i.codec == sl::STREAMING_CODEC::H264) ? ".h264" : ".hevc";

    std::map<sl::ENCODED_STREAM_SOURCE, std::string> paths = {
        {sl::ENCODED_STREAM_SOURCE::RECEIVING, std::string("recv") + recv_ext},
    };
    std::map<sl::ENCODED_STREAM_SOURCE, SinkState> sinks;
    for (auto& kv : paths)
        sinks[kv.first].file.open(kv.second, std::ios::binary | std::ios::trunc);

    run_loop(zed, {sl::ENCODED_STREAM_SOURCE::RECEIVING}, sinks, seconds);
    print_results(sinks, paths);
    zed.close();
    return 0;
}

static int mode_all(const std::string& ip, unsigned short port, int seconds) {
    sl::Camera zed;
    sl::InitParameters init;
    init.input.setFromStream(ip.c_str(), port);
    init.depth_mode = sl::DEPTH_MODE::NONE;
    init.sdk_verbose = 0;

    auto err = zed.open(init);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "open(): " << sl::toString(err) << "\n";
        return 1;
    }

    // Re-stream on a different port (and pick the opposite codec to make
    // the introspection contrast visible).
    sl::StreamingParameters sp;
    sp.codec = sl::STREAMING_CODEC::H264;
    sp.port = 30100;
    sp.bitrate = 5000;
    err = zed.enableStreaming(sp);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "enableStreaming: " << sl::toString(err) << "\n";
        zed.close();
        return 1;
    }

    // Record with transcode_streaming_input = true so the RECORDING source
    // is its own encoder output (not just a duplicate of RECEIVING).
    sl::RecordingParameters rp;
    rp.video_filename = "all_record.svo2";
    rp.compression_mode = sl::SVO_COMPRESSION_MODE::H265;
    rp.bitrate = 9000;
    rp.transcode_streaming_input = true;
    err = zed.enableRecording(rp);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "enableRecording: " << sl::toString(err) << "\n";
        zed.disableStreaming();
        zed.close();
        return 1;
    }

    print_status(zed);

    // Derive correct file extension per source's actual codec.
    auto ext_for = [&zed](sl::ENCODED_STREAM_SOURCE s) -> std::string {
        for (const auto& i : zed.getEncodedStreamsInfo())
            if (i.source == s && i.active)
                return (i.codec == sl::STREAMING_CODEC::H264) ? ".h264" : ".hevc";
        return ".h264";
    };

    std::map<sl::ENCODED_STREAM_SOURCE, std::string> paths = {
        {sl::ENCODED_STREAM_SOURCE::RECEIVING, std::string("all_recv") + ext_for(sl::ENCODED_STREAM_SOURCE::RECEIVING)  },
        {sl::ENCODED_STREAM_SOURCE::SENDING,   std::string("all_send") + ext_for(sl::ENCODED_STREAM_SOURCE::SENDING)    },
        {sl::ENCODED_STREAM_SOURCE::RECORDING, std::string("all_record") + ext_for(sl::ENCODED_STREAM_SOURCE::RECORDING)},
    };
    std::map<sl::ENCODED_STREAM_SOURCE, SinkState> sinks;
    for (auto& kv : paths)
        sinks[kv.first].file.open(kv.second, std::ios::binary | std::ios::trunc);

    run_loop(
        zed,
        {sl::ENCODED_STREAM_SOURCE::RECEIVING, sl::ENCODED_STREAM_SOURCE::SENDING, sl::ENCODED_STREAM_SOURCE::RECORDING},
        sinks,
        seconds
    );
    print_results(sinks, paths);

    zed.disableRecording();
    zed.disableStreaming();
    zed.close();
    return 0;
}

int main(int argc, char** argv) {
    std::signal(SIGINT, on_sigint);

    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " --live [seconds=10] [H264|H265]\n"
                  << "  " << argv[0] << " --stream <ip:port> [seconds=10]\n"
                  << "  " << argv[0] << " --all <ip:port> [seconds=10]\n";
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "--live") {
        int seconds = (argc > 2) ? std::atoi(argv[2]) : 10;
        bool h264 = (argc > 3) && (std::string(argv[3]) == "H264" || std::string(argv[3]) == "h264");
        return mode_live(seconds, h264);
    }
    if (mode == "--stream" || mode == "--all") {
        if (argc < 3) {
            std::cerr << "missing <ip:port>\n";
            return 1;
        }
        std::string ipport = argv[2];
        auto colon = ipport.find(':');
        if (colon == std::string::npos) {
            std::cerr << "expected ip:port\n";
            return 1;
        }
        std::string ip = ipport.substr(0, colon);
        unsigned short port = static_cast<unsigned short>(std::atoi(ipport.substr(colon + 1).c_str()));
        int seconds = (argc > 3) ? std::atoi(argv[3]) : 10;
        return (mode == "--stream") ? mode_stream(ip, port, seconds) : mode_all(ip, port, seconds);
    }
    std::cerr << "unknown mode " << mode << "\n";
    return 1;
}
