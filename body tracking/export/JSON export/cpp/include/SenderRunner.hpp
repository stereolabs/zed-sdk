#ifndef  __SENDER_RUNNER_HDR__
#define __SENDER_RUNNER_HDR__

#include <sl/Fusion.hpp>
#include <sl/Camera.hpp>

#include <thread>

class SenderRunner {

public:
    SenderRunner();
    ~SenderRunner();

    bool open(sl::FusionConfiguration);
    void start();
    void stop();

private:
    sl::Camera zed;
    sl::InitParameters init_params;
    void work();
    std::thread runner;
    bool running;
    sl::FusionConfiguration zed_config;
};

#endif // ! __SENDER_RUNNER_HDR__
