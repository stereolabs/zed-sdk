#ifndef  __FUSION_RUNNER_HDR__
#define __FUSION_RUNNER_HDR__

#include <sl/Fusion.hpp>

#include <thread>

class FusionRunner {

public:
    FusionRunner();
    ~FusionRunner();

    bool start(std::vector<sl::FusionConfiguration>&);
    void stop();

private:
    sl::Fusion fusion;
    sl::InitFusionParameters init_params;
    void work();
    std::thread runner;
    bool running;

    std::vector<sl::CameraIdentifier> cameras;
};

#endif // ! __FUSION_RUNNER_HDR__
