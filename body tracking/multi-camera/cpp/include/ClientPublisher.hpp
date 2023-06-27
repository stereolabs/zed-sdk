#ifndef  __SENDER_RUNNER_HDR__
#define __SENDER_RUNNER_HDR__

#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>

#include <thread>

class ClientPublisher{

public:
    ClientPublisher();
    ~ClientPublisher();

    bool open(sl::InputType);

    void start();
    void stop();

    sl::Mat getViewRef(){
        return view;
    }

    sl::Mat getPointCloufRef(){
        return point_cloud;
    }

    CUstream getStream() {
        return zed.getCUDAStream();
    }

    bool isRunning() {
        return running;
    }

    int getSerial() {
        return serial;
    }


private:
    sl::Resolution low_resolution;
    sl::Mat point_cloud, view;
    sl::Camera zed;
    sl::InitParameters init_parameters;
    void work();
    std::thread runner;
    bool running;
    int serial;
};

#endif // ! __SENDER_RUNNER_HDR__
