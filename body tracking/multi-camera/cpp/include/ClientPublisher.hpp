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
    void setStartSVOPosition(unsigned pos);

    bool isRunning() {
        return running;
    }

private:
    sl::Camera zed;
    void work();
    std::thread runner;
    bool running;
    int serial;
};

#endif // ! __SENDER_RUNNER_HDR__
