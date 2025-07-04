#ifndef  __SENDER_RUNNER_HDR__
#define __SENDER_RUNNER_HDR__

#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>

#include <thread>
#include <condition_variable>

struct Trigger{

    void notifyZED(){

        cv.notify_all();

        if(running){
            bool wait_for_zed = true;
            const int nb_zed = states.size();
            while(wait_for_zed){
                int count_r = 0;
                for(auto &it:states)
                    count_r += it.second;
                wait_for_zed = count_r != nb_zed;
                sl::sleep_ms(1);
            }
            for(auto &it:states)
                it.second = false;
        }
    }

    std::condition_variable cv;
    bool running = true;
    std::map<int, bool> states;
};

class ClientPublisher{

public:
    ClientPublisher();
    ~ClientPublisher();

    bool open(sl::InputType, Trigger* ref, int sdk_gpu_id);
    void start();
    void stop();
    void setStartSVOPosition(unsigned pos);

private:
    sl::Camera zed;
    void work();
    std::thread runner;
    int serial;
    std::mutex mtx;
    Trigger *p_trigger;
};

#endif // ! __SENDER_RUNNER_HDR__
