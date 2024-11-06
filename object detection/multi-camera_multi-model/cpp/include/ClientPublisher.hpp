#ifndef  __SENDER_RUNNER_HDR__
#define __SENDER_RUNNER_HDR__

#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>

#include <thread>
#include <condition_variable>
#include <unordered_map>

struct Trigger{

    void notifyZED() {
        cv.notify_all();
        if (running) {
            bool wait_for_zed = true;
            size_t const nb_zed{states.size()};
            while (wait_for_zed) {
                int count_r = 0;
                for (auto const& it : states)
                    count_r += it.second;
                wait_for_zed = count_r != nb_zed;
                sl::sleep_ms(1);
            }
            for (auto &it : states)
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

    bool open(const sl::InputType& input, Trigger* ref);
    void start();
    void stop();
    void setStartSVOPosition(const unsigned pos);
    sl::Objects getObjects() const;

    std::string optional_custom_onnx_yolo_model;

private:
    sl::Camera zed;
    void work();
    std::thread runner;
    int serial;
    std::mutex mtx;
    Trigger *p_trigger;
    sl::Objects objects;

    std::unordered_map<int, sl::ObjectDetectionParameters> object_detection_parameters;
    std::unordered_map<int, sl::ObjectDetectionRuntimeParameters> object_detection_runtime_parameters;
    std::unordered_map<int, sl::CustomObjectDetectionRuntimeParameters> custom_object_detection_runtime_parameters;
};

#endif // ! __SENDER_RUNNER_HDR__
