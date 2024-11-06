#include "ClientPublisher.hpp"

ClientPublisher::ClientPublisher()
{
}

ClientPublisher::~ClientPublisher()
{
    zed.close();
}

bool ClientPublisher::open(const sl::InputType& input, Trigger* ref) {
    // already running
    if (runner.joinable())
        return false;

    p_trigger = ref;

    sl::InitParameters init_parameters;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_parameters.input = input;
    init_parameters.coordinate_units = sl::UNIT::METER;
    init_parameters.depth_stabilization = 5;
    auto state = zed.open(init_parameters);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error: " << state << std::endl;
        return false;
    } 

    serial = zed.getCameraInformation().serial_number;
    p_trigger->states[serial] = false;

    // in most cases in object detection setup, the cameras are static
    sl::PositionalTrackingParameters positional_tracking_parameters;
    positional_tracking_parameters.set_as_static = true;
    state = zed.enablePositionalTracking(positional_tracking_parameters);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error: " << state << std::endl;
        return false;
    }

    sl::ObjectDetectionParameters object_detection_parameters;
    object_detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_FAST;
    object_detection_parameters.enable_tracking = true;
    object_detection_parameters.fused_objects_group_name = "MULTI_CLASS_BOX";
    state = zed.enableObjectDetection(object_detection_parameters);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error: " << state << std::endl;
        return false;
    }

    return true;
}

void ClientPublisher::start()
{
    if (zed.isOpened()) {
        // the camera should stream its data so the fusion can subscibe to it to gather the detected body and others metadata needed for the process.
        zed.startPublishing();
        // the thread can start to process the camera grab in background
        runner = std::thread(&ClientPublisher::work, this);
    }
}

void ClientPublisher::stop()
{
    if (runner.joinable())
        runner.join();
    zed.close();
}

void ClientPublisher::work()
{
    sl::RuntimeParameters rt;
    rt.confidence_threshold = 50;

    sl::Objects objects;

    // in this sample we use a dummy thread to process the ZED data.
    // you can replace it by your own application and use the ZED like you use to, retrieve its images, depth, sensors data and so on.
    // as long as you call the grab function and the retrieveObjects (which runs the detection) the camera will be able to seamlessly transmit the data to the fusion module.
    while (p_trigger->running) {
        std::unique_lock<std::mutex> lk(mtx);
        p_trigger->cv.wait(lk);
        if (p_trigger->running) {
            if (zed.grab(rt) == sl::ERROR_CODE::SUCCESS) {
                /*
                Your App
                */
                zed.retrieveObjects(objects, sl::ObjectDetectionRuntimeParameters());
            }
        }
        p_trigger->states[serial] = true;
    }
}

void ClientPublisher::setStartSVOPosition(unsigned pos) {
    zed.setSVOPosition(pos);
}
