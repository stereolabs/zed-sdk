#include "ClientPublisher.hpp"

ClientPublisher::ClientPublisher() : running(false)
{
}

ClientPublisher::~ClientPublisher()
{
    zed.close();
}

bool ClientPublisher::open(sl::InputType input) {
    // already running
    if (runner.joinable())
        return false;

    sl::InitParameters init_parameters;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_parameters.input = input;
    if (input.getType() == sl::InputType::INPUT_TYPE::SVO_FILE)
        init_parameters.svo_real_time_mode = true;
    init_parameters.coordinate_units = sl::UNIT::METER;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    auto state = zed.open(init_parameters);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error: " << state << std::endl;
        return false;
    }


    // define the body tracking parameters, as the fusion can does the tracking and fitting you don't need to enable them here, unless you need it for your app
    sl::BodyTrackingParameters body_tracking_parameters;
    body_tracking_parameters.detection_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
    body_tracking_parameters.body_format = sl::BODY_FORMAT::BODY_18;
    body_tracking_parameters.enable_body_fitting = false;
    body_tracking_parameters.enable_tracking = false;
    state = zed.enableBodyTracking(body_tracking_parameters);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error: " << state << std::endl;
        return false;
    }    

    // define the body tracking parameters, as the fusion can does the tracking and fitting you don't need to enable them here, unless you need it for your app
    sl::ObjectDetectionParameters object_detection_parameters;
    object_detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_ACCURATE;
    object_detection_parameters.enable_tracking = false;
    object_detection_parameters.instance_module_id = 20;
    state = zed.enableObjectDetection(object_detection_parameters);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error: " << state << std::endl;
        return false;
    }

    
    // in most cases in body tracking setup, the cameras are static
    sl::PositionalTrackingParameters positional_tracking_parameters;
    // in most cases for body detection application the camera is static:
    positional_tracking_parameters.set_as_static = true;
    state = zed.enablePositionalTracking(positional_tracking_parameters);
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
        running = true;
        // the camera should stream its data so the fusion can subscibe to it to gather the detected body and others metadata needed for the process.
        zed.startPublishing();
        // the thread can start to process the camera grab in background
        runner = std::thread(&ClientPublisher::work, this);
    }
}

void ClientPublisher::stop()
{
    running = false;
    if (runner.joinable())
        runner.join();
    zed.close();
}

void ClientPublisher::work()
{
    // in this sample we use a dummy thread to process the ZED data.
    // you can replace it by your own application and use the ZED like you use to, retrieve its images, depth, sensors data and so on.
    // as long as you call the grab function and the retrieveObjects (which runs the detection) the camera will be able to seamlessly transmit the data to the fusion module.
    while (running) {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            /*
            Your App

            */
            zed.retrieveObjects(objects, sl::ObjectDetectionRuntimeParameters(), 20);
            zed.retrieveBodies(bodies);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));

    }
}

void ClientPublisher::setStartSVOPosition(unsigned pos) {
    zed.setSVOPosition(pos);
    zed.grab();
}

sl::Objects ClientPublisher::getObjects(){
    return objects;
}

