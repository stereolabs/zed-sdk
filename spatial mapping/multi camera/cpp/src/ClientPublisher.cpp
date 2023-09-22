#include "ClientPublisher.hpp"

ClientPublisher::ClientPublisher() : running(false)
{
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
}

ClientPublisher::~ClientPublisher()
{
    zed.close();
}

bool ClientPublisher::open(sl::InputType input) {
    // already running
    if (runner.joinable())
        return false;

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

    sl::PositionalTrackingParameters positional_tracking_parameters; 
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
    // as long as you call the grab function and the retrieveBodies (wich run the detection) the camera will be able to seamlessly transmit the data to the fusion module.

    // Setup runtime parameters
    sl::RuntimeParameters runtime_parameters;
    // Use low depth confidence to avoid introducing noise in the constructed model
    runtime_parameters.confidence_threshold = 50;

    while (running) {
        if (zed.grab(runtime_parameters) == sl::ERROR_CODE::SUCCESS) {
           
        }
    }
}

void ClientPublisher::setStartSVOPosition(unsigned pos) {
    zed.setSVOPosition(pos);
    zed.grab();
}


