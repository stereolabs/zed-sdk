#include "ClientPublisher.hpp"

ClientPublisher::ClientPublisher() { }

ClientPublisher::~ClientPublisher()
{
    zed.close();
}

bool ClientPublisher::open(sl::InputType input, Trigger* ref) {

    p_trigger = ref;

    sl::InitParameters init_parameters;
    init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL;
    init_parameters.input = input;
    init_parameters.coordinate_units = sl::UNIT::METER;
    init_parameters.depth_stabilization = 10;
    auto state = zed.open(init_parameters);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error: " << state << std::endl;
        return false;
    }

    serial = zed.getCameraInformation().serial_number;
    p_trigger->states[serial] = false;

    // in most cases in body tracking setup, the cameras are static
    sl::PositionalTrackingParameters positional_tracking_parameters;


    std::string config_file_path(input.getConfiguration().c_str());
    size_t found = config_file_path.find_last_of("/\\");
    std::string config_dir = config_file_path.substr(0, found);

    auto roi_path = config_dir + "/Mask_"+std::to_string(serial)+"_Left.png";
    sl::Mat roi;
    roi.read(roi_path.c_str());
    if(roi.isInit())
        zed.setRegionOfInterest(roi);
    
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
    sl::Bodies bodies;
    sl::BodyTrackingRuntimeParameters body_runtime_parameters;
    body_runtime_parameters.detection_confidence_threshold = 40;

    sl::RuntimeParameters rt;
    rt.confidence_threshold = 50;

    // in this sample we use a dummy thread to process the ZED data.
    // you can replace it by your own application and use the ZED like you use to, retrieve its images, depth, sensors data and so on.
    // as long as you call the grab function and the retrieveBodies (which runs the detection) the camera will be able to seamlessly transmit the data to the fusion module.
    while (p_trigger->running) {
        std::unique_lock<std::mutex> lk(mtx);
        p_trigger->cv.wait(lk);
        if(p_trigger->running){
         if(zed.grab(rt) == sl::ERROR_CODE::SUCCESS){
            // just be sure to run the bodies detection
         }
        }
        p_trigger->states[serial] = true;
    }
}

void ClientPublisher::setStartSVOPosition(unsigned pos) {
    zed.setSVOPosition(pos);
}
