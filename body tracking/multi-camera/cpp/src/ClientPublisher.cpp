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
    init_parameters.coordinate_units = sl::UNIT::METER;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    auto state = zed.open(init_parameters);
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

    // define the body tracking parameters, as the fusion can does the tracking and fitting you don't need to enable them here, unless you need it for your app
    sl::BodyTrackingParameters body_tracking_parameters;
    body_tracking_parameters.detection_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
    body_tracking_parameters.body_format = sl::BODY_FORMAT::BODY_38;
    body_tracking_parameters.enable_body_fitting = false;
    body_tracking_parameters.enable_tracking = false;
    state = zed.enableBodyTracking(body_tracking_parameters);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error: " << state << std::endl;
        return false;
    }

    auto camera_infos = zed.getCameraInformation();

    serial = camera_infos.serial_number;

    auto resolution = camera_infos.camera_configuration.resolution;

    // Define display resolution and check that it fit at least the image resolution
    float image_aspect_ratio = resolution.width / (1.f * resolution.height);
    int requested_low_res_w = std::min(640, (int)resolution.width);
    low_resolution = sl::Resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio);
    
    view.alloc(low_resolution, sl::MAT_TYPE::U8_C4, sl::MEM::GPU);
    point_cloud.alloc(low_resolution, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);

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
    sl::Bodies bodies;
    sl::BodyTrackingRuntimeParameters body_runtime_parameters;
    body_runtime_parameters.detection_confidence_threshold = 40;

    sl::RuntimeParameters runtime_parameters;
    runtime_parameters.confidence_threshold = 30;

    // in this sample we use a dummy thread to process the ZED data.
    // you can replace it by your own application and use the ZED like you use to, retrieve its images, depth, sensors data and so on.
    // as long as you call the grab function and the retrieveBodies (wich run the detection) the camera will be able to seamlessly transmit the data to the fusion module.
    while (running) {
        if (zed.grab(runtime_parameters) == sl::ERROR_CODE::SUCCESS) {
            /*
            Your App
            */            
            zed.retrieveImage(view, sl::VIEW::LEFT, sl::MEM::GPU, low_resolution);
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZBGRA, sl::MEM::GPU, low_resolution);

            // just be sure to run the bodies detection
            zed.retrieveBodies(bodies, body_runtime_parameters);
        }
    }
}
