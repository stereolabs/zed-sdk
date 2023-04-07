#include "SenderRunner.hpp"


#include "global.hpp"

SenderRunner::SenderRunner() : running(false)
{
    init_params.coordinate_units = UNIT_SYS;
    init_params.coordinate_system = COORD_SYS;
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_params.sdk_verbose = 1;
}

SenderRunner::~SenderRunner()
{
    zed.close();
}

bool SenderRunner::open(sl::FusionConfiguration z_input) {
    // already running
    if (runner.joinable())
        return false;

    zed_config = z_input;
    init_params.input = zed_config.input_type;
    auto state = zed.open(init_params);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error open Camera " << state << std::endl;
        return false;
    }

    sl::PositionalTrackingParameters ptp;
    ptp.set_as_static = true;
    state = zed.enablePositionalTracking(ptp);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error enable Positional Tracking" << state << std::endl;
        return false;
    }


    sl::BodyTrackingParameters odp;
    odp.detection_model = BODY_MODEL;
    odp.body_format = BODY_FORMAT;
    odp.enable_body_fitting = false;
    odp.enable_tracking = false;
    state = zed.enableBodyTracking(odp);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error enable Object Detection " << state << std::endl;
        return false;
    }
    
    return true;
}

void SenderRunner::start()
{
    if (zed.isOpened()) {
        running = true;
        zed.startPublishing(zed_config.communication_parameters);
        runner = std::thread(&SenderRunner::work, this);
    }
}

void SenderRunner::stop()
{
    running = false;
    if (runner.joinable())
        runner.join();
    zed.close();
}


void SenderRunner::work()
{
    sl::Bodies local_sks;
    sl::BodyTrackingRuntimeParameters body_runtime_parameters;
    body_runtime_parameters.detection_confidence_threshold = 40;
    while (running)
    {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS)
        {
            zed.retrieveBodies(local_sks, body_runtime_parameters);
        }
    }
}
