#include "FusionRunner.hpp"

#include "global.hpp"

FusionRunner::FusionRunner() : running(false) {
    init_params.coordinate_units = UNIT_SYS;
    init_params.coordinate_system = COORD_SYS;
    init_params.output_performance_metrics = true;
    init_params.verbose = true;
}

FusionRunner::~FusionRunner() {
    fusion.close();
}

bool FusionRunner::start(std::vector<sl::FusionConfiguration>& z_inputs) {
    // already running
    if (runner.joinable()) return false;

    fusion.init(init_params);

    // connect to every cameras
    for (auto& it : z_inputs) {
        sl::CameraIdentifier uuid;
        uuid.sn = it.serial_number;
        auto state = fusion.subscribe(uuid, it.communication_parameters, it.pose);
        if (state != sl::FUSION_ERROR_CODE::SUCCESS)
            std::cout << "Unable to subscribe to " << std::to_string(uuid.sn) << " . " << state << std::endl;
        else
            cameras.push_back(uuid);
    }

    if (cameras.empty()) 
        return false;    

    runner = std::thread(&FusionRunner::work, this);
    return true;
}

void FusionRunner::stop() {
    running = false;
    if (runner.joinable()) runner.join();
    fusion.close();
}

void FusionRunner::work() {

    sl::BodyTrackingFusionParameters body_fusion_init_params;
    body_fusion_init_params.enable_tracking = true;
    body_fusion_init_params.enable_body_fitting = true; // skeletons will looks more natural but requires more computations
    fusion.enableBodyTracking(body_fusion_init_params);

    running = true;

    // define fusion behavior    
    sl::BodyTrackingFusionRuntimeParameters rt;
    rt.skeleton_minimum_allowed_keypoints = 7;

    auto ptr_data = SharedData::getInstance();

    while (running) {
        if (fusion.process() == sl::FUSION_ERROR_CODE::SUCCESS) {

            {
                const std::lock_guard<std::mutex> lock_m(ptr_data->bodiesData.mtx);
                // Retrieve detected objects
                fusion.retrieveBodies(ptr_data->bodiesData.bodies, rt);
                // for debug, you can retrieve the data send by each camera, as well as communication and process stat just to make sure everything is okay
                for (auto &id : cameras)
                    fusion.retrieveBodies(ptr_data->bodiesData.singledata[id], rt, id); 
            }

            if(init_params.output_performance_metrics)
                fusion.getProcessMetrics(ptr_data->metrics);
        }
    }
}