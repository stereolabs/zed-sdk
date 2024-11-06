#include "ClientPublisher.hpp"

ClientPublisher::ClientPublisher() {
}

ClientPublisher::~ClientPublisher() {
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
    if (state != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Error: " << state << std::endl;
        return false;
    }

    serial = zed.getCameraInformation().serial_number;
    p_trigger->states[serial] = false;

    // In most cases in object detection setup, the cameras are static
    sl::PositionalTrackingParameters positional_tracking_parameters;
    positional_tracking_parameters.set_as_static = true;
    state = zed.enablePositionalTracking(positional_tracking_parameters);
    if (state != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Error: " << state << std::endl;
        return false;
    }

    object_detection_parameters.clear();
    object_detection_runtime_parameters.clear();

    // Parameters for a MULTI CLASS object detection model instance
    sl::ObjectDetectionParameters param_instance_0;
    param_instance_0.instance_module_id = 0;
    param_instance_0.detection_model = sl::OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_FAST;
    param_instance_0.enable_tracking = true;
    param_instance_0.fused_objects_group_name = "MULTI_CLASS_BOX";
    sl::ObjectDetectionRuntimeParameters runtime_param_instance_0;
    runtime_param_instance_0.detection_confidence_threshold = 20;

    object_detection_parameters.insert({param_instance_0.instance_module_id, param_instance_0});
    object_detection_runtime_parameters.insert({param_instance_0.instance_module_id, runtime_param_instance_0});

    // Parameters for a PERSON HEAD object detection model instance
    sl::ObjectDetectionParameters param_instance_1;
    param_instance_1.instance_module_id = 1;
    param_instance_1.detection_model = sl::OBJECT_DETECTION_MODEL::PERSON_HEAD_BOX_FAST;
    param_instance_1.enable_tracking = true;
    param_instance_1.fused_objects_group_name = "PERSON_HEAD_BOX";
    sl::ObjectDetectionRuntimeParameters runtime_param_instance_1;
    runtime_param_instance_1.detection_confidence_threshold = 20;

    object_detection_parameters.insert({param_instance_1.instance_module_id, param_instance_1});
    object_detection_runtime_parameters.insert({param_instance_1.instance_module_id, runtime_param_instance_1});

    // Parameters for CUSTOM object detection model instance
    if (!optional_custom_onnx_yolo_model.empty()) {
        sl::ObjectDetectionParameters param_instance_2;
        param_instance_2.instance_module_id = 2;
        param_instance_2.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_YOLOLIKE_BOX_OBJECTS;
        param_instance_2.custom_onnx_file = optional_custom_onnx_yolo_model.c_str();
        param_instance_2.custom_onnx_dynamic_input_shape = sl::Resolution(512, 512);
        param_instance_2.enable_tracking = true;
        param_instance_2.fused_objects_group_name = "MY_CUSTOM_OUTPUT";
        sl::CustomObjectDetectionRuntimeParameters runtime_param_instance_2;
        runtime_param_instance_2.object_detection_properties.detection_confidence_threshold = 20;
        // runtime_param_instance_2.object_detection_properties.is_static = true;
        // runtime_param_instance_2.object_detection_properties.tracking_timeout = 100.f;

        object_detection_parameters.insert({param_instance_2.instance_module_id, param_instance_2});
        custom_object_detection_runtime_parameters.insert({param_instance_2.instance_module_id, runtime_param_instance_2});
    }

    // Enable the object detection instances
    for (auto const& params : object_detection_parameters) {
        state = zed.enableObjectDetection(params.second);
        std::cout << "Enabling " << params.second.detection_model << " " << state << std::endl;
        if (state != sl::ERROR_CODE::SUCCESS) {
            std::cout << "Error: " << state << std::endl;
            return false;
        }
    }

    return true;
}

void ClientPublisher::start() {
    if (zed.isOpened()) {
        // the camera should stream its data so the fusion can subscribe to it to gather the detected body and others metadata needed for the process.
        zed.startPublishing();
        // the thread can start to process the camera grab in background
        runner = std::thread(&ClientPublisher::work, this);
    }
}

void ClientPublisher::stop() {
    if (runner.joinable())
        runner.join();
    zed.close();
}

void ClientPublisher::work() {
    sl::RuntimeParameters rt;
    rt.confidence_threshold = 50;

    // In this sample we use a dummy thread to process the ZED data.
    // You can replace it by your own application and use the ZED like you use to, retrieve its images, depth, sensors data and so on.
    // as long as you call the grab function and the retrieveObjects (which runs the detection) the camera will be able to seamlessly transmit the data to the fusion module.
    while (p_trigger->running) {
        std::unique_lock<std::mutex> lk(mtx);
        p_trigger->cv.wait(lk);
        if (p_trigger->running) {
            if (zed.grab(rt) == sl::ERROR_CODE::SUCCESS) {
                /*
                Your App

                Here, as an example, we retrieve the enabled object detection models, but you can retrieve as many as you want and they will seamllessly be transmitted to the fusion module.
                It is important to note that for synchronisation purposes, at the moment, the retrieveObjects runs the detection and the sending happens only at the beginning of the next grab.
                That way, you can decide to only send to fusion a subset of the possible detection results by only calling a subset of the possible retrieveObjects.
                 */
                for (auto const& odp : object_detection_runtime_parameters)
                    zed.retrieveObjects(objects, odp.second, odp.first);
                for (auto const& codp : custom_object_detection_runtime_parameters)
                    zed.retrieveObjects(objects, codp.second, codp.first);
            }
        }
        p_trigger->states[serial] = true;
    }
}

void ClientPublisher::setStartSVOPosition(unsigned pos) {
    zed.setSVOPosition(pos);
}

sl::Objects ClientPublisher::getObjects() const {
    return objects;
}

