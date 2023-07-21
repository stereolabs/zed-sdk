#include "main_fun.cpp" 
#include <sl/Camera.hpp>

// #include <rclcpp/rclcpp.hpp>
// #include <std_msgs/msg/float32.hpp>
// #include <std_msgs/msg/float32_multi_array.hpp>

int getOCVtype(sl::MAT_TYPE type);

int getOCVtype(sl::MAT_TYPE type)
{
    int cv_type = -1;
    switch (type) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
}

std::vector<sl::uint2> cvt(const cv::Rect &bbox_in){
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x, bbox_in.y);
    bbox_out[1] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y);
    bbox_out[2] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y + bbox_in.height);
    bbox_out[3] = sl::uint2(bbox_in.x, bbox_in.y + bbox_in.height);
    return bbox_out;
}

cv::Mat slMat2cvMat(sl::Mat& input);
cv::Mat slMat2cvMat(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(sl::MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

int main(int argc, char *argv[]) {
    // rclcpp::init(argc, argv);
    // auto node = rclcpp::Node::make_shared("yolov8_trt");
    // auto pub_cv_pose = node->create_publisher<std_msgs::msg::Float32MultiArray>("cv_pose", 10);

    sl::Camera zed;
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_parameters.coordinate_units = sl::UNIT::METER;

    // Open the camera
    sl::ERROR_CODE err = zed.open(init_parameters);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Failed to open the ZED camera. Error code: " << err << std::endl;
        return 1;
    }

    err = zed.enablePositionalTracking();
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Failed to enable positional tracking. Error code: " << err << std::endl;
        zed.close();
        return 1;
    }

    // Custom OD
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_segmentation = true; // designed to give person pixel mask
    detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    err = zed.enableObjectDetection(detection_parameters);
    if (err != sl::ERROR_CODE::SUCCESS) {
        // print("enableObjectDetection", err, "\nExit program.");
        std::cout << "Failed to enableObjectDetection. Error code: " << err << std::endl;
        zed.close();
        return EXIT_FAILURE;
    }





    // sl::Mat image, point_cloud;
    int i = 0;
    sl::Mat left_sl, depth_image, point_cloud;
    cv::Mat left_cv_bgra, left_cv_bgr;
    sl::Pose pose;
    sl::Plane plane;
    sl::Mesh mesh;
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;
    // cv::Mat left_cv_bgra, left_cv_bgr;
    float tolerance = 0.01f;

    cudaSetDevice(kGpuId);
    std::string wts_name = "";
    std::string engine_name = "";
    std::string img_dir;
    std::string sub_type = "";
    std::string cuda_post_process="";
    int model_bboxes;

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, sub_type, cuda_post_process)) {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolov8_trt -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov8_trt -d [.engine] ../samples  [c/g]// deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        serialize_engine(wts_name, engine_name, sub_type);
        return 0;
    }

    // Deserialize the engine from file
    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    auto out_dims = engine->getBindingDimensions(1);
    model_bboxes = out_dims.d[0];
    // Prepare cpu and gpu buffers
    float *device_buffers[2];
    float *output_buffer_host = nullptr;
    float *decode_ptr_host=nullptr;
    float *decode_ptr_device=nullptr;

    // Read images from directory
    // std::vector<std::string> file_names;
    // if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
    //     std::cerr << "read_files_in_dir failed." << std::endl;
    //     return -1;
    // }

    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host, &decode_ptr_device, cuda_post_process);


    // batch predict
    while(zed.grab() == sl::ERROR_CODE::SUCCESS) 
    {
        zed.retrieveImage(left_sl, sl::VIEW::LEFT);
        zed.retrieveImage(depth_image, sl::VIEW::DEPTH);
        zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA);

        // Preparing inference
        left_cv_bgra = slMat2cvMat(left_sl);
        cv::cvtColor(left_cv_bgra, left_cv_bgr, cv::COLOR_BGRA2BGR);

        if (left_cv_bgr.empty()) continue;

        // Get a batch of images
        std::vector<cv::Mat> img_batch {left_cv_bgr};

        // auto message = std::make_shared<std_msgs::msg::Float32MultiArray>();
        // message->data = {1.1, 2.2, 3.3, 4.4};
        // pub_cv_pose->publish(*message);
        // std::vector<std::string> img_name_batch;
        // for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
        //     cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
        //     img_batch.push_back(img);
        //     img_name_batch.push_back(file_names[j]);
        // }
        // Preprocess
        cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
        // Run inference
        infer(*context, stream, (void **)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host, decode_ptr_device, model_bboxes, cuda_post_process);
        std::vector<std::vector<Detection>> res_batch;
        if (cuda_post_process == "c") {
            // NMS
            batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
        } else if (cuda_post_process == "g") {
            //Process gpu decode and nms results
            batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
        }
        // Draw bounding boxes
        draw_bbox(img_batch, res_batch);
        // Preparing for ZED SDK ingesting
        for (size_t j = 0; j < img_batch.size(); j++) {
            auto &res = res_batch[i];
            std::vector<sl::CustomBoxObjectData> objects_in;
            for (auto &it : res) {
                sl::CustomBoxObjectData tmp;
                cv::Rect r = get_rect(left_cv_bgr, it.bbox);
                // Fill the detections into the correct format
                tmp.unique_object_id = sl::generate_unique_id();
                tmp.probability = it.conf;
                tmp.label = (int) it.class_id;
                tmp.bounding_box_2d = cvt(r);
                tmp.is_grounded = ((int) it.class_id == 0); // Only the first class (person) is grounded, that is moving on the floor plane
                // others are tracked in full 3D space                
                objects_in.push_back(tmp);
            }
            // Send the custom detected boxes to the ZED
            zed.ingestCustomBoxObjects(objects_in);
            
        }

        // Retrieve the tracked objects, with 2D and 3D attributes
        zed.retrieveObjects(objects, objectTracker_parameters_rt);
        // for(auto object : objects.object_list)
        // {
        //     std::cout << object.id << std::endl;
        // }
        std::cout << objects.object_list.size() << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
            


        
        // Send the custom detected boxes to the ZED
        
        // for (size_t j = 0; j < img_batch.size(); j++) {
        //     cv::imshow("detected images", img_batch.at(j));
            
        // }
        int key = cv::waitKey(1);
        if (key == 'q')
        {
            std::cout << "q key is pressed by the user. Stopping the video" << std::endl;
            break;
        }

        
        // Save images
        // for (size_t j = 0; j < img_batch.size(); j++) {
        //     cv::imwrite("_" + img_name_batch[j], img_batch[j]);
        // }
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    CUDA_CHECK(cudaFree(decode_ptr_device));
    delete[] decode_ptr_host;
    delete[] output_buffer_host;
    cuda_preprocess_destroy();
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;
    cv::destroyAllWindows();
    zed.close();
    

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < kOutputSize; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;
    // rclcpp::spin(node);
    return 0;
}

