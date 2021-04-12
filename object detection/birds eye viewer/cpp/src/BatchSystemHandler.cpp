#include "BatchSystemHandler.hpp"

BatchSystemHandler::BatchSystemHandler(int data_retention_time) {
  batch_data_retention = data_retention_time;
}

BatchSystemHandler::~BatchSystemHandler()
{
   clear();
}

void BatchSystemHandler::clear() {
    objects_tracked_queue.clear();
    camWorldPoseMap_ms.clear();
    camLocalPoseMap_ms.clear();

    while(!imageMap_ms.empty()) {
        std::map<unsigned long long,sl::Mat>::iterator it  = imageMap_ms.begin();
        it->second.free();
        imageMap_ms.erase(it);
    }

    while(!depthMap_ms.empty()) {
        std::map<unsigned long long,sl::Mat>::iterator it  = depthMap_ms.begin();
        it->second.free(sl::MEM::GPU);
        depthMap_ms.erase(it);
    }
}

///
/// \brief push: push data in the FIFO system
/// \param local_pose_ : current pose of the camera in Camera reference frame
/// \param world_pose_ : current pose of the camera in World reference frame
/// \param image_ : as sl::Mat (on CPU memory) to be stored
/// \param pc_ : point cloud as sl::Mat (on GPU memory) to be stored
/// \param batch_ : batch_ from ZED SDK batching system
///
void BatchSystemHandler::push(sl::Pose local_pose_, sl::Pose world_pose_, sl::Mat image_, sl::Mat pc_, std::vector<sl::ObjectsBatch> &batch_) {
    ingestWorldPoseInMap(world_pose_);
    ingestLocalPoseInMap(local_pose_);
#if WITH_IMAGE_RETENTION
    ingestImageInMap(world_pose_.timestamp,image_);
    ingestDepthInMap(world_pose_.timestamp,pc_);
#endif
    ingestInObjectsQueue(batch_);
}

///
/// \brief pop : pop the data from the FIFO system
/// \param local_pose_ : pose at the sl::Objects timestamp in camera reference frame
/// \param world_pose_ : pose at the sl::Objects timestamp in world reference frame
/// \param image_ : as sl::Mat (on CPU memory) at the sl::Objects timestamp
/// \param depth_ : point cloud as sl::Mat (on GPU memory) at the sl::Objects timestamp
/// \param objects_ : sl::Objects in the past.
///
void BatchSystemHandler::pop(sl::Pose& local_pose_,sl::Pose& world_pose_,sl::Mat& image_, sl::Mat& depth_,sl::Objects& objects_) {
    memset(&objects_,0,sizeof(sl::Objects));
    memset(&local_pose_,0,sizeof(sl::Pose));
    memset(&world_pose_,0,sizeof(sl::Pose));

    if (objects_tracked_queue.size()>0) {
        sl::Objects tracked_merged_obj = objects_tracked_queue.front();
        if (init_queue_ts.data_ns==0ULL)
            init_queue_ts = tracked_merged_obj.timestamp;

        unsigned long long targetTS_ms = tracked_merged_obj.timestamp.getMilliseconds();

        local_pose_ =  findClosestLocalPoseFromTS(targetTS_ms);
        world_pose_ =  findClosestWorldPoseFromTS(targetTS_ms);

        #if WITH_IMAGE_RETENTION
        sl::Mat tmp_image=findClosestImageFromTS(targetTS_ms);
        tmp_image.copyTo(image_);
        tmp_image.free();
        imageMap_ms[targetTS_ms].free();
        imageMap_ms.erase(targetTS_ms);

        sl::Mat tmp_depth=findClosestDepthFromTS(targetTS_ms);
        tmp_depth.copyTo(depth_,sl::COPY_TYPE::GPU_GPU);
        tmp_depth.free();
        depthMap_ms[targetTS_ms].free(sl::MEM::GPU);
        depthMap_ms.erase(targetTS_ms);
        #endif
        objects_ = tracked_merged_obj;
        objects_tracked_queue.pop_front();
    }
}

///
/// \brief push: push data in the FIFO system. Overloaded fct for objects data only
/// \param batch_ : batch_ from ZED SDK batching system
///
void BatchSystemHandler::push(std::vector<sl::ObjectsBatch> &batch_) {
    ingestInObjectsQueue(batch_);
}

///
/// \brief pop : pop the data from the FIFO system. Overloaded fct for objects data only
/// \param objects_ : sl::Objects in the past.
///
void BatchSystemHandler::pop(sl::Objects& objects_) {
    memset(&objects_,0,sizeof(sl::Objects));
    if (objects_tracked_queue.size()) {
        sl::Objects tracked_merged_obj = objects_tracked_queue.front();
        objects_ = tracked_merged_obj;
        objects_tracked_queue.pop_front();
    }
}

///
/// \brief ingestWorldPoseInMap
/// \param pose : sl::Pose of the camera in world reference frame
///
void BatchSystemHandler::ingestWorldPoseInMap(sl::Pose pose) {
    std::map<unsigned long long,sl::Pose>::iterator it = camWorldPoseMap_ms.begin();
    sl::Timestamp ts = pose.timestamp;
    if (init_app_ts.data_ns==0ULL)
        init_app_ts =  ts;

    for(auto it = camWorldPoseMap_ms.begin(); it != camWorldPoseMap_ms.end(); ) {
        if(it->first<ts.getMilliseconds() - (unsigned long long)batch_data_retention*1000)
            it = camWorldPoseMap_ms.erase(it);
        else
            ++it;
    }

    camWorldPoseMap_ms[ts.getMilliseconds()]=pose;
}

///
/// \brief ingestLocalPoseInMap
/// \param pose : sl::Pose of the camera in camera reference frame
///
void BatchSystemHandler::ingestLocalPoseInMap(sl::Pose pose) {
    std::map<unsigned long long,sl::Pose>::iterator it = camLocalPoseMap_ms.begin();
    sl::Timestamp ts = pose.timestamp;
    if (init_app_ts.data_ns==0ULL)
        init_app_ts =  ts;

    for(auto it = camLocalPoseMap_ms.begin(); it != camLocalPoseMap_ms.end(); ) {
        if(it->first<ts.getMilliseconds() - (unsigned long long)batch_data_retention*1000)
            it = camLocalPoseMap_ms.erase(it);
        else
            ++it;
    }

    camLocalPoseMap_ms[ts.getMilliseconds()]=pose;
}

void BatchSystemHandler::ingestImageInMap(sl::Timestamp ts, sl::Mat &image) {
    imageMap_ms[ts.getMilliseconds()].clone(image);
    for(auto it = imageMap_ms.begin(); it != imageMap_ms.end(); ) {
        if(it->first<ts.getMilliseconds() - (unsigned long long)batch_data_retention*1000*2)
        {
            it->second.free();
            it = imageMap_ms.erase(it);
        }
        else
            ++it;
    }
}

void BatchSystemHandler::ingestDepthInMap(sl::Timestamp ts, sl::Mat &depth) {
    depthMap_ms[ts.getMilliseconds()].clone(depth);
    for(auto it = depthMap_ms.begin(); it != depthMap_ms.end(); ) {
        if(it->first<ts.getMilliseconds() - (unsigned long long)batch_data_retention*1000*2) {
            it->second.free(sl::MEM::GPU);
            it = depthMap_ms.erase(it);
        }
        else
            ++it;
    }
}

///
/// \brief findClosestPoseFromTS : find the sl::Pose that matched the given timestamp
/// \param timestamp in milliseconds. ( at least in the same unit than camPoseMap_ms)
/// \return sl::Pose found.
///
sl::Pose BatchSystemHandler::findClosestWorldPoseFromTS(unsigned long long timestamp) {
    sl::Pose pose = sl::Pose();
    unsigned long long ts_found = 0;
    if (camWorldPoseMap_ms.find(timestamp)!=camWorldPoseMap_ms.end()) {
        ts_found = timestamp;
        pose = camWorldPoseMap_ms[timestamp];
    }
    return pose;
}

sl::Pose BatchSystemHandler::findClosestLocalPoseFromTS(unsigned long long timestamp) {
    sl::Pose pose = sl::Pose();
    unsigned long long ts_found = 0;
    if (camLocalPoseMap_ms.find(timestamp)!=camLocalPoseMap_ms.end()) {
        ts_found = timestamp;
        pose = camLocalPoseMap_ms[timestamp];
    }
    return pose;
}

///
/// \brief ingestInObjectsQueue : convert a list of batched objects from SDK getObjectsBatch() to a sorted list of sl::Objects
/// \n Use this function to fill a std::deque<sl::Objects> that can be considered and used as a stream of objects with a delay.
/// \param batch_ from getObjectsBatch()
///
void BatchSystemHandler::ingestInObjectsQueue(std::vector<sl::ObjectsBatch> &batch_) {
    // If list is empty, do nothing.
    if (batch_.empty())
        return;

    // add objects in map with timestamp as a key.
    // This ensure
    std::map<uint64_t,sl::Objects> list_of_newobjects;
    for(auto &current_traj: batch_) {

        // Impossible (!!) but still better to check...
        if (current_traj.timestamps.size()!=current_traj.positions.size())
            continue;

        //For each sample, construct a objetdata and put it in the corresponding sl::Objects
        for (int j=0;j<current_traj.timestamps.size();j++) {
            sl::Timestamp ts = current_traj.timestamps.at(j);
            sl::ObjectData newObjectData;
            newObjectData.id = current_traj.id;
            newObjectData.tracking_state = current_traj.tracking_state;
            newObjectData.position = current_traj.positions.at(j);
            newObjectData.label = current_traj.label;
            newObjectData.sublabel = current_traj.sublabel;

            newObjectData.bounding_box_2d.clear();
			for (int p = 0; p < current_traj.bounding_boxes_2d.at(j).size(); p++) {
				newObjectData.bounding_box_2d.push_back(current_traj.bounding_boxes_2d.at(j).at(p));
			}

            newObjectData.bounding_box.clear();
            for (int k=0;k<current_traj.bounding_boxes.at(j).size();k++)
                newObjectData.bounding_box.push_back(current_traj.bounding_boxes.at(j).at(k));


            if (list_of_newobjects.find(ts.getMilliseconds())!=list_of_newobjects.end())
                list_of_newobjects[ts.getMilliseconds()].object_list.push_back(newObjectData);
            else
            {
                sl::Objects current_obj;
                current_obj.timestamp.setMilliseconds(ts.getMilliseconds());
                current_obj.is_new = true;
                current_obj.is_tracked = true;
                current_obj.object_list.push_back(newObjectData);
                list_of_newobjects[ts.getMilliseconds()] = current_obj;
            }
        }
    }

    // Ingest in Queue of objects that will be empty by the main loop
    // Since std::map is sorted by key, we are sure that timestamp are continous.
    for (auto &elem : list_of_newobjects)
       objects_tracked_queue.push_back(elem.second);
}

sl::Mat BatchSystemHandler::findClosestImageFromTS(unsigned long long timestamp) {
    sl::Mat image;
    unsigned long long ts_found = 0;
	if (imageMap_ms.find(timestamp) != imageMap_ms.end()) {
		ts_found = timestamp;
		image = imageMap_ms[timestamp];
	}
    return image;
}

sl::Mat BatchSystemHandler::findClosestDepthFromTS(unsigned long long timestamp) {
    sl::Mat depth;
    unsigned long long ts_found = 0;
    if (depthMap_ms.find(timestamp)!=depthMap_ms.end()) {
        ts_found = timestamp;
        depth = depthMap_ms[timestamp];
    }
    return depth;
}
