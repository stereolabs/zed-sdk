#include <BatchSystemHandler.hpp>

BatchSystemHandler::BatchSystemHandler(int data_retention_time)
{
  batch_data_retention = data_retention_time;
}




BatchSystemHandler::~BatchSystemHandler()
{
    objects_tracked_queue.clear();
    camPoseMap_ms.clear();
}




///
/// \brief push: push data in the FIFO system
/// \param pose_ : current pose of the camera
/// \param batch_ : batch_ from ZED SDK batching system
///
void BatchSystemHandler::push(sl::Pose pose_,std::vector<sl::ObjectsBatch> batch_)
{
    ingestPoseInMap(pose_);
    ingestInObjectsQueue(batch_);
}


///
/// \brief pop : pop the data from the FIFO system
/// \param pose_ : pose at the sl::Objects timestamp
/// \param objects_ : sl::Objects in the past.
///
void BatchSystemHandler::pop(sl::Timestamp c_ts, sl::Pose& pose_,sl::Objects& objects_)
{
    memset(&objects_,0,sizeof(sl::Objects));
    memset(&pose_,0,sizeof(sl::Pose));

    if (objects_tracked_queue.size()>0)
    {
        sl::Objects tracked_merged_obj = objects_tracked_queue.front();
        if (init_queue_ts.data_ns==0ULL)
            init_queue_ts = tracked_merged_obj.timestamp;

        pose_ =  findClosestPoseFromTS(tracked_merged_obj.timestamp.getMilliseconds());
        objects_ = tracked_merged_obj;
        objects_tracked_queue.pop_front();
    }
}


///
/// \brief ingestPoseInMap
/// \param ts: timestamp of the pose
/// \param pose : sl::Pose of the camera
/// \param batch_duration_sc: duration in seconds in order to remove past elements.
///
void BatchSystemHandler::ingestPoseInMap(sl::Pose pose)
{
    std::map<unsigned long long,sl::Pose>::iterator it = camPoseMap_ms.begin();
    sl::Timestamp ts = pose.timestamp;
    if (init_app_ts.data_ns==0ULL)
        init_app_ts =  ts;

    for(auto it = camPoseMap_ms.begin(); it != camPoseMap_ms.end(); ) {
        if(it->first<ts.getMilliseconds() - (unsigned long long)batch_data_retention*1000)
            it = camPoseMap_ms.erase(it);
        else
            ++it;
    }

    camPoseMap_ms[ts.getMilliseconds()]=pose;
}

///
/// \brief findClosestPoseFromTS : find the sl::Pose that matched the given timestamp
/// \param timestamp in milliseconds. ( at least in the same unit than camPoseMap_ms)
/// \return sl::Pose found.
///
sl::Pose BatchSystemHandler::findClosestPoseFromTS(unsigned long long timestamp)
{
    sl::Pose pose = sl::Pose();
    unsigned long long ts_found = 0;
    if (camPoseMap_ms.find(timestamp)!=camPoseMap_ms.end()) {
        ts_found = timestamp;
        pose = camPoseMap_ms[timestamp];
    }
    return pose;
}

///
/// \brief ingestInObjectsQueue : convert a list of batched objects from SDK retreiveObjectsBatch() to a sorted list of sl::Objects
/// \n Use this function to fill a std::deque<sl::Objects> that can be considered and used as a stream of objects with a delay.
/// \param trajs from retreiveObjectsBatch()
///
void BatchSystemHandler::ingestInObjectsQueue(std::vector<sl::ObjectsBatch> trajs)
{
    // If list is empty, do nothing.
    if (trajs.empty())
        return;

    // add objects in map with timestamp as a key.
    // This ensure
    std::map<uint64_t,sl::Objects> list_of_newobjects;
    for (int i=0;i<trajs.size();i++)
    {
        sl::ObjectsBatch current_traj = trajs.at(i);

        // Impossible (!!) but still better to check...
        if (current_traj.timestamps.size()!=current_traj.positions.size())
            continue;


        //For each sample, construct a objetdata and put it in the corresponding sl::Objects
        for (int j=0;j<current_traj.timestamps.size();j++)
        {
            sl::Timestamp ts = current_traj.timestamps.at(j);
            sl::ObjectData newObjectData;
            newObjectData.id = current_traj.id;
            newObjectData.tracking_state = current_traj.tracking_state;
            newObjectData.position = current_traj.positions.at(j);
            newObjectData.label = current_traj.label;
            newObjectData.sublabel = current_traj.sublabel;


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

    return;
}
