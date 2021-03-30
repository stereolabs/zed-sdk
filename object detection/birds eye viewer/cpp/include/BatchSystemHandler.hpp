#ifndef __BATCH_SYSTEM_HANDLER_H__
#define __BATCH_SYSTEM_HANDLER_H__

#include <iostream>
#include <deque>
#include <math.h>
#include <map>

// ZED includes
#include <sl/Camera.hpp>


///
/// \brief The BatchSystemHandler class
/// This class will transform a batch of objects ( std::vector<sl::ObjectsBatch>) from batching system into a queue/stream of sl::Objects.
/// This class also handles pose of the camera to be able to retrieve the pose of the camera at the object timestamp (obviously in the past).
///
class BatchSystemHandler {
public:

    ///
    /// \brief BatchSystemHandler
    /// \param data_retention_time : time to keep data in queue (in seconds)
    ///
    BatchSystemHandler(int data_retention_time);
    ~BatchSystemHandler();


    ///
    /// \brief push: push data in the FIFO system
    /// \param pose_ : current pose of the camera
    /// \param batch_ : batch_ from ZED SDK batching system
    ///
    void push(sl::Pose pose_,std::vector<sl::ObjectsBatch> batch_);


    ///
    /// \brief pop : pop the data from the FIFO system
    /// \param pose_ : pose at the sl::Objects timestamp
    /// \param objects_ : sl::Objects in the past.
    ///
    void pop(sl::Timestamp c_ts, sl::Pose& pose_, sl::Objects& objects_);


private:
    void ingestPoseInMap(sl::Pose pose);
    sl::Pose findClosestPoseFromTS(unsigned long long timestamp);
    void ingestInObjectsQueue(std::vector<sl::ObjectsBatch> trajs);

    int f_count  = 0;
    int batch_data_retention = 0;
    sl::Timestamp init_app_ts = 0ULL;
    sl::Timestamp init_queue_ts = 0ULL;
    std::deque<sl::Objects> objects_tracked_queue;
    std::map<unsigned long long,sl::Pose> camPoseMap_ms;


};

#endif
