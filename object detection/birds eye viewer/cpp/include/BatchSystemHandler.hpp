#ifndef __BATCH_SYSTEM_HANDLER_H__
#define __BATCH_SYSTEM_HANDLER_H__

#include <iostream>
#include <deque>
#include <math.h>
#include <map>

// ZED includes
#include <sl/Camera.hpp>


#define WITH_IMAGE_RETENTION true

///
/// \brief The BatchSystemHandler class
/// This class will transform a batch of objects ( std::vector<sl::ObjectsBatch>) from batching system into a queue/stream of sl::Objects.
/// This class also handles pose of the camera to be able to retrieve the pose of the camera at the object timestamp (obviously in the past).
/// If WITH_IMAGE_RETENTION flag is activated, the images/depth/point cloud will be stored via the push() and pop() synchonized to the datas.
/// \warning Note that image retention consumes a lot of CPU and GPU memory since we need to store them in memory so that we can output them later.
/// As an example, for a latency of 2s, it consumes between 500Mb and 1Gb of memory for CPU and same for GPU. Make sure you have enought space left on memory.
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
    /// \brief clear the remaining data in queue (free memory).
    /// Make sure it's called before zed is closed, otherwise you will have memory leaks.
    ///
    void clear();

    ///
    /// \brief push: push data in the FIFO system
    /// \param local_pose_ : current pose of the camera in Camera reference frame
    /// \param world_pose_ : current pose of the camera in World reference frame
    /// \param image_ : as sl::Mat (on CPU memory) to be stored
    /// \param pc_ : point cloud as sl::Mat (on GPU memory) to be stored
    /// \param batch_ : batch_ from ZED SDK batching system
    ///
    void push(sl::Pose local_pose_, sl::Pose world_pose_, sl::Mat image_, sl::Mat pc_, std::vector<sl::ObjectsBatch> &batch_);


    ///
    /// \brief pop : pop the data from the FIFO system
    /// \param local_pose_ : pose at the sl::Objects timestamp in camera reference frame
    /// \param world_pose_ : pose at the sl::Objects timestamp in world reference frame
    /// \param image_ : as sl::Mat (on CPU memory) at the sl::Objects timestamp
    /// \param depth_ : point cloud as sl::Mat (on GPU memory) at the sl::Objects timestamp
    /// \param objects_ : sl::Objects in the past.
    ///
    void pop(sl::Pose& local_pose_, sl::Pose &world_pose_, sl::Mat &image_, sl::Mat &depth_, sl::Objects& objects_);

    ///
    /// \brief push: push data in the FIFO system. Overloaded fct for objects data only
    /// \param batch_ : batch_ from ZED SDK batching system
    ///
    void push(std::vector<sl::ObjectsBatch> &batch_);

    ///
    /// \brief pop : pop the data from the FIFO system. Overloaded fct for objects data only
    /// \param objects_ : sl::Objects in the past.
    ///
    void pop(sl::Objects& objects_);

private:
    /// Ingest fcts
    void ingestWorldPoseInMap(sl::Pose pose);
    void ingestLocalPoseInMap(sl::Pose pose);
    void ingestImageInMap(sl::Timestamp ts, sl::Mat &image);
    void ingestDepthInMap(sl::Timestamp ts, sl::Mat &depth);
    void ingestInObjectsQueue(std::vector<sl::ObjectsBatch> &batch_);

    /// Retrieve fct
    sl::Pose findClosestWorldPoseFromTS(unsigned long long timestamp);
    sl::Pose findClosestLocalPoseFromTS(unsigned long long timestamp);
    sl::Mat findClosestImageFromTS(unsigned long long timestamp);
    sl::Mat findClosestDepthFromTS(unsigned long long timestamp);

    /// Data
    int f_count  = 0;
    int batch_data_retention = 0;
    sl::Timestamp init_app_ts = 0ULL;
    sl::Timestamp init_queue_ts = 0ULL;
    std::deque<sl::Objects> objects_tracked_queue;
    std::map<unsigned long long,sl::Pose> camWorldPoseMap_ms;
    std::map<unsigned long long,sl::Pose> camLocalPoseMap_ms;
    std::map<unsigned long long,sl::Mat> imageMap_ms;
    std::map<unsigned long long,sl::Mat> depthMap_ms;
};

#endif
