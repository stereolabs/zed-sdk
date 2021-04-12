//======= Copyright (c) Stereolabs Corporation, All rights reserved. ===============
using System;
using System.Collections.Generic;
using System.Numerics;

namespace sl
{
    ///
    /// \brief The BatchSystemHandler class
    /// This class will transform a batch of objects (List<sl.ObjectsBatch>) from batching system into a queue/stream of sl.Objects.
    /// This class also handles pose of the camera to be able to retrieve the pose of the camera at the object timestamp (obviously in the past).
    /// If WITH_IMAGE_RETENTION flag is activated, the images/depth/point cloud will be stored via the push() and pop() synchonized to the datas.
    /// \warning Note that image retention consumes a lot of CPU and GPU memory since we need to store them in memory so that we can output them later.
    /// As an example, for a latency of 2s, it consumes between 500Mb and 1Gb of memory for CPU and same for GPU. Make sure you have enought space left on memory
    class BatchSystemHandler
    {
        public static bool WITH_IMAGE_RETENTION = true;

        public BatchSystemHandler(int dataRetentionTime)
        {
            batchDataRetention = dataRetentionTime;
        }
        /// <summary>
        /// \brief clear the remaining data in queue (free memory).
        /// Make sure it's called before zed is closed, otherwise you will have memory leaks.
        /// </summary>
        public void clear()
        {
            objectsTrackedQueue.Clear();
            camWorldPoseMap_ms.Clear();
            camLocalPoseMap_ms.Clear();

            imageMap_ms.Clear();
            depthMap_ms.Clear();
        }
        /// <summary>
        /// push: push data in the FIFO system
        /// </summary>
        /// <param name="localPose"></param>
        /// <param name="worldPose"></param>
        /// <param name="image_"></param>
        /// <param name="pc_"></param>
        /// <param name="objectsBatch"></param>
        public void push(sl.Pose localPose_, sl.Pose worldPose_, sl.Mat image_, sl.Mat pc_, ref List<ObjectsBatch> batch_)
        {
            ingestWorldPoseInMap(worldPose_);
            ingestLocalPoseInMap(localPose_);
            if (WITH_IMAGE_RETENTION)
            {
                ingestImageInMap(worldPose_.timestamp, ref image_);
                ingestDepthInMap(worldPose_.timestamp, ref pc_);
            }

            ingestInObjectsQueue(ref batch_);
        }
        /// <summary>
        /// pop : pop the data from the FIFO system
        /// </summary>
        /// <param name="localPose"></param>
        /// <param name="worldPose"></param>
        /// <param name="image_"></param>
        /// <param name="depth_"></param>
        /// <param name="objects"></param>
        public void pop(ref sl.Pose localPose, ref sl.Pose worldPose, ref sl.Mat image_, ref sl.Mat depth_, ref sl.Objects objects_)
        {
            objects_ = new Objects();

            if (objectsTrackedQueue.Count > 0)
            {
                sl.Objects trackedMergedObj = objectsTrackedQueue[0];
                if (initQueueTS == 0)
                {
                    initQueueTS = trackedMergedObj.timestamp;
                }

                ulong targetTS_ms = Utils.getMilliseconds(trackedMergedObj.timestamp);

                localPose = findClosestLocalPoseFromTS(targetTS_ms);
                worldPose = findClosestWorldPoseFromTS(targetTS_ms);

                if (WITH_IMAGE_RETENTION)
                {
                    image_.SetFrom(findClosestImageFromTS(targetTS_ms));
                    imageMap_ms[targetTS_ms].Free();
                    imageMap_ms.Remove(targetTS_ms);

                    depth_.SetFrom(findClosestDepthFromTS(targetTS_ms));
                    depthMap_ms[targetTS_ms].Free();
                    depthMap_ms.Remove(targetTS_ms);
                }
                objects_ = trackedMergedObj;
                objectsTrackedQueue.RemoveAt(0);
            }
        }

        public void push(ref List<sl.ObjectsBatch> batch_)
        {
            ingestInObjectsQueue(ref batch_);
        }

        public void pop(ref sl.Objects objects_)
        {
            if (objectsTrackedQueue.Count > 0)
            {
                sl.Objects trackedMergedObj = objectsTrackedQueue[0];
                objects_ = trackedMergedObj;
                objectsTrackedQueue.RemoveAt(0);
            }
        }

        // ingest functions
        void ingestWorldPoseInMap(sl.Pose pose)
        {
            ulong ts = pose.timestamp;
            if (initAppTS == 0)
                initAppTS = ts;

            List<ulong> removals = new List<ulong>();
            foreach (ulong key in camWorldPoseMap_ms.Keys)
            {
                if (key < Utils.getMilliseconds(ts) - (ulong)batchDataRetention * 1000)
                {
                    removals.Add(key);
                }
            }

            foreach (ulong key in removals)
            {
                camWorldPoseMap_ms.Remove(key);
            }

            camWorldPoseMap_ms[Utils.getMilliseconds(ts)] = pose;
        }

        void ingestLocalPoseInMap(sl.Pose pose)
        {
            ulong ts = pose.timestamp;
            if (initAppTS == 0)
                initAppTS = ts;

            List<ulong> removals = new List<ulong>();
            foreach (ulong key in camLocalPoseMap_ms.Keys)
            {
                if (key < Utils.getMilliseconds(ts) - (ulong)batchDataRetention * 1000)
                {
                    removals.Add(key);
                }
            }
            foreach (ulong key in removals)
            {
                camLocalPoseMap_ms.Remove(key);
            }

            camLocalPoseMap_ms[Utils.getMilliseconds(ts)] = pose;
        }

        void ingestImageInMap(ulong ts, ref sl.Mat image)
        {
            sl.Mat newImage = new Mat();
            newImage.Create(image.GetResolution(), MAT_TYPE.MAT_8U_C4);
            image.CopyTo(newImage);
            imageMap_ms[Utils.getMilliseconds(ts)] = newImage;

            List<ulong> removals = new List<ulong>();
            foreach (ulong key in imageMap_ms.Keys)
            {
                if (key < Utils.getMilliseconds(ts) - (ulong)batchDataRetention * 1000 * 2)
                {
                    removals.Add(key);
                }
            }

            foreach (ulong key in removals)
            {
                imageMap_ms[key].Free();
                imageMap_ms.Remove(key);
            }
        }
        void ingestDepthInMap(ulong ts, ref sl.Mat depth)
        {
            sl.Mat newDepth = new Mat();
            newDepth.Create(depth.GetResolution(), MAT_TYPE.MAT_32F_C4);
            depth.CopyTo(newDepth);
            depthMap_ms[Utils.getMilliseconds(ts)] = newDepth;

            List<ulong> removals = new List<ulong>();
            foreach (ulong key in depthMap_ms.Keys)
            {
                if (key < Utils.getMilliseconds(ts) - (ulong)batchDataRetention * 1000 * 2)
                {
                    removals.Add(key);
                }
            }

            foreach (ulong key in removals)
            {
                depthMap_ms[key].Free();
                depthMap_ms.Remove(key);
            }
        }
        void ingestInObjectsQueue(ref List<sl.ObjectsBatch> batch_)
        {
            if (batch_.Count == 0) return;

            Dictionary<ulong, sl.Objects> listOfNewobjects = new Dictionary<ulong, Objects>();
            foreach (var it in batch_)
            {
                for (int j = 0; j < it.numData; j++)
                {
                    ulong ts = it.timestamps[j];
                    sl.ObjectData newObjectData = new ObjectData();
                    newObjectData.id = it.id;
                    newObjectData.objectTrackingState = it.trackingState;
                    newObjectData.position = it.positions[j];
                    newObjectData.label = it.label;
                    newObjectData.sublabel = it.sublabel;

                    newObjectData.boundingBox2D = new Vector2[it.boundingBoxes2D.GetLength(1)];
                    for (int p = 0; p < it.boundingBoxes2D.GetLength(1); p++)
                    {
                        newObjectData.boundingBox2D[p] = it.boundingBoxes2D[j, p];
                    }
                    newObjectData.boundingBox = new Vector3[it.boundingBoxes.GetLength(1)];
                    for (int k = 0; k < it.boundingBoxes.GetLength(1); k++)
                    {
                        newObjectData.boundingBox[k] = it.boundingBoxes[j, k];
                    }

                    if (listOfNewobjects.ContainsKey(Utils.getMilliseconds(ts)))
                    {
                        int index = listOfNewobjects[Utils.getMilliseconds(ts)].numObject;
                        sl.Objects obj = listOfNewobjects[Utils.getMilliseconds(ts)];
                        obj.objectData[index] = newObjectData;
                        obj.numObject++;

                        listOfNewobjects[Utils.getMilliseconds(ts)] = obj;
                    }
                    else
                    {
                        sl.Objects currentObj = new sl.Objects();
                        currentObj.objectData = new ObjectData[(int)Constant.MAX_OBJECTS];
                        currentObj.timestamp = ts;
                        currentObj.isNew = 1;
                        currentObj.isTracked = 1;
                        currentObj.objectData[currentObj.numObject] = newObjectData;
                        currentObj.numObject++;
                        listOfNewobjects[Utils.getMilliseconds(ts)] = currentObj;
                    }
                }
            }
            foreach (var elem in listOfNewobjects)
            {
                objectsTrackedQueue.Add(elem.Value);
            }
        }

        // Retrieve functions
        sl.Pose findClosestWorldPoseFromTS(ulong ts) // in ms
        {
            sl.Pose pose = new sl.Pose();
            ulong ts_found = 0;

            if (camWorldPoseMap_ms.ContainsKey(ts))
            {
                ts_found = ts;
                pose = camWorldPoseMap_ms[ts];
            }
            return pose;
        }
        sl.Pose findClosestLocalPoseFromTS(ulong ts) // in ms
        {
            sl.Pose pose = new sl.Pose();
            ulong ts_found = 0;

            if (camLocalPoseMap_ms.ContainsKey(ts))
            {
                ts_found = ts;
                pose = camLocalPoseMap_ms[ts];
            }
            return pose;
        }
        sl.Mat findClosestImageFromTS(ulong ts) // in ms
        {
            sl.Mat image = new sl.Mat();
            ulong ts_found = 0;
            
            if (imageMap_ms.ContainsKey(ts))
            {
                ts_found = ts;
                image = imageMap_ms[ts];
            }
            return image;
        }
        sl.Mat findClosestDepthFromTS(ulong ts) // in ms
        {
            sl.Mat depth = new sl.Mat();
            ulong ts_found = 0;

            if (depthMap_ms.ContainsKey(ts))
            {
                ts_found = ts;
                depth = depthMap_ms[ts];
            }
            return depth;
        }

        // Data
        int f_count = 0;
        int batchDataRetention = 0;
        ulong initAppTS = 0;
        ulong initQueueTS = 0;
        List<sl.Objects> objectsTrackedQueue = new List<Objects>();
        Dictionary<ulong, sl.Pose> camWorldPoseMap_ms = new Dictionary<ulong, Pose>();
        Dictionary<ulong, sl.Pose> camLocalPoseMap_ms = new Dictionary<ulong, Pose>();
        Dictionary<ulong, sl.Mat> imageMap_ms = new Dictionary<ulong, Mat>();
        Dictionary<ulong, sl.Mat> depthMap_ms = new Dictionary<ulong, Mat>();
    }
}