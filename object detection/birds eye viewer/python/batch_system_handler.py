from collections import deque
import numpy as np
import pyzed.sl as sl
import cv2

WITH_IMAGE_RETENTION = True

class BatchSystemHandler:
    '''
    The BatchSystemHandler class
    This class will transform a batch of objects (list[sl.ObjectsBatch]) from batching system into a queue/stream of sl.Objects.
    This class also handles pose of the camera in order to retrieve the pose of the camera at the object timestamp (obviously in the past).
    If WITH_IMAGE_RETENTION parameter is set to True, the images/depth/point cloud will be stored via the push() and pop() synchonized to the data.
    
    Note that image retention consumes a lot of CPU and GPU memory since we need to store them in memory so that we can output them later.
    As an example, for a latency of 2s, it consumes between 500Mb and 1Gb of memory for CPU and same for GPU. Make sure you have enought space left on memory.
    '''
    def __init__(self, data_retention_time):
        '''
        BatchSystemHandler constructor
        
        Parameters:
            data_retention_time (int): time to keep data in queue (in seconds)
        '''
        self.batch_data_retention = data_retention_time

        self.init_app_ts = sl.Timestamp()
        self.init_queue_ts = sl.Timestamp()
        self.objects_tracked_queue = deque()    # deque of sl.Objects
        self.cam_world_pose_map_ms = {}         # dict timestamp -> sl.Pose
        self.cam_local_pose_map_ms = {}         # dict timestamp -> sl.Pose
        self.image_map_ms = {}                  # dict timestamp -> image sl.Mat
        self.depth_map_ms = {}                  # dict timestamp -> depth sl.Mat

    def __del__(self):
        '''
        BatchSystemHandler destructor
        '''
        self.clear()

    def clear(self):
        '''
        clear
            Clears the remaining data in queue (free memory).
            Make sure it is called before zed is closes, otherwise you will have memory leaks
        '''
        self.objects_tracked_queue.clear()
        self.cam_world_pose_map_ms.clear()
        self.cam_local_pose_map_ms.clear()

        for key in list(self.image_map_ms.keys()):
            self.image_map_ms[key].free(sl.MEM.CPU)
            del self.image_map_ms[key]

        for key in list(self.depth_map_ms.keys()):
            self.depth_map_ms[key].free(sl.MEM.CPU)
            del self.depth_map_ms[key]

    def push(self, local_pose, world_pose, image, pc, batch):
        '''
        push
            push data in the FIFO system

        Parameters:
            local_pose (sl.Pose): current pose of the camera in camera reference frame
            world_pose (sl.Pose): current pose of the camera in world reference frame
            image (sl.Mat): image data
            pc (sl.Mat): point cloud data
            batch (list[sl.ObjectsBatch]): from ZED SDK batching system
        '''
        self.ingest_world_pose_in_map(world_pose)
        self.ingest_local_pose_in_map(local_pose)

        if WITH_IMAGE_RETENTION == True:
            self.ingest_image_in_map(world_pose.timestamp,image)
            self.ingest_depth_in_map(world_pose.timestamp,pc)
        
        self.ingest_in_objects_queue(batch)

    def pop(self, local_pose, world_pose, image, depth, objects):
        '''
        pop
            pop data from the FIFO system
        
        Parameters:
            local_pose (sl.Pose): pose of the camera in camera reference frame at objects timestamp
            world_pose (sl.Pose): pose of the camera in world reference frame at objects timestamp
            image (sl.Mat): image data at objects timestamp
            depth (sl.Mat): depth data at objects timestamp
            objects (sl.Objects): objects in the past
        '''
        objects = sl.Objects()
        local_pose = sl.Pose()
        world_pose = sl.Pose()

        if self.objects_tracked_queue:
            tracked_merged_obj = self.objects_tracked_queue[0]
            if (self.init_queue_ts.data_ns == 0):
                self.init_queue_ts = tracked_merged_obj.timestamp
        
            targetTS_ms = tracked_merged_obj.timestamp.get_milliseconds()

            local_pose = self.find_closest_local_pose_from_ts(targetTS_ms)
            world_pose = self.find_closest_world_pose_from_ts(targetTS_ms)
        
            if WITH_IMAGE_RETENTION:
                tmp_image = self.find_closest_image_from_ts(targetTS_ms)
                tmp_image.copy_to(image)
                tmp_image.free(sl.MEM.CPU)
                self.image_map_ms[targetTS_ms].free(sl.MEM.CPU)
                del self.image_map_ms[targetTS_ms]

                tmp_depth = self.find_closest_depth_from_ts(targetTS_ms)
                tmp_depth.copy_to(depth)
                tmp_depth.free(sl.MEM.CPU)
                self.depth_map_ms[targetTS_ms].free(sl.MEM.CPU)
                del self.depth_map_ms[targetTS_ms]

            objects = tracked_merged_obj
            self.objects_tracked_queue.popleft()

        return local_pose, world_pose, image, depth, objects

    def push_batch(self, batch):
        '''
        push_batch
            push data (objects data only) in the FIFO system
        
        Parameters:
            batch (list[sl.ObjectsBatch]): from ZED SDK batching system
        '''          
        self.ingest_in_objects_queue(batch)

    def pop_objects(self, objects):
        '''
        pop_objects
            pop data (objects data only) from the FIFO system
        
        Parameters:
            objects (sl.Objects): objects in the past
        '''
        if self.objects_tracked_queue:
            tracked_merged_obj = self.objects_tracked_queue[0]
            objects = tracked_merged_obj
            self.objects_tracked_queue.popleft()


    def ingest_world_pose_in_map(self, pose):
        '''
        ingest_world_pose_in_map

        Parameters:
            pose (sl.Pose): pose of the camera
        '''
        ts = pose.timestamp
        if self.init_app_ts.data_ns == 0:
            self.init_app_ts = ts

        for key in list(self.cam_world_pose_map_ms.keys()):
            if (key < (ts.get_milliseconds() - self.batch_data_retention * 1000)):
                del self.cam_world_pose_map_ms[key]
            else:
                continue
        
        self.cam_world_pose_map_ms[ts.get_milliseconds()] = pose
    
    def ingest_local_pose_in_map(self, pose):
        '''
        ingest_local_pose_in_map

        Parameters:
            pose (sl.Pose): pose of the camera
        '''
        ts = pose.timestamp
        if self.init_app_ts.data_ns == 0:
            self.init_app_ts = ts

        for key in list(self.cam_local_pose_map_ms.keys()):
            if (key < (ts.get_milliseconds() - self.batch_data_retention * 1000)):
                del self.cam_local_pose_map_ms[key]
            else:
                continue
        
        self.cam_local_pose_map_ms[ts.get_milliseconds()] = pose

    def ingest_in_objects_queue(self, batch):
        '''
        ingest_in_objects_queue: 
            Converts a list of batched objects from SDK get_objects_batch() to a sorted list of sl.Objects
            Use this function to fill a deque of sl.Objects that can be considered and used as a stream of objects with a delay
        
        Parameters:
            batch (list[sl.ObjectsBatch]): list of sl.ObjectsBatch objects obtained from calling get_objects_batch
        '''
        # If the list is empty, do nothing
        if not batch:
            return

        # Add objects in dict with timestamp as key
        list_of_new_objects = {}
        for current_traj in batch:
            # Impossible but still better to check
            if len(current_traj.timestamps) != len(current_traj.positions):
                continue
            # For each sample, construct an ObjectData and put it in the corresponding sl.Objects
            for i in range(len(current_traj.timestamps)):
                ts = current_traj.timestamps[i]
                new_object_data = sl.ObjectData()
                new_object_data.id = current_traj.id
                new_object_data.tracking_state = current_traj.tracking_state
                new_object_data.position = current_traj.positions[i]
                new_object_data.label = current_traj.label
                new_object_data.sublabel = current_traj.sublabel
                new_object_data.bounding_box_2d = current_traj.bounding_boxes_2d[i]
                new_object_data.bounding_box = current_traj.bounding_boxes[i]

                # Check if a detected object with the current timestamp already exists in the map
                if (ts.get_milliseconds() in list_of_new_objects.keys()):
                    # Append new_object_data to the object_list
                    list_of_new_objects[ts.get_milliseconds()].object_list = list_of_new_objects[ts.get_milliseconds()].object_list + [new_object_data]
                else:
                    current_obj = sl.Objects()
                    current_obj.timestamp = ts.data_ns      # sl.Objects.timestamp can be set with time in nanoseconds 
                    current_obj.is_new = True
                    current_obj.is_tracked = True 
                    # Append new_object_data to the object_list
                    current_obj.object_list = current_obj.object_list + [new_object_data]
                    list_of_new_objects[ts.get_milliseconds()] = current_obj
        
        # Ingest in Queue of objects that will be emptied by the main loop
        # Since dicts are sorted by key, we are sure that timestamps are continous.
        for value in list_of_new_objects.values():
            self.objects_tracked_queue.append(value)

    def ingest_image_in_map(self, ts, image):
        '''
        ingest_image_in_map

        Parameters:
            ts (sl.Timestamp)
            image (sl.Mat)
        '''
        self.image_map_ms[ts.get_milliseconds()] = sl.Mat()
        self.image_map_ms[ts.get_milliseconds()].clone(image)

        for key in list(self.image_map_ms.keys()):
            if key < (ts.get_milliseconds() - self.batch_data_retention * 1000 * 2):
                self.image_map_ms[key].free(sl.MEM.CPU)
                del self.image_map_ms[key]
            else:
                continue

    def ingest_depth_in_map(self, ts, depth):
        '''
        ingest_depth_in_map

        Parameters:
            ts (sl.Timestamp)
            depth (sl.Mat)
        '''
        self.depth_map_ms[ts.get_milliseconds()] = sl.Mat()
        self.depth_map_ms[ts.get_milliseconds()].clone(depth)

        for key in list(self.depth_map_ms.keys()):
            if key < (ts.get_milliseconds() - self.batch_data_retention * 1000 * 2):
                self.depth_map_ms[key].free(sl.MEM.CPU)
                del self.depth_map_ms[key]
            else:
                continue

    def find_closest_local_pose_from_ts(self, timestamp):
        '''
        find_closest_local_pose_from_ts
            Find the sl.Pose (in camera reference frame) that matches the given timestamp
        
        Parameters:
            timestamp (int): timestamp in ms (or at least in the same unit as self.cam_local_pose_map_ms)
        Return:
            The matching sl.Pose
        '''
        pose = sl.Pose()
        if timestamp in self.cam_local_pose_map_ms.keys():
            pose = self.cam_local_pose_map_ms[timestamp]
        return pose

    def find_closest_world_pose_from_ts(self, timestamp):
        '''
        find_closest_world_pose_from_ts
            Find the sl.Pose (in world reference frame) that matches the given timestamp
        
        Parameters:
            timestamp (int): timestamp in ms (or at least in the same unit as self.cam_world_pose_map_ms)
        Return:
            The matching sl.Pose
        '''
        pose = sl.Pose()
        if timestamp in self.cam_world_pose_map_ms.keys():
            pose = self.cam_world_pose_map_ms[timestamp]
        return pose

    def find_closest_image_from_ts(self, timestamp):
        '''
        find_closest_image_from_ts
            Find the sl.Mat that matches the given timestamp
        
        Parameters:
            timestamp (int): timestamp in ms (or at least in the same unit as self.image_map_ms)
        Return:
            The matching sl.Mat
        '''
        image = sl.Mat()
        if timestamp in self.image_map_ms.keys():
            image = self.image_map_ms[timestamp]
        return image

    def find_closest_depth_from_ts(self, timestamp):
        '''
        find_closest_depth_from_ts
            Find the sl.Mat that matches the given timestamp
        
        Parameters:
            timestamp (int): timestamp in ms (or at least in the same unit as self.depth_map_ms)
        Return:
            The matching sl.Mat
        '''
        depth = sl.Mat()
        if timestamp in self.depth_map_ms.keys():
            depth = self.depth_map_ms[timestamp]
        return depth


    

    