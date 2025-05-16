# Tutorial 9: Global Localization with the ZED

This tutorial shows how to use the ZED as a positional tracker with a GNSS equipment. The program will loop until 200 position are grabbed.
We assume that you have followed previous tutorials.

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [ZED SDK Python API](https://www.stereolabs.com/docs/app-development/python/install/)

# Code overview
## Create a camera

As in previous tutorials, we create, configure and open the ZED. 

```python
    # step 1
    # create the camera that will input the position from its odometry
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open: " + repr(status) + ". Exit program.")
        exit()
    
    # set up communication parameters and start publishing
    communication_parameters = sl.CommunicationParameters()
    communication_parameters.set_for_shared_memory()
    zed.start_publishing(communication_parameters)

    # warmup for camera 
    if zed.grab() != sl.ERROR_CODE.SUCCESS:
        print("Camera grab: " + repr(status) + ". Exit program.")
        exit()
    else:
        zed.get_position(odometry_pose, sl.REFERENCE_FRAME.WORLD)

    tracking_params = sl.PositionalTrackingParameters()
    # These parameters are mandatory to initialize the transformation between GNSS and ZED reference frames.
    tracking_params.enable_imu_fusion = True
    tracking_params.set_gravity_as_origin = True
    err = zed.enable_positional_tracking(tracking_params)
    if (err != sl.ERROR_CODE.SUCCESS):
        print("Camera positional tracking: " + repr(status) + ". Exit program.")
        exit()
    camera_info = zed.get_camera_information()

```
For the list of available parameters for tracking, check the online documentation.

## Setup fusion 
```python
    # step 2
    # init the fusion module that will input both the camera and the GPS
    fusion = sl.Fusion()
    init_fusion_parameters = sl.InitFusionParameters()
    init_fusion_parameters.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_fusion_parameters.coordinate_units = sl.UNIT.METER

    fusion.init(init_fusion_parameters)
    fusion.enable_positionnal_tracking()
    
    uuid = sl.CameraIdentifier(camera_info.serial_number)
    print("Subscribing to", uuid.serial_number, communication_parameters.comm_type) #Subscribe fusion to camera
    status = fusion.subscribe(uuid, communication_parameters, sl.Transform(0,0,0))
    if status != sl.FUSION_ERROR_CODE.SUCCESS:
        print("Failed to subscribe to", uuid.serial_number, status)
        exit(1)
```

## Capture fusion pose data

Now that the ZED is opened and fusion is setup, we create a loop to grab and retrieve the camera position.

We create here a GNSS structure containing dummy information to process fusion. 

```python
    x = 0
    
    i = 0
    while i < 200:
        # get the odometry information
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.get_position(odometry_pose, sl.REFERENCE_FRAME.WORLD)

        elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break

        # dummy GPS value
        x = x + 0.000000001
        # get the GPS information
        gnss_data = sl.GNSSData()
        gnss_data.ts = sl.get_current_timestamp()

        # put your GPS coordinates here : latitude, longitude, altitude
        gnss_data.set_coordinates(x,0,0)

        # put your covariance here if you know it, as an matrix 3x3 in a line
        # This is the default value
        covariance = [  
                        1,0.1,0.1,
                        0.1,1,0.1,
                        0.1,0.1,1
                    ]

        gnss_data.position_covariances = covariance
        fusion.ingest_gnss_data(gnss_data)

        # get the fused position
        if fusion.process() == sl.FUSION_ERROR_CODE.SUCCESS:
            fused_tracking_state = fusion.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
            if fused_tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                
                rotation = camera_pose.get_rotation_vector()
                translation = camera_pose.get_translation(py_translation)
                text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))
                pose_data = camera_pose.pose_data(sl.Transform())
                print("get position translation  = ",text_translation,", rotation_message = ",text_rotation)
        i = i + 1

    zed.close()
```

This will loop until the ZED has been tracked during 200 frames. We display the camera translation (in meters) in the console window and close the camera before exiting the application.

You can now use the ZED with a GNSS device to get precise global localization.
