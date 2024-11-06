using sl;
using System;
using System.Threading;
using static Khronos.Platform;

class ClientPublisher
{
    sl.Camera zedCamera;
    Thread thread;
    bool running = false;
    int id = 0;

    /// <summary>
    /// 
    /// </summary>
    public ClientPublisher(int id_)
    {
        id = id_;
        running = false;
        zedCamera = new Camera(id_);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="inputType"></param>
    /// <returns></returns>
    public bool Open(sl.InputType inputType)
    {
        if (thread != null && thread.IsAlive)  return false;

        sl.InitParameters initParameters = new sl.InitParameters();
        initParameters.resolution = sl.RESOLUTION.AUTO;
        initParameters.cameraFPS = 30;
        initParameters.depthMode = sl.DEPTH_MODE.ULTRA;
        initParameters.coordinateUnits = sl.UNIT.METER;
        initParameters.coordinateSystem = COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP;
        initParameters.sdkVerbose = 1;
        
        switch(inputType.GetType())
        {
            case sl.INPUT_TYPE.USB:
                initParameters.inputType = sl.INPUT_TYPE.USB;
                initParameters.cameraDeviceID = id;
                break;
            case sl.INPUT_TYPE.SVO:
                initParameters.inputType = sl.INPUT_TYPE.SVO;
                initParameters.pathSVO = new string(inputType.svoInputFilename);
                break;
            case sl.INPUT_TYPE.STREAM:
                initParameters.inputType = sl.INPUT_TYPE.STREAM;
                initParameters.ipStream = new string(inputType.streamInputIp);
                initParameters.portStream = inputType.streamInputPort;
                break;
            case sl.INPUT_TYPE.GMSL:
                initParameters.inputType = sl.INPUT_TYPE.GMSL;
                break;
            default:
                Console.WriteLine("ERROR: Invalid input type");
                return false;
        }

        ERROR_CODE err = zedCamera.Open(ref initParameters);

        if (err != ERROR_CODE.SUCCESS)
        {
            Console.WriteLine("ERROR while opening the camera. Exiting...");
            Environment.Exit(-1);
        }


        if (zedCamera.CameraModel == MODEL.ZED)
        {
            Console.WriteLine(" ERROR : not compatible camera model");
            Environment.Exit(-1);
        }

        // Enable tracking (mandatory for body trackin)
        PositionalTrackingParameters positionalTrackingParameters = new PositionalTrackingParameters();
        err = zedCamera.EnablePositionalTracking(ref positionalTrackingParameters);

        if (err != ERROR_CODE.SUCCESS)
        {
            Console.WriteLine("ERROR while enable the positional tracking module. Exiting...");
            Environment.Exit(-1);
        }

        // Enable the Objects detection module
        sl.BodyTrackingParameters bodyTrackingParameters = new BodyTrackingParameters();
        bodyTrackingParameters.enableObjectTracking = false; // the body tracking will track bodies across multiple images, instead of an image-by-image basis
        bodyTrackingParameters.enableSegmentation = false;
        bodyTrackingParameters.enableBodyFitting = false; // smooth skeletons moves
        bodyTrackingParameters.detectionModel = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM;
        bodyTrackingParameters.bodyFormat = sl.BODY_FORMAT.BODY_38;
        bodyTrackingParameters.allowReducedPrecisionInference = true;
        err = zedCamera.EnableBodyTracking(ref bodyTrackingParameters);

        if (err != ERROR_CODE.SUCCESS)
        {
            Console.WriteLine("ERROR while enable the body tracking module. Exiting...");
            Environment.Exit(-1);
        }

        return true;
    }

    /// <summary>
    /// 
    /// </summary>
    public void Start()
    {
        if (zedCamera.IsOpened())
        {
            running = true;
            sl.CommunicationParameters communicationParameters = new sl.CommunicationParameters();
            communicationParameters.communicationType = sl.COMM_TYPE.INTRA_PROCESS;
            ERROR_CODE err =zedCamera.StartPublishing(ref communicationParameters);
            if (err != ERROR_CODE.SUCCESS)
            {
                Console.WriteLine("ERROR while startPublishing" + err + " . Exiting...");
                Environment.Exit(-1);
            }
            thread = new Thread(new ThreadStart(Work));
            thread.Priority = System.Threading.ThreadPriority.Highest;
            thread.Start();
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public void Stop()
    {
        running = false;
        if (thread != null && thread.IsAlive)
        {
            thread.Join();
        }
        zedCamera.Close();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="pos"></param>
    public void SetStartSVOPosition(int pos)
    {
        zedCamera.SetSVOPosition(pos);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <returns></returns>
    public bool IsRunning()
    {
        return running;
    }

    /// <summary>
    /// 
    /// </summary>
    private void Work()
    {
        sl.Bodies bodies = new sl.Bodies();
        sl.BodyTrackingRuntimeParameters bodyTrackingRuntimeParameters = new sl.BodyTrackingRuntimeParameters();
        bodyTrackingRuntimeParameters.detectionConfidenceThreshold = 40;

        sl.RuntimeParameters runtimeParameters = new sl.RuntimeParameters();
        sl.ERROR_CODE err = ERROR_CODE.FAILURE;
        while (IsRunning())
        {
            err = zedCamera.Grab(ref runtimeParameters);
            if (err == sl.ERROR_CODE.SUCCESS)
            {
                err = zedCamera.RetrieveBodies(ref bodies, ref bodyTrackingRuntimeParameters);
            }
            else
            {
                Console.WriteLine("Error while grabbing: " + err);
            }
            Thread.Sleep(2);
        }
    }
}