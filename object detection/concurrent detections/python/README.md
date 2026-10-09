# ZED SDK - Multi Instance Object Detection

This sample shows how to detect and track objects in space as well as human body tracking.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
 - Check the [Documentation](https://docs.stereolabs.com/docs/)

## Setting up (Optional)

For improved data retrieval and handling on GPU:
 - Install CuPy using pip
    ```sh
    pip install cupy-cuda11x  # For CUDA 11.x
    pip install cupy-cuda12x  # For CUDA 12.x
    ```
 - Install cuda bindings using pip
    ```sh
    pip install cuda-python
    ```

## Run the program
*NOTE: The ZED v1 is not compatible with this module*
- Navigate to the build directory and launch the executable
- Or open a terminal in the build directory and run the sample :

      python concurrent_object_detection_body_tracking.py

### Features
 - The camera point cloud is displayed in a 3D OpenGL view
 - 3D bounding boxes and human skeletons around detected objects are drawn
 - Objects classes and confidences can be changed

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/
