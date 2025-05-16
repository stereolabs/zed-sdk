# Tutorial 2: Image capture

This tutorial shows how to capture and draw on an WPF frame support left/depth or normals images of the ZED camera. 
It also show how to interact with RuntimeParameters using XAML controls.
We assume that you have read the tutorial 1 and successfully opened your ZED.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

## Build for Windows

- Create a "build" folder in the source folder
- Open cmake-gui and select the source and build folders
- Generate the Visual Studio `Win64` solution
- Open the resulting solution and change configuration to `Release`
- Build solution

# Code overview
## Create a camera

As with previous tutorial, we create, configure and open the ZED. Here we show how to set a resolution and a framerate. 

If you want to use the SVO input, you need to specify both the Input type and the SVO file path  : 

```
 init_params.inputType = sl.INPUT_TYPE.INPUT_TYPE_SVO;
 init_params.pathSVO =  "D:/mySVOfile.svo";
```

## Capture data

Now that the ZED is opened, we can now capture the images coming from it.
We create a loop that capture images continously

To capture an image and process it, you need to call `zedCamera.grab()` function. This function take runtime parameters as well. Those runtime parameters are created as a `private sl.RuntimeParameters = new sl.RuntimeParameters();`
Then, some of the structure parameters are modified in the callback function of UI Event

```
private void OnTextureConfidenceValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
{
    int newVal = (int)e.NewValue;
    runtimeParameters.textureConfidenceThreshold = newVal;
}
```

Each time you want a new image, you need to call this function. if grab() returns ERROR_CODE.SUCCESS, a new image has been capture and is now available. Otherwise, you can check the status of grab() which will tell you if there is no new frame available (depending on the framerate of the camera) or if something wrong happened.

Once grab has been done, you can get all the data provided with the ZED SDK. In this tutorial, we want to retrieve the left image and its timestamp. To do so, we use the `zedCamera.RetrieveImage()` function.

RetrieveImage() takes a sl.ZEDMat as parameter, as well as a VIEW mode. We first need to create the Mat before starting the loop. Note that creating a Mat allocates its memory and it must be done before calling any RetrieveXXX() functions. 
The ZEDMat will not be automatically created in that function.


![Image Capture Wnd](./../../Documentation/img/image_capture.jpg)


<i>Note:</i> the image timestamp is given in nanoseconds. You can compare the timestamp between two grab() using `zedCamera.GetCameraTimestamp()` : it should be close to the framerate time, if you don't have frames dropped.