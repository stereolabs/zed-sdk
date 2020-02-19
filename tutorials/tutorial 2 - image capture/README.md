# Tutorial 2: Image capture

This tutorial shows how to capture left images of the ZED camera. The program will loop until we have successfully grabbed 50 images.
We assume that you have read the tutorial 1 and successfully opened your ZED.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

# Code overview
## Create a camera

As with previous tutorial, we create, configure and open the ZED. Here we show how to set a resolution and a framerate. 

## Capture data

Now that the ZED is opened, we can now capture the images coming from it.
We create a loop that capture 50 images and exit.

To capture an image and process it, you need to call Camera::grab() function. This function take runtime parameters as well, but we leave them to default in this tutorial.
Each time you want a new image, you need to call this function. if grab() returns ERROR_CODE::SUCCESS, a new image has been capture and is now available. Otherwise, you can check the status of grab() which will tell you if there is no new frame available (depending on the framerate of the camera) or if something wrong happened.

Once grab has been done, you can get all the data provided with the ZED SDK. In this tutorial, we want to retrieve the left image and its timestamp. To do so, we use the `Camera::retrieveImage()` and `Camera::getCameraTimestamp()` functions.

retrieveImage() takes a sl::Mat as parameter, as well as a VIEW mode. We first need to create the Mat before starting the loop. Note that creating a Mat does not allocate its memory, therefore the first retrieveImage() will automatically allocate its memory for us.

Since we want to stop the loop once we capture 50 images, we just increment the counter when a grab is successful.


<i>Note:</i> the image timestamp is given in nanoseconds. You can compare the timestamp between two grab() : it should be close to the framerate time, if you don't have frames dropped.