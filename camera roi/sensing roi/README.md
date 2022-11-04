# ZED SDK - Exclusion ROI measure

## This sample shows how to apply an exclusion ROI, to avoid noise from known invalid regions

### Features

 - Camera images are displayed on an OpenCV windows
 - The ROI can be constructed from multiple rectangles, selected using OpenCV mouse callback
 - The ROI can be saved and reloaded from an image, the result is shown for the depth map image, but it's also applied for the positional tracking, object detection and all modules that use the depth map.