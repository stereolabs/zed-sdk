import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2 as cv

point = (400, 300)

def show_distance(event, x, y, args, params): # for getting mouse_position
    global point
    point = (x, y)

cv.namedWindow("Color frame")
cv.setMouseCallback("Color frame", show_distance) # taking mouse callback

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("can't find camera, please make sure the camera is connected")
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 100 
    runtime_parameters.texture_confidence_threshold = 100

    image = sl.Mat()   # intialising camera
    depth = sl.Mat()
    point_cloud = sl.Mat()


    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m

    while True:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            
            image_ocv = image.get_data() #frame for the cv2
            cv.circle(image_ocv, point, 4, (0, 0, 255))

            err, point_cloud_value = point_cloud.get_value(point[0], point[1])


            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])

            point_cloud_np = point_cloud.get_data()
            point_cloud_np.dot(tr_np)

            if not np.isnan(distance) and not np.isinf(distance):
               print("Distance to Camera at ({}, {}) (image center): {:1.3} m".format(point[0], point[1], distance), end="\r")
                
            else:
                print("Can't estimate distance at this position.")
                print("Your camera is probably too close to the scene, please move it backwards.\n")
            #sys.stdout.flush()
            cv.putText(image_ocv, "{:1.3} mtr".format((distance)), (point[0], point[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
            cv.imshow("Color frame", image_ocv)
        if cv.waitKey(10) == ord('q'):
            # Close the camera
            cv.destroyAllWindows()
            zed.close()

if __name__ == "__main__":
    main()
