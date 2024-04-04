from display.gl_viewer import GLViewer
from exporter.KMLExporter import *
import time


class GenericDisplay:
    def __init__(self):
        pass

    def __del__(self):
        closeAllKMLFiles()

    def init(self, camera_model):
        self.glviewer = GLViewer()
        self.glviewer.init(camera_model)
        # Replace this part with the appropriate connection to your IoT system

    def updatePoseData(self, zed_rt, str_t, str_r, state):
        self.glviewer.updateData(zed_rt, str_t, str_r, state)

    def isAvailable(self):
        return self.glviewer.is_available()

    def updateRawGeoPoseData(self, geo_data):
        try:
            # Replace this part with the appropriate sending of data to your IoT system
            latitude, longitude, _ = geo_data.get_coordinates(False)
            f = open('../../map server/raw_data.txt', 'w')
            f.write("{},{},{}".format(latitude, longitude, geo_data.ts.get_milliseconds()))

        except ImportError:
            print("An exception was raised: the raw geo-pose data was not sent.")

    def updateGeoPoseData(self, geo_pose, current_timestamp):
        try:
            # Replace this part with the appropriate sending of data to your IoT system
            f = open('../../map server/data.txt', 'w')
            f.write("{},{},{}"
                    .format(geo_pose.latlng_coordinates.get_latitude(False),
                            geo_pose.latlng_coordinates.get_longitude(False),
                            current_timestamp.get_milliseconds()))

        except ImportError:
            print("An exception was raised: the geo-pose data was not sent.")


if __name__ == "__main__":
    generic_display = GenericDisplay()
    generic_display.init(0, [])

    try:
        while True:
            # Your logic here...
            pass
    except KeyboardInterrupt:
        pass
