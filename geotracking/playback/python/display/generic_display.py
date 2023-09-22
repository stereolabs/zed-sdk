from display.gl_viewer import GLViewer
import pyzed.sl as sl  
import time
import json


class GenericDisplay:
    def __init__(self):
        pass

    def __del__(self):
        pass

    def init(self,camera_model):
        self.glviewer = GLViewer() 
        self.glviewer.init(camera_model)
        # Remplacez cette partie par la connexion appropriée à votre système IoT

    def updatePoseData(self, zed_rt,str_t,str_r, state):
        self.glviewer.updateData(zed_rt,str_t,str_r, state)

    def isAvailable(self):
        return self.glviewer.is_available()

    def updateGeoPoseData(self, geo_pose, current_timestamp):
        try:
            # Remplacez cette partie par l'envoi approprié des données à votre système IoT
            zedhub_message = {
                "layer_type": "geolocation",
                "label": "Fused_position",
                "position": {
                    "latitude": geo_pose.latlng_coordinates.get_latitude(False),
                    "longitude": geo_pose.latlng_coordinates.get_latitude(False),
                    "altitude": geo_pose.latlng_coordinates.get_altitude()
                },
                "epoch_timestamp": int(current_timestamp)
            }
            time.sleep(0.005)
        except ImportError:
            already_display_warning_message = False
            if not already_display_warning_message:
                already_display_warning_message = True
                print("\nZEDHub n'a pas été trouvé ... la GeoPose calculée sera sauvegardée sous forme de fichier KML.")
                print("Les résultats seront enregistrés dans le fichier \"fused_position.kml\".")
                print("Vous pouvez utiliser Google MyMaps (https://www.google.com/maps/about/mymaps/) pour le visualiser.")
            self.saveKMLData("fused_position.kml", geo_pose)

    def saveKMLData(self, filename, geo_pose):
        # Implémentez la sauvegarde de données KML appropriée ici
        pass

if __name__ == "__main__":
    generic_display = GenericDisplay()
    generic_display.init(0, [])
    
    try:
        while True:
            # Votre logique ici...
            pass
    except KeyboardInterrupt:
        pass