import numpy as np 
import json  
from datetime import datetime

def get_current_datetime():
    now = datetime.now()
    return now.strftime("%d-%m-%Y_%H-%M-%S")


class GNSSSaver:
    def __init__(self):
        self.current_date = get_current_datetime()
        self.file_path = "GNSS_"+self.current_date+".json"
        self.all_gnss_data = []
        
    def addGNSSData(self,gnss_data):
        self.all_gnss_data.append(gnss_data)
        
    def saveAllData(self):
        print("Start saving GNSS data...")
        all_gnss_measurements = [] 
        for i in range(len(self.all_gnss_data)):
            latitude, longitude, altitude = self.all_gnss_data[i].get_coordinates(False) 
            gnss_measure = {}
            gnss_measure["ts"] = self.all_gnss_data[i].ts.get_nanoseconds()
            coordinates_dict = {} 
            coordinates_dict["latitude"] = latitude 
            coordinates_dict["longitude"] = longitude
            coordinates_dict["altitude"] = altitude
            gnss_measure["coordinates"] = coordinates_dict
            position_covariance = [self.all_gnss_data[i].position_covariances[j] for j in range(9)] 
            gnss_measure["position_covariance"] = position_covariance
            gnss_measure["longitude_std"] = np.sqrt(position_covariance[0 * 3 + 0])
            gnss_measure["latitude_std"] = np.sqrt(position_covariance[1 * 3 + 1])
            gnss_measure["altitude_std"] = np.sqrt(position_covariance[2 * 3 + 2])
            all_gnss_measurements.append(gnss_measure)
        final_dict = {"GNSS" : all_gnss_measurements}
        with open(self.file_path, "w") as outfile:
            # json_data refers to the above JSON
            json.dump(final_dict, outfile)
            print("All GNSS data saved")
