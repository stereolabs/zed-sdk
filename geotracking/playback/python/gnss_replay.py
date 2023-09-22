import json
import pyzed.sl as sl 
import numpy as np 


class GNSSReplay:
    def __init__(self, file_name):
        self._file_name = file_name
        self.current_gnss_idx = 0
        self.previous_ts = 0
        self.last_cam_ts = 0
        self.gnss_data = None
        self.initialize()

    def initialize(self):
        try:
            with open(self._file_name, 'r') as gnss_file_data:
                self.gnss_data = json.load(gnss_file_data)
        except FileNotFoundError:
            print(f"Unable to open {self._file_name}")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"Error while reading GNSS data: {e}")
           
    def is_microseconds(self, timestamp):
        return 1_000_000_000_000_000 <= timestamp < 10_000_000_000_000_000

    def is_nanoseconds(self, timestamp):
        return 1_000_000_000_000_000_000 <= timestamp < 10_000_000_000_000_000_000
    
    def getGNSSData(self, gnss_data,gnss_idx):
        current_gnss_data = sl.GNSSData()
        
        #if we are at the end of GNSS data, exit 
        if gnss_idx>=len(gnss_data["GNSS"]):
            print("Reached the end of the GNSS playback data.")
            return current_gnss_data
        current_gnss_data_json = gnss_data["GNSS"][gnss_idx]
        
        if (
            current_gnss_data_json["coordinates"] is None 
            or current_gnss_data_json["coordinates"]["latitude"] is None
            or current_gnss_data_json["coordinates"]["longitude"] is None
            or current_gnss_data_json["coordinates"]["altitude"] is None
            or current_gnss_data_json["ts"] is None
        ):
            print("Null GNSS playback data.")
            return current_gnss_data_json

        gnss_timestamp = current_gnss_data_json["ts"]
        ts = sl.Timestamp()
        if self.is_microseconds(gnss_timestamp):
            ts.set_microseconds(gnss_timestamp)
        elif self.is_nanoseconds(gnss_timestamp):
            ts.set_nanoseconds(gnss_timestamp)
        else:
            print("Warning: Invalid timestamp format from GNSS file")
        current_gnss_data.ts = ts
        # Fill out coordinates:
        current_gnss_data.set_coordinates(
            current_gnss_data_json["coordinates"]["latitude"],
            current_gnss_data_json["coordinates"]["longitude"],
            current_gnss_data_json["coordinates"]["altitude"],
            False
        )

        # Fill out default standard deviation:
        current_gnss_data.longitude_std = current_gnss_data_json["longitude_std"]
        current_gnss_data.latitude_std = current_gnss_data_json["latitude_std"]
        current_gnss_data.altitude_std = current_gnss_data_json["altitude_std"]

        # Fill out covariance [must not be null]
        position_covariance = [
                    current_gnss_data.longitude_std **2, 
                    0.0,
                    0.0,
                    0.0,
                    current_gnss_data.latitude_std **2, 
                    0.0,
                    0.0,
                    0.0,
                    current_gnss_data.altitude_std **2
                ]
        
        current_gnss_data.position_covariances = position_covariance

        return current_gnss_data

    def getNextGNSSValue(self, current_timestamp):
        current_gnss_data = self.getGNSSData(self.gnss_data,self.current_gnss_idx)

        if current_gnss_data is None or current_gnss_data.ts.data_ns == 0:
            return current_gnss_data

        if current_gnss_data.ts.data_ns > current_timestamp:
            current_gnss_data.ts.data_ns = 0
            return current_gnss_data

        last_data = current_gnss_data
        step = 1
        while True:
            last_data = current_gnss_data
            diff_last = current_timestamp - current_gnss_data.ts.data_ns
            current_gnss_data = self.getGNSSData(self.gnss_data,
                self.current_gnss_idx + step
            )
            
            if current_gnss_data is None or current_gnss_data.ts.data_ns == 0:
                break

            if current_gnss_data.ts.data_ns > current_timestamp:
                if (
                    current_gnss_data.ts.data_ns - current_timestamp 
                    > diff_last
                ):
                    current_gnss_data = last_data
                break
            self.current_gnss_idx += 1
        return current_gnss_data

    def grab(self, current_timestamp):
        current_data = sl.GNSSData()
        current_data.ts.data_ns = 0

        if current_timestamp > 0 and current_timestamp > self.last_cam_ts:
            current_data = self.getNextGNSSValue(current_timestamp)
        if current_data.ts.data_ns == self.previous_ts:
            current_data.ts.data_ns = 0

        self.last_cam_ts = current_timestamp

        if current_data.ts.data_ns == 0:
            return sl.FUSION_ERROR_CODE.FAILURE, None 

        self.previous_ts = current_data.ts.data_ns
        return sl.FUSION_ERROR_CODE.SUCCESS, current_data