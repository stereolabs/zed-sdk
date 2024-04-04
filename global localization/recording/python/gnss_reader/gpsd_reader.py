import threading
import time
import pyzed.sl as sl
from gpsdclient import GPSDClient
import random
import datetime


class GPSDReader:
    def __init__(self):
        self.continue_to_grab = True
        self.new_data = False
        self.is_initialized = False
        self.current_gnss_data = None
        self.is_initialized_mtx = threading.Lock()
        self.client = None
        self.gnss_getter = None
        self.grab_gnss_data = None

    def initialize(self):
        try:
            self.client = GPSDClient(host="127.0.0.1")
        except:
            print("No GPSD running .. exit")
            return -1

        self.grab_gnss_data = threading.Thread(target=self.grabGNSSData)
        self.grab_gnss_data.start()
        print("Successfully connected to GPSD")
        print("Waiting for GNSS fix")
        received_fix = False

        self.gnss_getter = self.client.dict_stream(convert_datetime=True, filter=["TPV"])
        while not received_fix:
            gpsd_data = next(self.gnss_getter)
            if "class" in gpsd_data and gpsd_data["class"] == "TPV" and "mode" in gpsd_data and gpsd_data["mode"] >= 2:
                received_fix = True
        print("Fix found !!!")
        with self.is_initialized_mtx:
            self.is_initialized = True
        return 0

    def getNextGNSSValue(self):
        gpsd_data = None
        while gpsd_data is None:
            gpsd_data = next(self.gnss_getter)

        if "class" in gpsd_data and gpsd_data["class"] == "TPV" and "mode" in gpsd_data and gpsd_data["mode"] >= 2:
            current_gnss_data = sl.GNSSData()
            current_gnss_data.set_coordinates(gpsd_data["lat"], gpsd_data["lon"], gpsd_data["altMSL"], False)
            current_gnss_data.longitude_std = 0.001
            current_gnss_data.latitude_std = 0.001
            current_gnss_data.altitude_std = 1.0

            position_covariance = [
                gpsd_data["eph"] * gpsd_data["eph"],
                0.0,
                0.0,
                0.0,
                gpsd_data["eph"] * gpsd_data["eph"],
                0.0,
                0.0,
                0.0,
                gpsd_data["epv"] * gpsd_data["epv"]
            ]
            current_gnss_data.position_covariances = position_covariance
            timestamp_microseconds = int(gpsd_data["time"].timestamp() * 1000000)
            ts = sl.Timestamp()
            ts.set_microseconds(timestamp_microseconds)
            current_gnss_data.ts = ts
            return current_gnss_data
        else:
            print("Fix lost : GNSS reinitialization")
            self.initialize()
            return None

    def grab(self):
        if self.new_data:
            self.new_data = False
            return sl.ERROR_CODE.SUCCESS, self.current_gnss_data
        return sl.ERROR_CODE.FAILURE, None

    def grabGNSSData(self):
        while self.continue_to_grab:
            with self.is_initialized_mtx:
                if self.is_initialized:
                    break
            time.sleep(0.001)

        while self.continue_to_grab:
            self.current_gnss_data = self.getNextGNSSValue()
            if self.current_gnss_data is not None:
                self.new_data = True

    def stop_thread(self):
        self.continue_to_grab = False
