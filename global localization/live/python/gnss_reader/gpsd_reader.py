import threading
import time
import pyzed.sl as sl
from gpsdclient import GPSDClient


class GPSDReader:
    def __init__(self):
        self.continue_to_grab = True
        self.new_data = False
        self.is_initialized = False
        self.current_gnss_data = None
        self.is_initialized_mtx = threading.Lock()
        self.client = None
        self.gnss_getter = None 

    def initialize(self):
        try : 
            self.client = GPSDClient(host="127.0.0.1")
        except : 
            print("No GPSD running .. exit")
            return -1 
        self.grab_gnss_data = threading.Thread(target=self.grabGNSSData)
        self.grab_gnss_data.start()
        print("Successfully connected to GPSD")
        print("Waiting for GNSS fix")
        received_fix = False
        while not received_fix:
            self.gnss_getter = self.client.dict_stream(convert_datetime=True, filter=["TPV", "SKY"])
            gpsd_data = next(self.gnss_getter)
            if "class" in gpsd_data and gpsd_data["class"] == "TPV" and "mode" in gpsd_data and gpsd_data["mode"] >=2:
                received_fix = True
        print("Fix found !!!")
        with self.is_initialized_mtx:
            self.is_initialized = True
        return 0 

    def getNextGNSSValue(self):
        while self.continue_to_grab :
            gpsd_data = None 
            while gpsd_data is None:
                gpsd_data = next(self.gnss_getter)
            if "class" in gpsd_data and gpsd_data["class"] == "TPV" and "mode" in gpsd_data and gpsd_data["mode"] >=2:                    
                current_gnss_data = sl.GNSSData()
                current_gnss_data.set_coordinates(gpsd_data["lat"], gpsd_data["lon"], gpsd_data["altMSL"], False)
                current_gnss_data.longitude_std =  0.001
                current_gnss_data.latitude_std = 0.001
                current_gnss_data.altitude_std = 1.0

                gpsd_mode = gpsd_data["mode"]
                sl_mode = sl.GNSS_MODE.UNKNOWN

                if gpsd_mode == 0:  # MODE_NOT_SEEN
                    sl_mode = sl.GNSS_MODE.UNKNOWN
                elif gpsd_mode == 1:  # MODE_NO_FIX
                    sl_mode = sl.GNSS_MODE.NO_FIX
                elif gpsd_mode == 2:  # MODE_2D
                    sl_mode = sl.GNSS_MODE.FIX_2D
                elif gpsd_mode == 3:  # MODE_3D
                    sl_mode = sl.GNSS_MODE.FIX_3D

                sl_status = sl.GNSS_STATUS.UNKNOWN
                if 'status' in gpsd_data:
                    gpsd_status = cgpsd_data["status"]
                    if gpsd_status == 0:  # STATUS_UNK
                        sl_status = sl.GNSS_STATUS.UNKNOWN
                    elif gpsd_status == 1:  # STATUS_GPS
                        sl_status = sl.GNSS_STATUS.SINGLE
                    elif gpsd_status == 2:  # STATUS_DGPS
                        sl_status = sl.GNSS_STATUS.DGNSS
                    elif gpsd_status == 3:  # STATUS_RTK_FIX
                        sl_status = sl.GNSS_STATUS.RTK_FIX
                    elif gpsd_status == 4:  # STATUS_RTK_FLT
                        sl_status = sl.GNSS_STATUS.RTK_FLOAT
                    elif gpsd_status == 5:  # STATUS_DR
                        sl_status = sl.GNSS_STATUS.SINGLE
                    elif gpsd_status == 6:  # STATUS_GNSSDR
                        sl_status = sl.GNSS_STATUS.DGNSS
                    elif gpsd_status == 7:  # STATUS_TIME
                        sl_status = sl.GNSS_STATUS.UNKNOWN
                    elif gpsd_status == 8:  # STATUS_SIM
                        sl_status = sl.GNSS_STATUS.UNKNOWN
                    elif gpsd_status == 9:  # STATUS_PPS_FIX
                        sl_status = sl.GNSS_STATUS.SINGLE


                current_gnss_data.gnss_mode = sl_mode.value
                current_gnss_data.gnss_status = sl_status.value
                
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
            elif "class" in gpsd_data and gpsd_data["class"] == "SKY":
                nb_low_snr = 0
                if 'satellites' in gpsd_data:
                    for satellite in gpsd_data['satellites']:
                        if satellite['used'] and satellite['ss'] < 16:
                            nb_low_snr += 1
                    if nb_low_snr > 0:
                        if 'uSat' in gpsd_data and 'nSat' in gpsd_data:
                            print("[Warning] Low SNR (<16) on {} satellite(s) (using {} out of {} visible)".format(nb_low_snr, gpsd_data['uSat'], gpsd_data['nSat']))
                        else:
                            print("[Warning] Low SNR (", nb_low_snr, "< 16 )")
                    return self.getNextGNSSValue()
            else:  
                print("Fix lost: GNSS reinitialization")
                self.initialize()

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

        while self.continue_to_grab :
            self.current_gnss_data = self.getNextGNSSValue()
            self.new_data = True

    def stop_thread(self):
        self.continue_to_grab = False
        