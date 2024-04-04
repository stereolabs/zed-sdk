import json
import pyzed.sl as sl

class GNSSReplay:
    def __init__(self, file_name, zed=None):
        self._file_name = file_name
        self._zed = zed
        self.current_gnss_idx = 0
        self.previous_ts = 0
        self.last_cam_ts = 0
        self.gnss_data = None
        self.initialize()

    def initialize(self):
        if self._file_name is not None:
            try:
                with open(self._file_name, 'r') as gnss_file_data:
                    self.gnss_data = json.load(gnss_file_data)
            except FileNotFoundError:
                print(f"Unable to open {self._file_name}")
                exit(1)
            except json.JSONDecodeError as e:
                print(f"Error while reading GNSS data: {e}")
        elif self._zed is not None:
            keys = self._zed.get_svo_data_keys()
            gnss_key = "GNSS_json"
            if gnss_key not in keys:
                print("SVO doesn't contain GNSS data")
                exit(1)
            else:
                ts_begin = sl.Timestamp()
                data = {}
                # self.gnss_data["GNSS"] = []
                self.gnss_data = {"GNSS": []}
                err = self._zed.retrieve_svo_data(gnss_key, data, ts_begin, ts_begin)
                for k, d in data.items():
                    gnss_data_point_json = json.loads(d.get_content_as_string())
                    gnss_data_point_formatted = {}

                    latitude = gnss_data_point_json["Geopoint"]["Latitude"]
                    longitude = gnss_data_point_json["Geopoint"]["Longitude"]
                    altitude = gnss_data_point_json["Geopoint"]["Altitude"]

                    gnss_data_point_formatted["coordinates"] = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "altitude": altitude
                    }

                    gnss_data_point_formatted["ts"] = gnss_data_point_json["EpochTimeStamp"]

                    latitude_std = gnss_data_point_json["Eph"]
                    longitude_std = gnss_data_point_json["Eph"]
                    altitude_std = gnss_data_point_json["Epv"]

                    gnss_data_point_formatted["latitude_std"] = latitude_std
                    gnss_data_point_formatted["longitude_std"] = longitude_std
                    gnss_data_point_formatted["altitude_std"] = altitude_std

                    gnss_data_point_formatted["position_covariance"] = {
                        longitude_std + longitude_std, 0, 0, 0, latitude_std + latitude_std, 0, 0, 0,
                        altitude_std + altitude_std
                    }

                    if "mode" in gnss_data_point_json:
                        gnss_data_point_formatted["mode"] = gnss_data_point_json["mode"]
                    if "status" in gnss_data_point_json:
                        gnss_data_point_formatted["status"] = gnss_data_point_json["status"]

                    if "fix" in gnss_data_point_json:
                        gnss_data_point_formatted["fix"] = gnss_data_point_json["fix"]

                    gnss_data_point_formatted["original_gnss_data"] = gnss_data_point_json

                    self.gnss_data["GNSS"].append(gnss_data_point_formatted)
                    # print(json.loads(d.get_content_as_string()))

    def is_microseconds(self, timestamp):
        return 1_000_000_000_000_000 <= timestamp < 10_000_000_000_000_000

    def is_nanoseconds(self, timestamp):
        return 1_000_000_000_000_000_000 <= timestamp < 10_000_000_000_000_000_000

    def getGNSSData(self, gnss_data, gnss_idx):
        current_gnss_data = sl.GNSSData()

        # if we are at the end of GNSS data, exit
        if gnss_idx >= len(gnss_data["GNSS"]):
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
            current_gnss_data.longitude_std ** 2,
            0.0,
            0.0,
            0.0,
            current_gnss_data.latitude_std ** 2,
            0.0,
            0.0,
            0.0,
            current_gnss_data.altitude_std ** 2
        ]

        current_gnss_data.position_covariances = position_covariance

        if "mode" in current_gnss_data_json:
            current_gnss_data.gnss_mode = current_gnss_data_json["mode"]
        if "status" in current_gnss_data_json:
            current_gnss_data.gnss_status = current_gnss_data_json["status"]

        if "fix" in current_gnss_data_json:
            # Acquisition comes from GPSD https:#gitlab.com/gpsd/gpsd/-/blob/master/include/gps.h#L183-211
            gpsd_mode = current_gnss_data_json["fix"]["mode"]
            sl_mode = sl.GNSS_MODE.UNKNOWN

            if gpsd_mode == 0:  # MODE_NOT_SEEN
                sl_mode = sl.GNSS_MODE.UNKNOWN
            elif gpsd_mode == 1:  # MODE_NO_FIX
                sl_mode = sl.GNSS_MODE.NO_FIX
            elif gpsd_mode == 2:  # MODE_2D
                sl_mode = sl.GNSS_MODE.FIX_2D
            elif gpsd_mode == 3:  # MODE_3D
                sl_mode = sl.GNSS_MODE.FIX_3D

            gpsd_status = current_gnss_data_json["fix"]["status"]
            sl_status = sl.GNSS_STATUS.UNKNOWN

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

        return current_gnss_data

    def getNextGNSSValue(self, current_timestamp):
        current_gnss_data = self.getGNSSData(self.gnss_data, self.current_gnss_idx)

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
