from data_management.io.base_file_reader import BaseReader
import datetime as dt
import numpy as np
import pandas as pd
import os
import logging


class ACERTReader(BaseReader):
    """
    This class reads
    """
    DATA_FIELDS = ["proton_density", "speed", "bx", "by", "bz", "b"]

    def __init__(self, ace_output_folder="/PAGER/WP6/data/outputs/RBM_Forecast/realtime_stream/ace_realtime_stream/"):
        """
        :param ace_output_folder: The path to the output folder of WP2 products.
        :type ace_output_folder: str
        """
        super().__init__()
        self.data_folder = ace_output_folder
        self._check_data_folder()

    def _check_data_folder(self):
        if not os.path.exists(self.data_folder):
            msg = "Data folder for WP2 ACE REAL-TIME solar wind output not found...impossible to retrieve data."
            logging.error(msg)
            raise FileNotFoundError(msg)

    @staticmethod
    def _to_date(x):
        year = int(x["year"])
        month = int(x["month"])
        day = int(x["day"])
        hour = int(str(x["time"])[0:2])
        minute = int(str(x["time"])[2:4])
        return dt.datetime(year, month, day, hour, minute)

    @staticmethod
    def _read_mag_file(filename):
        header_mag = ["year", "month", "day", "time", "Discard1", "Discard2",
                      "status_mag", "bx", "by", "bz", "b", "lat", "lon"]

        data_mag = pd.read_csv(filename, comment='#',
                               skiprows=2, delim_whitespace=True, names=header_mag, dtype={"time": str})

        data_mag["t"] = data_mag.apply(lambda x: ACERTReader._to_date(x), 1)
        data_mag.index = data_mag["t"]
        data_mag.drop(["Discard1", "Discard2", "year", "month", "day", "time", "t", "status_mag", "lat", "lon"], 1,
                      inplace=True)
        for k in ["bx", "by", "bz", "b"]:
            mask = data_mag[k] < -999.0
            data_mag.loc[mask, k] = np.nan
        return data_mag

    @staticmethod
    def _read_swepam_file(filename):
        header_sw = ["year", "month", "day", "time", "Discard1", "Discard2", "status_sw", "proton_density", "speed",
                     "temperature"]
        data_sw = pd.read_csv(filename, comment='#',
                              skiprows=2, delim_whitespace=True, names=header_sw, dtype={"time": str})
        data_sw["t"] = data_sw.apply(lambda x: ACERTReader._to_date(x), 1)
        data_sw.index = data_sw["t"]
        data_sw.drop(["Discard1", "Discard2", "year", "month", "day", "time", "t", "status_sw"], 1, inplace=True)
        for k in ["proton_density", "speed"]:
            mask = data_sw[k] < -9999.0
            data_sw.loc[mask, k] = np.nan

        mask = data_sw["temperature"] < -99999.0
        data_sw.loc[mask, "temperature"] = np.nan
        return data_sw

    def read(self, date=None, fields=None) -> pd.DataFrame:
        """
        This function reads output data from SWIFT and returns it in the form of a tuple of two pandas dataframe,
        each for each coordinate system available, GSM and HGC.

        :param date: The date in which data has been produced. It assumes that the data is produced once a day. If None
                     the data with current date is requested.
        :type date: datetime.datetime or None
        :param fields: List of fields to be extracted from the available data.
        :type fields: list or None
        :raises: FileNotFoundError: when one of the data files requested is not found.
                 KeyError: when a field requested is not among available field list
        :return: Tuple of GSM and HGC data as pandas data frames
        """
        if date is None:
            date = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        if fields is not None:
            for f in fields:
                if f not in ACERTReader.DATA_FIELDS:
                    msg = "Requested field from ACE Real time data not available..."
                    logging.error(msg)
                    raise KeyError(msg)

        file_swepam = self.data_folder + "{}{}{}_ace_swepam_1m.txt".format(str(date.year), str(date.month).zfill(2),
                                                                           str(date.day).zfill(2))
        file_mag = self.data_folder + "{}{}{}_ace_mag_1m.txt".format(str(date.year), str(date.month).zfill(2),
                                                                     str(date.day).zfill(2))
        data_swepam = ACERTReader._read_swepam_file(file_swepam)
        data_mag = ACERTReader._read_mag_file(file_mag)
        data = pd.concat([data_swepam, data_mag], 1)
        if fields is not None:
            data = data[fields]
        return data
