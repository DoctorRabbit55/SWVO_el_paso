import typing
import datetime as dt
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray
from swvo.io.RBMDataSet import (
    FolderTypeEnum,
    InstrumentLike,
    MfmLike,
    RBMDataSet,
    SatelliteLike,
    Variable,
    VariableEnum,
)
from swvo.io.RBMDataSet.utils import join_var, matlab2python


def _read_all_datasets_netcdf(file_path: str | Path) -> dict[str, Any]:
    """Reads all datasets (variables) from a NetCDF file, including those in groups.

    This function recursively traverses all groups and variables in a NetCDF-4
    file and stores their data in a dictionary. The key for each dataset is its
    full hierarchical path.

    Args:
        file_path (str | Path): The path to the NetCDF file.

    Returns:
        Dict[str, Any]: A dictionary where keys are the full variable paths
                        and values are the corresponding NumPy arrays.
    """
    datasets:dict[str,Any] = {}

    def _read_all_recursively(group:netCDF4.Group|netCDF4.Dataset, path:str=""):
        for var_name, var_obj in group.variables.items():
            full_path = f"{path}/{var_name}" if path else var_name
            datasets[full_path] = var_obj[:]

        for group_name, group_obj in group.groups.items():
            new_path = f"{path}/{group_name}" if path else group_name
            _read_all_recursively(group_obj, new_path)

    with netCDF4.Dataset(file_path, "r") as nc_file:
        _read_all_recursively(nc_file)

    return datasets

class RBMNcDataSet(RBMDataSet):
    """Class for handling RBM NetCDF data files."""

    datetime: list[dt.datetime]
    time: NDArray[np.float64]
    energy_channels: NDArray[np.float64]
    alpha_local: NDArray[np.float64]
    alpha_eq_model: NDArray[np.float64]
    alpha_eq_real: NDArray[np.float64]
    InvMu: NDArray[np.float64]
    InvMu_real: NDArray[np.float64]
    InvK: NDArray[np.float64]
    InvV: NDArray[np.float64]
    Lstar: NDArray[np.float64]
    Flux: NDArray[np.float64]
    PSD: NDArray[np.float64]
    MLT: NDArray[np.float64]
    B_SM: NDArray[np.float64]
    B_total: NDArray[np.float64]
    B_sat: NDArray[np.float64]
    xGEO: NDArray[np.float64]
    P: NDArray[np.float64]
    R0: NDArray[np.float64]
    density: NDArray[np.float64]

    def __init__(
        self,
        start_time: dt.datetime,
        end_time: dt.datetime,
        folder_path: Path,
        satellite: SatelliteLike,
        instrument: InstrumentLike,
        mfm: str,
        *,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            start_time,
            end_time,
            folder_path,
            satellite,
            instrument,
            mfm="T89",
            verbose=verbose,
        )

        self._mfm = mfm

    def _create_file_path_stem(self) -> Path:
        # implement special cases here
        # if self._satellite == SatelliteEnum.THEMIS:
        #     pass
        if self._folder_type == FolderTypeEnum.DataServer:
            return self._folder_path / self._satellite.mission / self._satellite.sat_name

        if self._folder_type == FolderTypeEnum.SingleFolder:
            return self._folder_path

        msg = "Encountered invalid FolderTypeEnum!"
        raise ValueError(msg)

    def _load_variable(self, var: Variable|VariableEnum) -> None:
        loaded_var_arrs: dict[str, NDArray[np.number]] = {}
        var_names_storred: list[str] = []

        for date in self._date_of_files:
            if self._folder_type == FolderTypeEnum.DataServer:
                start_month = date.replace(day=1)
                next_month = start_month + relativedelta(months=1, days=-1)
                date_str = start_month.strftime("%Y%m%d") + "to" + next_month.strftime("%Y%m%d")

                file_name = self._file_name_stem + date_str + "_" + self._mfm + ".nc"
            else:
                raise NotImplementedError

            datasets = _read_all_datasets_netcdf(self._file_path_stem / file_name)

            # also store python datetimes for binning
            datetimes = typing.cast(
                NDArray[np.object_],
                np.asarray([dt.datetime.fromtimestamp(t.astype(np.int64), tz=dt.timezone.utc) for t in datasets["time"]]),
            )  # type: ignore
            datasets["datetime"] = datetimes

            # limit in time
            correct_time_idx = (datetimes >= self._start_time) & (datetimes <= self._end_time)

            for key, var_arr in datasets.items():

                if ((not isinstance(var_arr, np.ndarray)) or (not np.issubdtype(var_arr.dtype, np.number))) and (
                    key != "datetime"
                ):
                    # var represents some strings or metadata objects; don't read them
                    continue
                var_arr = typing.cast("NDArray[np.number]", var_arr)

                # check if var is time dependent
                if var_arr.shape[0] == correct_time_idx.shape[0]:
                    var_arr_trimmed = var_arr[correct_time_idx.reshape(-1), ...]

                    joined_value = join_var(loaded_var_arrs[key], var_arr_trimmed) if key in loaded_var_arrs else var_arr_trimmed
                else:
                    joined_value = var_arr

                loaded_var_arrs[key] = joined_value

                if key not in var_names_storred:
                    var_names_storred.append(key)

        # not a single file was found
        if var.var_name not in var_names_storred:
            setattr(self, var.var_name, np.asarray([]))

        for var_name in var_names_storred:
            if var_name == "datetime":
                loaded_var_arrs[var_name] = list(loaded_var_arrs[var_name])  # type: ignore

            setattr(self, var_name, loaded_var_arrs[var_name])