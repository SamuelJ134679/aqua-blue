from typing import IO, Union
from pathlib import Path

from dataclasses import dataclass
import numpy as np

from numpy.typing import NDArray


@dataclass
class TimeSeries:

    dependent_variable: NDArray
    times: NDArray

    def __post_init__(self):

        timesteps = np.diff(self.times)
        if not np.isclose(np.std(timesteps), 0.0):
            raise ValueError("TimeSeries.times must be uniformly spaced")
        if np.isclose(np.mean(timesteps), 0.0):
            raise ValueError("TimeSeries.times must have a timestep greater than zero")

    def save(self, file: IO, header="", delimiter=","):
        np.savetxt(
            file,
            np.vstack((self.times, self.dependent_variable.T)).T,
            delimiter=delimiter,
            header=header,
            comments=""
        )

    @property
    def num_dims(self) -> int:

        return self.dependent_variable.shape[1]

    @classmethod
    def from_csv(cls, fp: Union[IO, str, Path], time_index: int = 0):

        data = np.loadtxt(fp, delimiter=",")
        return cls(
            dependent_variable=np.delete(data, obj=time_index, axis=1),
            times=data[:, time_index]
        )

    @property
    def timestep(self) -> float:

        return self.times[1] - self.times[0]

    def __eq__(self, other) -> bool:

        return bool(np.all(self.times == other.times) and np.all(
            np.isclose(self.dependent_variable, other.dependent_variable)
        ))

    def __setitem__(self,key,value: TimeSeries):
    if not isinstance(value, TimeSeries):
        raise TypeError("Value must be a TimeSeries object")
        
    if isinstance(key, slice):
        if key.stop is not None and key.stop > len(self.dependent_variable):
            raise ValueError("Slice stop index must be within valid range")
        # Make sure the slice lengths match
        src_len = len(value.dependent_variable[key])
        dst_len = len(range(*key.indices(len(self.dependent_variable))))
        if src_len != dst_len:
            raise ValueError("Source and destination slices must have the same length")
        else:
            if key >= len(self.dependent_variable) or key >= len(self.times):
                raise ValueError("Key must be less than the length of the time series")

        self.dependent_variable[key] = value.dependent_variable[key] # numpy handles setting slices nicely so no extra explicit logic is needed.
        self.times[key] = value.times[key]
        return

    def __delitem__(self, key): # You can pass slices as key, __getslice__ and __setslice__ is dpereciated. If we wanted to handle slices
                                # numpy will have done the work for us by implimenting it for theirNDArray type. Same for __setitem__.

        if isinstance(key,slice):
            if (key.stop >= len(self.dependent_variable) or key.stop >= len(self.times)):
                raise ValueError("if key is a slice it must have a valid range")
            mask = np.ones(len(self.dependent_variable),dtype=bool)
            mask[key] = False
            self.times = self.times[mask]
            self.dependent_variable = self.dependent_variable[mask]
            return

        if key >= len(self.dependent_variable) or key >= len(self.times):
            raise ValueError("Key must be less than the length of the time series")

        self.dependent_variable = np.delete(self.dependent_variable,key)
        self.times = np.delete(self.times,key)
