import xarray as xr
import numpy as np
import zarr
import pandas as pd


import xarray as xr
import numpy as np

# Create some example data
data = np.random.rand(4, 3)

# Create an xarray DataArray
da = xr.DataArray(
    data,
    dims=["time", "space"],
    coords={
        "time": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "space": ["x", "y", "z"]
    },
    name="example_data"
)

# Print the DataArray
print(da)

# Save to NetCDF file (optional)
da.to_netcdf("example_output.nc")
print("Saved example_output.nc")


# def get_idx(array1, array2):
#     min = np.where(array1==array2[0])[0][0]
#     max = np.where(array1==array2[-1])[0][0]+1
#     return min, max

# var="TD2M"

# data = xr.open_dataset("INCA_data/INCAL_HOURLY_TD2M_201103.nc", chunks={}, mask_and_scale=False)

# store = zarr.open(f'INCA_test.zarr/TD2M', mode='r')
# arr = store[f'{var}']
# dtype=arr.dtype
# fill_value = arr.attrs.get('_FillValue', None)

# store = zarr.storage.LocalStore("INCA_test.zarr")
# group = zarr.group(store=store)[var]
# x_extent = group["x"][:]
# y_extent = group["y"][:]

# data = xr.open_dataset("/home/otto/s1_zarr/INCA_data/INCAL_HOURLY_TD2M_201103.nc", chunks={}, mask_and_scale=False)
# data = data.load()

# x_min, x_max = get_idx(x_extent, data["x"].values)
# y_min, y_max = get_idx(y_extent, data["y"].values)

# origin = np.datetime64("2011-03-15T00:00:00").astype("datetime64[h]")
# time_min, time_max = data.time.values[0].astype("datetime64[h]"), data.time.values[-1].astype("datetime64[h]")+1
# time_delta_min, time_delta_max = (time_min - origin).astype("int64"), (time_max - origin).astype("int64")

# full_range = pd.date_range(time_min, time_max, freq="1H").values.astype("datetime64[ns]")

# for value in data.time.values:
#     if value in set(full_range):
#         continue
#     else:
#         print(f"Data incomplete")
#         empty_array = np.full((full_range.shape[0], data["x"].values.shape[0], data["y"].values.shape[0]),
#                             fill_value=fill_value, dtype=dtype)

#         template = xr.Dataset({f"{var}": (("time", "x", "y"), empty_array)},
#                                 coords={
#                                 "time": full_range,
#                                 "x": data["x"].values,
#                                 "y": data["y"].values
#                                 }
#                                 )

#         data_filled = data.combine_first(template)
#         print(f"Data gaps filled with no data values")
#         break


# group[var][time_delta_min:time_delta_max, y_min:y_max, x_min:x_max] = data[var].values

# print(f" written to zarr store. complete💌")

