import zarr
import numpy as np
import pystac_client as pc
import xarray as xr
import dask.array as da
from datetime import datetime, timezone
from dask.diagnostics import ProgressBar
import rioxarray
import pandas as pd

def load_data(item, pol):
    return rioxarray.open_rasterio(item.assets[pol].href).load().expand_dims(time=pd.to_datetime([item.properties["datetime"]]).tz_convert(None))

def lookup(arr1, arr2):
    '''
    Get Index of values from arr2 in arr1
    '''
    lookup = {val: idx for idx, val in enumerate(arr1)}
    indices = np.array([lookup.get(val, np.nan) for val in arr2])

    return indices

def get_idx(array, value):
    return np.where(array==value)[0][0]

start_time = datetime.now(timezone.utc)

pc_client = pc.Client.open("https://stac.eodc.eu/api/v1")
time_range = "2024-01-01/2024-02-01"

search = pc_client.search(
    collections=["SENTINEL1_SIG0_20M"],
    datetime=time_range,
    query={"Equi7_TileID": {"eq": "EU020M_E054N015T3"}}
)

items_eodc = search.item_collection()
item_list = list(items_eodc)

data = []
i=0
data=[]
for item in item_list:
    d = load_data(item, "VH")

    if not data:
        data.append(d)

    else:
        if data[-1].time.values-d.time.values <= pd.Timedelta(seconds=50):
            d = xr.where(d.values==-9999, data[-1], d.values, keep_attrs=True)
            data[-1]=d
        else:
            data.append(d)
    i=1

print(f"{i}/{len(item_list)}")

data = xr.concat(data, dim="time")
data = data.squeeze()
data = data.sortby("time")

print(f"Finished loading data {datetime.now(timezone.utc)-start_time}")

store = zarr.storage.LocalStore("empty_2.zarr")
group = zarr.group(store=store)["s1sig0"]

origin = np.datetime64("2014-10-01T00:00:00")
times = data["time"].values.astype("datetime64[s]")
time_delta = (times - origin).astype("timedelta64[s]").astype("int64")

existing_x = group["x"][:]
existing_y = group["y"][:]
existing_times = group["time"][:]

new_x = np.setdiff1d(data.x, existing_x)
new_y = np.setdiff1d(data.y, existing_y)
new_times = np.setdiff1d(time_delta, existing_times)

new_shape_x = np.append(existing_x, new_x).shape[0]
new_shape_y = np.append(existing_y, new_y).shape[0]
new_shape_time = np.append(existing_times, new_times).shape[0]

group["x"].append(new_x)
group["y"].append(new_y)
group["time"].append(new_times)

indices_x = lookup(group["x"][:], data.x.values)
indices_y = lookup(group["y"][:], data.y.values)
indices_times = lookup(group["time"][:], time_delta)

group["VH"].resize((new_shape_time,new_shape_x,new_shape_y))

for idx, timestamp in zip(indices_times, data.time):
    group["VH"][idx, indices_x[0]:indices_x[-1], indices_y[0]:indices_y[-1]] = data.sel(time = timestamp).values