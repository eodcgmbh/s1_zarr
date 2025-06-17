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
    query={"Equi7_TileID": {"eq": "EU020M_E051N015T3"}}
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
    i+=1

    print(f"{i}/{len(item_list)}")

data = xr.concat(data, dim="time")
data = data.squeeze()
data = data.sortby("time")

print(f"Finished loading data {datetime.now(timezone.utc)-start_time}")

print("Intializing zarr store")

shape = (22,15000,15000)
chunk_shape = (22,100,100)
shard_shape = (22,7500,7500)
compressors_array = zarr.codecs.BloscCodec()
x_shape = (15000) #subset["x"].shape
y_shape = (15000) #subset["y"].shape
time_shape = (22)

store = zarr.storage.LocalStore("empty_2.zarr")
root = zarr.create_group(store=store, overwrite=True)
s1sig0 = root.create_group("s1sig0")

overwrite=True

vh_array = s1sig0.create_array(name="VH",
                shape=shape,
                shards=shard_shape,
                chunks=chunk_shape,
                compressors=compressors_array,
                dtype="int16",
                fill_value=-9999,
                dimension_names=["time", "x", "y"],
                #attributes={"_FillValue": -9999},
                overwrite=overwrite)

x_array = s1sig0.create_array(name="x",
                shape=x_shape,
                chunks=(15000,),
                dtype="float64",
                dimension_names=["x"],
                attributes={"_FillValue": "AAAAAAAA+H8="}, #fill value is NaN
                overwrite=overwrite)

y_array = s1sig0.create_array(name="y",
                shape=y_shape,
                chunks=(15000,),
                dtype="float64",
                dimension_names=["y"],
                attributes={"_FillValue": "AAAAAAAA+H8="}, #fill value is NaN
                overwrite=overwrite)

time_array = s1sig0.create_array(name="time",
                shape=time_shape,
                chunks=time_shape,
                dtype="int64",
                dimension_names=["time"],
                attributes={"units": "seconds since 2014-10-01 00:00:00",
                            "calendar": "proleptic_gregorian"},
                overwrite=overwrite)

origin = np.datetime64("2014-10-01T00:00:00")
times = data.time.values.astype("datetime64[s]")
time_delta = (times - origin).astype("timedelta64[s]").astype("int64")

x_array[:] = data.x.values
y_array[:] = data.y.values
time_array[:] = time_delta

print(f"finished_intializing {datetime.now(timezone.utc)-start_time}")

vh_array[:, :, :] = data.values

print(f"finished all {datetime.now(timezone.utc)-start_time}")