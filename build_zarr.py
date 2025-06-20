import zarr
import numpy as np
import pystac_client as pc
import xarray as xr
from numcodecs import Blosc
import rioxarray
import pandas as pd
from datetime import datetime, timezone
#os.environ["ZARR_V3_EXPERIMENTAL_API"] = "1"

tile = "EU020M_E051N015T3"
time_range = "2024-01-01/2024-02-01"

def lookup(arr1, arr2):
    '''
    Get Index of values from arr2 in arr1
    '''
    lookup = {val: idx for idx, val in enumerate(arr1)}
    indices = np.array([lookup.get(val, np.nan) for val in arr2])

    return indices

def get_idx(array, value):
    return np.where(array==value)[0][0]

def load_data(item, pol):
    return rioxarray.open_rasterio(item.assets[pol].href).load().expand_dims(time=pd.to_datetime([item.properties["datetime"]]).tz_convert(None))

pc_client = pc.Client.open("https://stac.eodc.eu/api/v1")

search = pc_client.search(
    collections=["SENTINEL1_SIG0_20M"],
    datetime=time_range,
    #bbox = bbox_aut
    query={"Equi7_TileID": {"eq": "EU020M_E051N015T3"}}
)

items_eodc = search.item_collection()

item_list = list(items_eodc)[::-1]

mapping_x = np.arange(5100010, 5400000, 20)
mapping_y = np.arange(1799990, 1500000, -20)
mapping_t = np.arange(0,5000,1)

shape = (mapping_t.shape[0],mapping_x.shape[0],mapping_y.shape[0])
chunk_shape = (20,100,100)
shard_shape = (20,3000,3000)
compressors_array = zarr.codecs.BloscCodec()
x_shape = mapping_x.shape #subset["x"].shape
y_shape = mapping_y.shape #subset["y"].shape
time_shape = mapping_t.shape

overwrite=True

store = zarr.storage.LocalStore("EO_Data2.zarr")
root = zarr.create_group(store=store, overwrite=overwrite)
s1sig0 = root.create_group("s1sig0")
eu = s1sig0.create_group("EU")

vh_array = eu.create_array(name="VH",
                shape=shape,
                shards=shard_shape,
                chunks=chunk_shape,
                compressors=compressors_array,
                dtype="int16",
                fill_value=-9999,
                dimension_names=["time", "y", "x"],
                config={"write_empty_chunks":False},
                #attributes={"_FillValue": -9999},
                overwrite=overwrite)

sensing_date = eu.create_array(name="sensing_date",
                shape=shape,
                #shards=(180,15000,150000),
                chunks=shard_shape,
                dtype="int64",
                fill_value=-9999,
                dimension_names=["time", "y", "x"],
                attributes={"units": "seconds since 2014-10-01 00:00:00",
                            "calendar": "proleptic_gregorian",
                            "_FillValue": -9999},
                overwrite=overwrite)

x_array = eu.create_array(name="x",
                shape=x_shape,
                chunks=(15000,),
                dtype="float64",
                dimension_names=["x"],
                attributes={"_FillValue": "AAAAAAAA+H8="}, #fill value is NaN
                overwrite=overwrite)

y_array = eu.create_array(name="y",
                shape=y_shape,
                chunks=(15000,),
                dtype="float64",
                dimension_names=["y"],
                attributes={"_FillValue": "AAAAAAAA+H8="}, #fill value is NaN
                overwrite=overwrite)

time_array = eu.create_array(name="time",
                shape=time_shape,
                chunks=time_shape,
                dtype="int64",
                dimension_names=["time"],
                attributes={"units": "days since 2014-10-01",
                            "calendar": "proleptic_gregorian"},
                overwrite=overwrite)

x_array[:] = mapping_x
y_array[:] = mapping_y
time_array[:] = mapping_t

zarr.consolidate_metadata(store)
zarr.consolidate_metadata(store, path="s1sig0")
zarr.consolidate_metadata(store, path="s1sig0/EU")

store = zarr.storage.LocalStore("EO_Data2.zarr")
group = zarr.group(store=store)["s1sig0/EU"]

i = 1
tot_start = datetime.now(timezone.utc)
tot_reading_data = []
tot_processing_data = []
tot_data = []
tot_sensing = []

for item in item_list:

    start_time = datetime.now(timezone.utc)

    data_vh = []
    mask = []
    existing_data_vh = []
    data2append_vh = []
    existing_sensing = []
    sensing2append = []

    t_read = datetime.now(timezone.utc)
    data_vh = load_data(item, "VH").squeeze()
    tot_reading_data.append(datetime.now(timezone.utc)-t_read)
    print("Time to read: ", tot_reading_data[i-1])

    t_process = datetime.now(timezone.utc)
    mask = data_vh!=-9999
    ymin, ymax = [np.where(mask)[0].min(), np.where(mask)[0].max()+1]
    xmin, xmax = [np.where(mask)[1].min(), np.where(mask)[1].max()+1]
    data_vh = data_vh.isel(x=slice(xmin, xmax), y=slice(ymin,ymax)).expand_dims(dim={"time": [data_vh.attrs["sensing_date"]]})
    data_vh=data_vh.squeeze()
    tot_processing_data.append(datetime.now(timezone.utc)-t_process)
    print(f"Time to process {data_vh.shape}: {tot_processing_data[i-1]}")

    time_origin = np.datetime64("2014-10-01")
    times = data_vh.time.values.astype("datetime64[D]")
    time_delta = (times - time_origin).astype("int64")

    sensing_origin = np.datetime64("2014-10-01T00:00:00")
    sensing = data_vh.time.values.astype("datetime64[s]")
    sensing_delta = (sensing - sensing_origin).astype("int64")

    x_min, x_max = [get_idx(mapping_x, data_vh["x"].values[0]), get_idx(mapping_x, data_vh["x"].values[-1])+1]
    y_min, y_max = [get_idx(mapping_y, data_vh["y"].values[0]), get_idx(mapping_y, data_vh["y"].values[-1])+1]

    t_data = datetime.now(timezone.utc)
    existing_data_vh = group["VH"][time_delta, y_min:y_max, x_min:x_max]
    data2append_vh = np.where(data_vh.values==-9999, existing_data_vh, data_vh.values)
    group["VH"][time_delta, y_min:y_max, x_min:x_max] = data2append_vh
    tot_data.append(datetime.now(timezone.utc)-t_data)
    print("Time to write data: ", tot_data[i-1])

    t_sensing = datetime.now(timezone.utc)
    existing_sensing = group["sensing_date"][time_delta, y_min:y_max, x_min:x_max]
    sensing2append= np.where(data2append_vh!=-9999, int(sensing_delta), existing_sensing)
    group["sensing_date"][time_delta, y_min:y_max, x_min:x_max] = sensing2append
    tot_sensing.append(datetime.now(timezone.utc)-t_sensing)
    print("Time to write metadata: ", tot_sensing[i-1])

    print(datetime.now(timezone.utc)-start_time," / ",datetime.now(timezone.utc)-tot_start)
    print(f"{i}/{len(item_list)}\n--------------------------------")
    i+=1
