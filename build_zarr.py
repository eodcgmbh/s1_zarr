import zarr
import numpy as np
import pystac_client as pc
import stackstac
from pyproj import CRS
import dask.array as da
import time
from dask.diagnostics import ProgressBar

start_time = time.time()

print(f"Start time: {start_time}")

pc_client = pc.Client.open("https://stac.eodc.eu/api/v1")
time_range = "2024-01-01/2024-02-01"

search = pc_client.search(
    collections=["SENTINEL1_SIG0_20M"],
    datetime=time_range,
    #bbox = bbox_aut
    query={"Equi7_TileID": {"eq": "EU020M_E051N015T3"}}
)

items_eodc = search.item_collection()

item_list = list(items_eodc)
crs = CRS.from_wkt(item_list[0].properties["proj:wkt2"])
res = item_list[0].properties["gsd"]
epsg = crs.to_epsg()
bbox = item_list[0].properties["proj:bbox"]

data = stackstac.stack(items_eodc,
                        epsg=epsg,
                        assets=["VH"],
                        bounds = bbox,
                        resolution=res,
                        chunksize=[66,1,100,100],
                        fill_value=-9999,
                        rescale=False,
                        snap_bounds=False)


data = data.squeeze()
data = data.drop_vars([coord for coord in data.coords if coord not in ["x","y","time"]])
data.attrs = {}

print(f"Succesfully built data stack. Passed time: {int(time.time()-start_time)}s")

subset = data.isel(
    time=slice(0, 66),
    x=slice(0, 7500),
    y=slice(0, 7500)
)

shape = subset.shape
chunk_shape = (66,500,500)
shard_shape = (66,7500,7500)
compressor = zarr.codecs.BloscCodec()
x_shape = subset["x"].shape
y_shape = subset["y"].shape
time_shape = subset["time"].shape

store = zarr.storage.LocalStore("empty.zarr")
root = zarr.create_group(store=store, overwrite=True)
s1sig0 = root.create_group("s1sig0")

overwrite=True

vh_array = s1sig0.create_array(name="VH",
                shape=shape,
                shards=shard_shape,
                chunks=chunk_shape,
                compressors=compressor,
                dtype="int16",
                dimension_names=["time", "y", "x"],
                overwrite=overwrite)

x_array = s1sig0.create_array(name="x",
                shape=x_shape,
                dtype="float64",
                dimension_names=["x"],
                attributes={"_FillValue": "AAAAAAAA+H8="}, #fill value is NaN
                overwrite=overwrite)

y_array = s1sig0.create_array(name="y",
                shape=y_shape,
                dtype="float64",
                dimension_names=["y"],
                attributes={"_FillValue": "AAAAAAAA+H8="}, #fill value is NaN
                overwrite=overwrite)

time_array = s1sig0.create_array(name="time",
                shape=time_shape,
                dtype="int64",
                dimension_names=["time"],
                attributes={"units": "seconds since 2014-10-01 00:00:00",
                            "calendar": "proleptic_gregorian"},
                overwrite=overwrite)

print(f"Succesfully built empty zarr. Passed time: {int(time.time()-start_time)}s")

origin = np.datetime64("2014-10-01T00:00:00")
times = subset["time"].values.astype("datetime64[s]")
time_delta = (times - origin).astype("timedelta64[s]").astype("int64")

x_array[:] = subset["x"].values
y_array[:] = subset["y"].values
time_array[:] = time_delta

print(f"Succesfully wrote coordinate data. Passed time: {int(time.time()-start_time)}s")

with ProgressBar():
    da.store(subset.data, vh_array)

print(f"Succesfully wrote data. Passed time: {int(time.time()-start_time)/60}min")

zarr.consolidate_metadata(store)
zarr.consolidate_metadata(store, path="s1sig0")

print(f"Succesfully finished. Passed time: {int(time.time()-start_time)/60}min")