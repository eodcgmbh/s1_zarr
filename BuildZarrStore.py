import zarr
import numpy as np
import pystac_client as pc
import xarray as xr
from numcodecs import Blosc
import rioxarray
import pandas as pd
from datetime import datetime, timezone
from pystac import Item

class BuildZarrStore:

    def __init__(self, item, store_name, time_extent, x_extent, y_extent):
        self.item = item
        self.store_name = store_name
        self.time_extent = time_extent
        self.x_extent = x_extent
        self.y_extent = y_extent

    def initialize_zarr_store(self, chunk_shape, shard_shape, compressor=zarr.codecs.BloscCodec(), overwrite=True, super_group_name="s1sig0", sub_group_name="EU"):
        
        store_shape = (self.time_extent.shape[0], self.y_extent.shape[0], self.x_extent.shape[0])
        x_shape = self.x_extent.shape
        y_shape = self.y_extent.shape
        time_shape = self.time_extent.shape

        store = zarr.storage.LocalStore(self.store_name)
        root = zarr.create_group(store=store, overwrite=overwrite)
        s1sig0 = root.create_group(super_group_name)
        eu = s1sig0.create_group(sub_group_name)

        eu.create_array(name="VH",
        shape=store_shape,
        shards=shard_shape,
        chunks=chunk_shape,
        compressors=compressor,
        dtype="int16",
        fill_value=-9999,
        dimension_names=["time", "y", "x"],
        config={"write_empty_chunks":False},
        #attributes={"_FillValue": -9999},
        overwrite=overwrite)

        eu.create_array(name="sensing_date",
        shape=store_shape,
        shards=shard_shape,
        chunks=chunk_shape,
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

        x_array[:] = self.x_extent
        y_array[:] = self.y_extent
        time_array[:] = self.time_extent

    def read_item(self, asset_key):
        href = self.item.assets[asset_key].href
        timestamp = pd.to_datetime([self.item.properties["datetime"]]).tz_convert(None)

        data = (
            rioxarray
            .open_rasterio(href)
            .load()
            .expand_dims(time=timestamp)
            .squeeze()
        )
        return data

    def get_extent(self, data):
        mask = data.values != -9999

        ymin, ymax = [np.where(mask)[0].min(), np.where(mask)[0].max()+1]
        xmin, xmax = [np.where(mask)[1].min(), np.where(mask)[1].max()+1]

        return [ymin, ymax, xmin, xmax]
    
    def clip2extent(self, data):
        ymin, ymax, xmin, xmax = self.get_extent(data)

        data_clipped = data.isel(x=slice(xmin, xmax), y=slice(ymin,ymax)).expand_dims(dim={"time": [data.attrs["sensing_date"]]})
        data_clipped = data_clipped.squeeze()

        return data

    def time_encoding(self, data, origin=datetime(2014,10,1,0,0,0)):
        origin_date = np.datetime64(origin.date())
        date = data.time.values.astype("datetime64[D]")
        date_delta = (date - origin_date).astype("int64")

        origin_datetime = np.datetime64(origin)
        datetime = data.time.values.astype("datetime64[s]")
        datetime_delta = (datetime - origin_datetime).astype("int64")

        return date_delta, datetime_delta
    
    def get_idx(self, array, value):
        return np.where(array==value)[0][0]

    def write2store(self, path_to_group, asset_key, write_metadata=True):
        data = self.read_item(asset_key)
        data = self.clip2extent(data)

        store = zarr.storage.LocalStore(self.store_name)
        group = zarr.group(store=store)[path_to_group]

        x_min, x_max = [self.get_idx(self.x_extent, data["x"].values[0]), self.get_idx(self.x_extent, data["x"].values[-1])+1]
        y_min, y_max = [self.get_idx(self.y_extent, data["y"].values[0]), self.get_idx(self.y_extent, data["y"].values[-1])+1]

        date_delta, datetime_delta = self.time_encoding(data)
        
        existing_data_vh = group["VH"][date_delta, y_min:y_max, x_min:x_max]
        data2append_vh = np.where(data.values==-9999, existing_data_vh, data.values)
        group["VH"][date_delta, y_min:y_max, x_min:x_max] = data2append_vh

        if write_metadata:
            existing_sensing = group["sensing_date"][date_delta, y_min:y_max, x_min:x_max]
            sensing2append= np.where(data2append_vh!=-9999, int(datetime_delta), existing_sensing)
            group["sensing_date"][date_delta, y_min:y_max, x_min:x_max] = sensing2append
        

    

    

        
    
    
        


        