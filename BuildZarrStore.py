import zarr
import numpy as np
import pystac_client as pc
import xarray as xr
from numcodecs import Blosc
import rioxarray
import pandas as pd
from datetime import datetime, timezone
from pystac import Item

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")      

def get_idx(array1, array2):
    min = np.where(array1==array2[0])[0][0]
    max = np.where(array1==array2[-1])[0][0]+1
    return min, max

def load_data(item, pols):
    if type(pols)==str:
        data = rioxarray.open_rasterio(item.assets[pols].href).load().expand_dims(time=pd.to_datetime([item.properties["datetime"]]).tz_convert(None)).rename(pols)
    else:
        data = []
        for pol in pols:
            data.append(rioxarray.open_rasterio(item.assets[pol].href).load().expand_dims(time=pd.to_datetime([item.properties["datetime"]]).tz_convert(None)).rename(pol))
        
        data = xr.merge(data)
    return data.squeeze()

def get_datetime(item):
    return datetime.strptime(item.properties["datetime"], "%Y-%m-%dT%H:%M:%SZ")

def group_dates(item_list):
    grouped_items = [[]]
    i=0
    for item in item_list:
        
        if not grouped_items[i]:
            grouped_items[i].append(item)
        
        else: 
            if get_datetime(item) - get_datetime(grouped_items[i][-1]) <= pd.Timedelta(seconds=50):
                grouped_items[i].append(item)

            else:
                grouped_items.append([item])
                i+=1
    return grouped_items

def read_and_merge_items(items, pols):
    first = True
    if type(pols)==list:
        datasets = []
        for pol in pols:
            for item in items:
                ds = load_data(item, pol)
                
                if first:
                    data = ds
                    first = False
                
                else:
                    data = xr.where(data==-9999, ds, data, keep_attrs=True)

            if "time" in data.dims:      
                datasets.append(data)
            else:
                datasets.append(data.expand_dims(time=pd.to_datetime([item.properties["datetime"]]).tz_convert(None)))

            first=True
        data = xr.merge(datasets)

    else:
        for item in items:
            ds = load_data(item, pols)
            
            if first:
                data = ds
                first = False
            
            else:
                data = xr.where(data==-9999, ds, data, keep_attrs=True)

        data = data.to_dataset(name=pols)

    return data.squeeze()

def clip_data(dataset, fillvalue=-9999, multiple_vars = False):
    if len(list(dataset.data_vars)) > 1 and not multiple_vars:
        raise Warning("All variables are clipped to the extent of first variable! Set multiple_vars to TRUE if you want to proceed.")
    mask = dataset[list(dataset.data_vars)[0]]!=fillvalue
    ymin, ymax = [np.where(mask)[0].min(), np.where(mask)[0].max()+1]
    xmin, xmax = [np.where(mask)[1].min(), np.where(mask)[1].max()+1]
    data = dataset.isel(x=slice(xmin, xmax), y=slice(ymin,ymax))
    return data

def read_data(grouped_data):
    merged = []
    #metadata_items = ["sat:relative_orbit", "time"]
    for group in grouped_data:
        d = group[1]
        clean = []
        first_band=True
        for b in d.band.values:
            d_band = d.sel(band=b)
            first=True
            if len(d_band.time.values)==1:
                d_time = d_band.sel(time=time)
            
            else:
                for time in d_band.time.values:

                    datetime_origin = np.datetime64("2014-10-01T00:00:00")
                    datetime = d_band.sel(time=time).time.values.astype("datetime64[s]")
                    datetime_delta = (datetime - datetime_origin).astype("int64")

                    if first:
                        d_time = d_band.sel(time=time)
                        first=False

                        if first_band:
                            datetime_arr = d_time.astype("int64")
                            datetime_arr = xr.where(datetime_arr!=-9999, datetime_delta, datetime_arr, keep_attrs=True)

                    else:
                        d_time2 = d_band.sel(time=time)
                        
                        if first_band:
                            datetime_arr = xr.where(datetime_arr!=-9999, datetime_delta, datetime_arr, keep_attrs=True)

                        d_time = xr.where(d_time==-9999, d_time2, d_time, keep_attrs=True)
                        
            if "time" in d_time.coords:
                d_time = d_time.drop_vars("time")         
            
            clean.append(d_time)
            
            if first_band:
                if "time" in datetime_arr.coords:
                    datetime_arr = datetime_arr.drop_vars("time") 
                clean.append(datetime_arr.assign_coords(band="datetime"))
                first_band=False

        
        merged.append(xr.concat(clean, dim="band"))

    new_data = xr.concat(merged, dim="time_days")
    return new_data

def fill_data(new_data, time_start, time_end, freq):
    full_range = pd.date_range(time_start, time_end, freq=freq).values.astype("datetime64[s]")

    # 2. Create dummy array with desired time_days
    template = xr.DataArray(
        np.full((len(full_range),) + new_data.shape[1:], fill_value=-9999, dtype=new_data.dtype),
        dims=("time_days", "band", "y", "x"),
        coords={
            "time_days": full_range,
            "band": new_data.coords["band"],
            "y": new_data.coords["y"],
            "x": new_data.coords["x"],
        },
        attrs=new_data.attrs,
    )

    # 3. Use `combine_first` to fill missing time entries with -9999
    data_filled = new_data.combine_first(template)
    return data_filled

def make_group_INCA(group, name, shape, shards, chunks, x_shape, y_shape, time_shape, x_extent, y_extent, time_extent, scale_factor, dtype ="int16", fill_value=-999, overwrite=True):
    data_array = group.create_array(name=name,
                        shape=shape,
                        shards=shards,
                        chunks=chunks,
                        compressors = zarr.codecs.BloscCodec(),
                        dtype=dtype,
                        fill_value=fill_value,
                        dimension_names=["time", "y", "x"],
                        config={"write_empty_chunks":False},
                        attributes={"_FillValue": fill_value,
                                    "scale_factor": scale_factor},
                        overwrite=overwrite)

    x_array = group.create_array(name="x",
                    shape=x_shape,
                    chunks=x_shape,
                    dtype="float64",
                    dimension_names=["x"],
                    attributes={"_FillValue": "AAAAAAAA+H8="}, #fill value is NaN
                    overwrite=overwrite)

    y_array = group.create_array(name="y",
                    shape=y_shape,
                    chunks=y_shape,
                    dtype="float64",
                    dimension_names=["y"],
                    attributes={"_FillValue": "AAAAAAAA+H8="}, #fill value is NaN
                    overwrite=overwrite)

    time_array = group.create_array(name="time",
                    shape=time_shape,
                    chunks=time_shape,
                    dtype="int64",
                    dimension_names=["time"],
                    attributes={"units": "hours since 2011-03-15 00:00:00",
                                "calendar": "proleptic_gregorian"},
                    overwrite=overwrite)
    
    x_array[:] = x_extent
    y_array[:] = y_extent
    time_array[:] = time_extent

def make_group_S1(group, name, shape, shards, chunks, x_shape, y_shape, time_shape, x_extent, y_extent, time_extent, scale_factor=0.1, dtype ="int16", fill_value=-9999, overwrite=True):
    data_array = group.create_array(name=name,
                        shape=shape,
                        shards=shards,
                        chunks=chunks,
                        compressors = zarr.codecs.BloscCodec(),
                        dtype=dtype,
                        fill_value=fill_value,
                        dimension_names=["time", "y", "x"],
                        config={"write_empty_chunks":False},
                        attributes={"_FillValue": fill_value,
                                    "scale_factor": scale_factor},
                        overwrite=overwrite)

    x_array = group.create_array(name="x",
                    shape=x_shape,
                    chunks=x_shape,
                    dtype="float64",
                    dimension_names=["x"],
                    attributes={"_FillValue": "AAAAAAAA+H8="}, #fill value is NaN
                    overwrite=overwrite)

    y_array = group.create_array(name="y",
                    shape=y_shape,
                    chunks=y_shape,
                    dtype="float64",
                    dimension_names=["y"],
                    attributes={"_FillValue": "AAAAAAAA+H8="}, #fill value is NaN
                    overwrite=overwrite)

    time_array = group.create_array(name="time",
                    shape=time_shape,
                    dtype="int64",
                    dimension_names=["time"],
                    attributes={"units": "days since 2014-10-01",
                                "calendar": "proleptic_gregorian"},
                    overwrite=overwrite)
    
    x_array[:] = x_extent
    y_array[:] = y_extent
    time_array[:] = time_extent