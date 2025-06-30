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

    for item in items:
        ds = load_data(item, pols)
        
        if first:
            data = ds
            first = False
        
        else:
            data = xr.where(data==-9999, ds, data, keep_attrs=True)

        data = data.squeeze()
    return data

def clip_data(dataset, fillvalue=-9999, multiple_vars = False):
    if len(list(dataset.data_vars)) > 1 and not multiple_vars:
        raise Warning("All variables are clipped to the extent of first variable! Set multiple_vars to TRUE if you want to proceed.")
    mask = dataset[list(dataset.data_vars)[0]]!=fillvalue
    ymin, ymax = [np.where(mask)[0].min(), np.where(mask)[0].max()+1]
    xmin, xmax = [np.where(mask)[1].min(), np.where(mask)[1].max()+1]
    data = dataset.isel(x=slice(xmin, xmax), y=slice(ymin,ymax))
    return data

# def merge_data(items):
#     first = True
    

#     for item in items:
#         ds = bzs.load_data(item, "VH").values
        
#         if first:
#             data = ds
#             dt = np.datetime64(item.properties["datetime"], 's')
#             coords = item.properties["proj:bbox"]
#             x_coords = [coords[0], coords[2]]
#             y_coords = [coords[1], coords[3]]

#             first=False
        
#         else:
#             data = np.where(ds==-9999, data, ds)
#     return np.squeeze(data), dt, x_coords, y_coords