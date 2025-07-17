import xarray as xr
import numpy as np
import zarr
from datetime import datetime, timezone
import BuildZarrStore as bzs
import pandas as pd
import os

def main(variable):
    var = variable


    folder_path = f'INCA_data/{var}'

    filepaths = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            filepaths.append(file_path)

    store = zarr.open(f'INCA.zarr/{var}', mode='r')
    arr = store[f'{var}']
    dtype=arr.dtype
    fill_value = arr.attrs.get('_FillValue', None)

    store = zarr.storage.LocalStore("INCA.zarr")
    group = zarr.group(store=store)[var]
    x_extent = group["x"][:]
    y_extent = group["y"][:]

    for i, file in enumerate(filepaths):
        data = xr.open_dataset(file, chunks={}, mask_and_scale=False)
        data = data.load()

        x_min, x_max = bzs.get_idx(x_extent, data["x"].values)
        y_min, y_max = bzs.get_idx(y_extent, data["y"].values)

        origin = np.datetime64("2011-03-15T00:00:00").astype("datetime64[h]")
        time_min, time_max = data.time.values[0].astype("datetime64[h]"), data.time.values[-1].astype("datetime64[h]")+1
        time_delta_min, time_delta_max = (time_min - origin).astype("int64"), (time_max - origin).astype("int64")

        full_range = pd.date_range(time_min, time_max, freq="1H").values.astype("datetime64[ns]")

        for value in data.time.values:
            if value in set(full_range):
                continue
            else:
                print(f"{file} Data incomplete")
                empty_array = np.full((full_range.shape[0], data["x"].values.shape[0], data["y"].values.shape[0]),
                                    fill_value=fill_value, dtype=dtype)

                template = xr.Dataset({f"{var}": (("time", "x", "y"), empty_array)},
                                    coords={
                                        "time": full_range,
                                        "x": data["x"].values,
                                        "y": data["y"].values
                                    }
                                    )

                data_filled = data.combine_first(template)
                print(f"{file} Data gaps filled with no data values")
                break


        group[var][time_delta_min:time_delta_max, y_min:y_max, x_min:x_max] = data[var].values

        print(f"{file} written to zarr store. {i}/{len(filepaths)} complete💌")

if __name__ == "__main__":
    main("TD2M")
    