def main(time_range, tileID):
    import zarr
    import numpy as np
    import pystac_client as pc
    import xarray as xr
    import rioxarray
    import pandas as pd
    from datetime import datetime, timezone
    from tqdm import tqdm
    import BuildZarrStore as bzs
    #os.environ["ZARR_V3_EXPERIMENTAL_API"] = "1"

    pc_client = pc.Client.open("https://stac.eodc.eu/api/v1")

    search = pc_client.search(
        collections=["SENTINEL1_SIG0_20M"],
        datetime=time_range,
        query={"Equi7_TileID": {"eq": tileID}}
    )

    items_eodc = search.item_collection()

    item_list = list(items_eodc)[::-1]
    grouped_items = bzs.group_dates(item_list)

    store = zarr.storage.LocalStore("s1sig0_v2.zarr")
    group = zarr.group(store=store)["AT"]
    x_extent = group["x"][:]
    y_extent = group["y"][:]
    

    for item in tqdm(item_list, leave=True):

        x_extent = group["x"]

        dataset = bzs.load_data(item, ["VH", "VV"])
        if (dataset[list(dataset.data_vars)[0]]==-9999).all().item():
            print(f"Item {item} is empty. Will be skipped!")
            continue

        dataset_clipped = bzs.clip_data(dataset, multiple_vars=True)
        aon = dataset_clipped.attrs["abs_orbit_number"]
        ron = dataset_clipped.attrs["rel_orbit_number"]
        dataset = None

        time_origin = np.datetime64("2014-10-01")
        times = dataset_clipped.time.values.astype("datetime64[D]")
        time_delta = (times - time_origin).astype("int64")

        sensing_origin = np.datetime64("2014-10-01T00:00:00")
        sensing = dataset_clipped.time.values.astype("datetime64[s]")
        sensing_delta = (sensing - sensing_origin).astype("int64")

        x_min, x_max = bzs.get_idx(x_extent, dataset_clipped["x"].values)
        y_min, y_max = bzs.get_idx(y_extent, dataset_clipped["y"].values)

        data_vh = dataset_clipped["VH"].values
        existing_data_vh = group["VH"][time_delta, y_min:y_max, x_min:x_max]
        np.copyto(existing_data_vh, data_vh, where=(existing_data_vh==-9999))
        group["VH"][time_delta, y_min:y_max, x_min:x_max] = existing_data_vh
        data_vh = None

        data_vv = dataset_clipped["VV"].values
        dataset_clipped=None
        existing_data_vv = group["VV"][time_delta, y_min:y_max, x_min:x_max]
        np.copyto(existing_data_vv, data_vv, where=(existing_data_vv==-9999))
        group["VV"][time_delta, y_min:y_max, x_min:x_max] = existing_data_vv
        data_vv = None
        existing_data_vv = None

        new_aon = existing_data_vh.astype(np.int32)
        new_aon[new_aon!=-9999] = aon
        group["absolute_orbit_number"][time_delta, y_min:y_max, x_min:x_max] = new_aon
        new_aon = None

        new_ron = existing_data_vh
        new_ron[new_ron!=-9999] = ron
        group["relative_orbit_number"][time_delta, y_min:y_max, x_min:x_max] = new_ron
        new_ron = None

        new_sensing = existing_data_vh.astype(np.int64)
        existing_data_vh = None
        new_sensing[new_sensing!=-9999] = int(sensing_delta)
        group["sensing_date"][time_delta, y_min:y_max, x_min:x_max] = new_sensing
        new_sensing = None

if __name__ == "__main__":
    main("2024-02-01/2024-03-01", "EU020M_E048N015T3")