import pystac_client as pc
from datetime import datetime
from dateutil.relativedelta import relativedelta
import csv

start = datetime(2022, 12, 30)
end = datetime(2025, 6, 8)

ranges = []

current = start
while current < end:
    next_month = current + relativedelta(days=1)
    range_str = f"{current.strftime('%Y-%m-%d')}/{next_month.strftime('%Y-%m-%d')}"
    ranges.append(range_str)
    current = next_month

pc_client = pc.Client.open("https://stac.eodc.eu/api/v1")

faulty_items = []

for range in ranges:

    search = pc_client.search(
    collections=["SENTINEL1_SIG0_20M"],
    datetime=range
    )

    items_eodc = search.item_collection_as_dict()

    results = [
        {
        "id": feature["id"],
        "parent": feature["properties"].get("parent")
        }
        for feature in items_eodc["features"]
    ]

    for entry in results:
        id_parts = entry['id'].split('_')
        id_time_str = next(p for p in id_parts if 'T' in p and len(p) == 15)
        id_time = datetime.strptime(id_time_str, "%Y%m%dT%H%M%S")

        parent_parts = entry['parent'].split('_')
        parent_time_strs = [p for p in parent_parts if len(p) == 15 and 'T' in p]
        parent_start = datetime.strptime(parent_time_strs[0], "%Y%m%dT%H%M%S")
        parent_end   = datetime.strptime(parent_time_strs[1], "%Y%m%dT%H%M%S")

        is_within = parent_start <= id_time <= parent_end

        if not is_within:
            faulty_items.append(entry["id"])

    print(f"{range} completed")

with open("faulty_items.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for item in faulty_items:
        writer.writerow([item])