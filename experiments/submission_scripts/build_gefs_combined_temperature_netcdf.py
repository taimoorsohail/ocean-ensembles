#!/usr/bin/env python3
"""Build one native-unit GEFS Tmax ensemble NetCDF for all city-days."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr
from eccodes import codes_get, codes_get_array, codes_get_double_elements, codes_grib_new_from_file, codes_release

US_LABELS = {
    "atlanta": "Atlanta", "austin": "Austin", "chicago": "Chicago", "dallas": "Dallas",
    "denver": "Denver", "houston": "Houston", "los-angeles": "Los Angeles", "miami": "Miami",
    "nyc": "NYC", "san-francisco": "San Francisco", "seattle": "Seattle",
}
US_LOCATIONS = {
    "Atlanta": (33.6407, -84.4277), "Austin": (30.1975, -97.6664),
    "Chicago": (41.9742, -87.9073), "Dallas": (32.8998, -97.0403),
    "Denver": (39.8561, -104.6737), "Houston": (29.9844, -95.3414),
    "Los Angeles": (33.9416, -118.4085), "Miami": (25.7959, -80.2870),
    "NYC": (40.6413, -73.7781), "San Francisco": (37.6213, -122.3790),
    "Seattle": (47.4502, -122.3088),
}
NONUS_TIMEZONES = {
    "Beijing": "Asia/Shanghai", "Buenos Aires": "America/Argentina/Buenos_Aires",
    "Cape Town": "Africa/Johannesburg", "Hong Kong": "Asia/Hong_Kong", "Jeddah": "Asia/Riyadh",
    "Karachi": "Asia/Karachi", "Lucknow": "Asia/Kolkata", "Mexico City": "America/Mexico_City",
    "Munich": "Europe/Berlin", "Paris": "Europe/Paris", "Sao Paulo": "America/Sao_Paulo",
    "Seoul": "Asia/Seoul", "Tel Aviv": "Asia/Jerusalem", "Tokyo": "Asia/Tokyo",
    "Toronto": "America/Toronto", "Wellington": "Pacific/Auckland",
}


def source_path(root: Path, run: pd.Timestamp, member: str, lead: int) -> Path:
    source = "gec00" if member == "c00" else f"gep{member[1:]}"
    return root / f"gefs.{run:%Y%m%d}" / f"{run:%H}" / "tmax" / (
        f"{source}.t{run:%H}z.pgrb2a.0p50.f{lead:03d}.tmax.grib2"
    )


def read_grid(path: Path, locations: np.ndarray) -> np.ndarray:
    with path.open("rb") as stream:
        handle = codes_grib_new_from_file(stream)
        if handle is None:
            raise ValueError(f"No GRIB message in {path}")
        try:
            if str(codes_get(handle, "shortName")) != "tmax" or int(codes_get(handle, "level")) != 2:
                raise ValueError(f"Unexpected field in {path}")
            lat = np.asarray(codes_get_array(handle, "latitudes"), dtype=float)
            lon = np.asarray(codes_get_array(handle, "longitudes"), dtype=float)
            indices = []
            for target_lat, target_lon in locations:
                delta = (lon - target_lon + 180.0) % 360.0 - 180.0
                distance = (lat - target_lat) ** 2 + (delta * np.cos(np.deg2rad(target_lat))) ** 2
                indices.append(int(np.argmin(distance)))
            return np.asarray(indices, dtype=np.int64)
        finally:
            codes_release(handle)


def read_values(path: Path, indices: np.ndarray) -> np.ndarray:
    with path.open("rb") as stream:
        handle = codes_grib_new_from_file(stream)
        if handle is None:
            raise ValueError(f"No GRIB message in {path}")
        try:
            values = np.asarray(codes_get_double_elements(handle, "values", indices.tolist()), dtype=np.float32)
            return values - 273.15
        finally:
            codes_release(handle)


def us_records(project: Path) -> list[dict]:
    plan = pd.read_csv(project / "us-forecast-data" / "polymarket_city_days.csv")
    records = []
    for (city_key, local_day), group in plan.groupby(["city", "local_day"], sort=True):
        city = US_LABELS[city_key]
        event = str(group.event_ticker.iloc[0])
        market_path = project / "US_polymarket_data" / "_series" / f"{city_key}-daily-weather" / "markets.parquet"
        markets = pd.read_parquet(market_path)
        bins = markets[markets.event_ticker.astype(str) == event].sort_values(["floor_strike", "cap_strike"])
        if bins.empty:
            continue
        records.append({
            "city": city, "local_day": pd.Timestamp(local_day), "snapshot": pd.Timestamp(group.snapshot_utc.iloc[0]),
            "run": pd.Timestamp(group.run_utc.iloc[0]), "release": pd.Timestamp(group.release_utc.iloc[0]),
            "first_valid": pd.Timestamp(group.valid_time_utc.min()), "last_valid": pd.Timestamp(group.valid_time_utc.max()),
            "leads": tuple(sorted(group.lead_hour.astype(int))), "location": US_LOCATIONS[city], "unit": "F",
            "lower_c": bins.floor_strike.to_numpy(float), "upper_c": bins.cap_strike.to_numpy(float),
        })
    return records


def nonus_records(project: Path) -> list[dict]:
    root = project / "us-forecast-data"
    days = pd.read_csv(root / "nonus_city_days.csv")
    market = xr.open_dataset(project / "PolyMarket_24h_PDF" / "polymarket_combined_temperature_24h_pdf.nc")
    records = []
    for row in days.itertuples(index=False):
        city_index = list(map(str, market.city.values)).index(row.city)
        day_index = list(map(str, market.day.values)).index(row.local_day)
        lower = market.bin_lower_c.isel(city=city_index, day=day_index).values.astype(float)
        upper = market.bin_upper_c.isel(city=city_index, day=day_index).values.astype(float)
        location = tuple(float(x) for x in market.location.isel(city=city_index).values)
        leads = tuple(int(x) for x in str(row.lead_hours).split(","))
        records.append({
            "city": row.city, "local_day": pd.Timestamp(row.local_day), "snapshot": pd.Timestamp(row.snapshot_utc),
            "run": pd.Timestamp(row.run_utc), "release": pd.Timestamp(row.run_utc) + pd.Timedelta(hours=4),
            "first_valid": pd.Timestamp(row.first_valid_utc), "last_valid": pd.Timestamp(row.last_valid_utc),
            "leads": leads, "location": location, "unit": "C", "lower_c": lower, "upper_c": upper,
        })
    market.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_dir / "us-forecast-data"
    records = us_records(args.project_dir) + nonus_records(args.project_dir)
    records.sort(key=lambda x: (x["city"], x["local_day"]))
    members = ["c00"] + [f"p{i:02d}" for i in range(1, 31)]
    locations = np.asarray([r["location"] for r in records], dtype=np.float32)
    member_high_c = np.full((len(records), len(members)), np.nan, dtype=np.float32)
    needed: dict[Path, list[tuple[int, int]]] = {}
    for ri, record in enumerate(records):
        for mi, member in enumerate(members):
            for lead in record["leads"]:
                needed.setdefault(source_path(root, record["run"], member, lead), []).append((ri, mi))
    sample = next(iter(needed))
    indices = read_grid(sample, locations)
    for n, (path, targets) in enumerate(needed.items(), 1):
        if not path.exists():
            continue
        values = read_values(path, indices)
        for ri, mi in targets:
            member_high_c[ri, mi] = np.nanmax([member_high_c[ri, mi], values[ri]])
        if n % 1000 == 0:
            print(f"Read {n}/{len(needed)} forecast fields", flush=True)

    bins = max(len(r["lower_c"]) for r in records)
    lower_c = np.full((len(records), bins), np.nan, dtype=np.float32)
    upper_c = np.full((len(records), bins), np.nan, dtype=np.float32)
    probability = np.full((len(records), bins), np.nan, dtype=np.float32)
    probability_normalized = np.full((len(records), bins), np.nan, dtype=np.float32)
    native = member_high_c.copy()
    unit = np.asarray([r["unit"] for r in records], dtype="U1")
    for i, record in enumerate(records):
        n = len(record["lower_c"]); lower_c[i, :n] = record["lower_c"]; upper_c[i, :n] = record["upper_c"]
        if record["unit"] == "F":
            native[i] = member_high_c[i] * 9 / 5 + 32
            lo, hi = record["lower_c"] * 9 / 5 + 32, record["upper_c"] * 9 / 5 + 32
        else:
            lo, hi = record["lower_c"], record["upper_c"]
        for b in range(n):
            probability[i, b] = np.mean((native[i] >= lo[b]) & (native[i] <= hi[b]))
        total = np.nansum(probability[i])
        if total > 0: probability_normalized[i, :n] = probability[i, :n] / total

    ds = xr.Dataset(
        data_vars={
            "forecast_high_native": (("city_day", "member"), native.astype(np.float32)),
            "forecast_high_c": (("city_day", "member"), member_high_c),
            "forecast_high_f": (("city_day", "member"), (member_high_c * 9 / 5 + 32).astype(np.float32)),
            "probability_bin": (("city_day", "bin"), probability),
            "probability_bin_normalized": (("city_day", "bin"), probability_normalized),
            "bin_lower_c": (("city_day", "bin"), lower_c), "bin_upper_c": (("city_day", "bin"), upper_c),
            "location": (("city_day", "coord"), locations),
        },
        coords={
            "city_day": np.arange(len(records)), "member": members, "bin": np.arange(bins), "coord": ["latitude", "longitude"],
            "city": ("city_day", [r["city"] for r in records]), "day": ("city_day", [r["local_day"].date() for r in records]),
            "temperature_unit": ("city_day", unit), "snapshot_utc": ("city_day", [r["snapshot"].to_datetime64() for r in records]),
            "run_utc": ("city_day", [r["run"].to_datetime64() for r in records]), "release_utc": ("city_day", [r["release"].to_datetime64() for r in records]),
            "first_valid_utc": ("city_day", [r["first_valid"].to_datetime64() for r in records]), "last_valid_utc": ("city_day", [r["last_valid"].to_datetime64() for r in records]),
        },
        attrs={"title": "Global GEFS ensemble daily maximum temperature forecasts", "source": "NOAA GEFS global 0.5-degree pgrb2a Tmax fields", "units_note": "forecast_high_native and bin bounds use temperature_unit per city-day; Celsius and Fahrenheit companion variables are also included", "probability_method": "Fraction of 31 GEFS members whose local-calendar-day maximum falls in each Polymarket bin", "city_day_count": len(records), "member_count": len(members)},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoding={v:{"zlib":True,"complevel":4} for v in ds.data_vars if ds[v].dtype.kind not in "OUS"}
    ds.to_netcdf(args.output, encoding=encoding)
    print(f"Wrote {args.output} with {len(records)} city-days")


if __name__ == "__main__": main()
