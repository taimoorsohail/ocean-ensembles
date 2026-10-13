#!/usr/bin/env python3
"""Download the supplemental global GEFS Tmax files for non-US cities."""
from __future__ import annotations
import argparse, csv, concurrent.futures
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import xarray as xr
from download_gefs_tmax import download_one

TZ = {
    "Beijing":"Asia/Shanghai", "Buenos Aires":"America/Argentina/Buenos_Aires",
    "Cape Town":"Africa/Johannesburg", "Hong Kong":"Asia/Hong_Kong", "Jeddah":"Asia/Riyadh",
    "Karachi":"Asia/Karachi", "Lucknow":"Asia/Kolkata", "Mexico City":"America/Mexico_City",
    "Munich":"Europe/Berlin", "Paris":"Europe/Paris", "Sao Paulo":"America/Sao_Paulo",
    "Seoul":"Asia/Seoul", "Tel Aviv":"Asia/Jerusalem", "Tokyo":"Asia/Tokyo",
    "Toronto":"America/Toronto", "Wellington":"Pacific/Auckland",
}
MEMBERS = ["c00"] + [f"p{i:02d}" for i in range(1, 31)]

def build_rows(project: Path, output: Path) -> list[dict]:
    existing = set()
    with (output / "gefs_tmax_plan.csv").open() as f:
        for r in csv.DictReader(f): existing.add((r["run_utc"], int(r["lead_hour"])))
    ds = xr.open_dataset(project / "PolyMarket_24h_PDF" / "polymarket_combined_temperature_24h_pdf.nc")
    rows, citydays = [], []
    for ci, city_value in enumerate(ds.city.values):
        city, tz = str(city_value), ZoneInfo(TZ[str(city_value)])
        for di, day_value in enumerate(ds.day.values):
            s = pd.to_datetime(ds.snapshot_utc_time.isel(city=ci, day=di).values.reshape(-1), utc=True, errors="coerce")
            s = s[~pd.isna(s)]
            if not len(s): continue
            snapshot = pd.Timestamp(s.view("int64").mean(), unit="ns", tz="UTC")
            day = pd.Timestamp(day_value).date()
            start = pd.Timestamp(day).tz_localize(tz).tz_convert("UTC")
            end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC")
            valid = pd.date_range(start.floor("6h") + pd.Timedelta(hours=6), end.ceil("6h"), freq="6h", tz="UTC")
            run = snapshot.floor("6h")
            if run + pd.Timedelta(hours=4) > snapshot: run -= pd.Timedelta(hours=6)
            while run >= valid[0]: run -= pd.Timedelta(hours=6)
            leads = [int((v-run) / pd.Timedelta(hours=1)) for v in valid]
            citydays.append({"city":city,"local_day":str(day),"snapshot_utc":snapshot.isoformat(),"run_utc":run.isoformat(),"first_valid_utc":valid[0].isoformat(),"last_valid_utc":valid[-1].isoformat(),"lead_hours":",".join(map(str,leads))})
            for lead in leads:
                if (run.isoformat(), lead) in existing: continue
                for member in MEMBERS:
                    source = "gec00" if member == "c00" else f"gep{member[1:]}"
                    rows.append({"run_utc":run.isoformat(),"run_date":run.strftime("%Y%m%d"),"cycle":run.strftime("%H"),"member":member,"lead_hour":lead,"source_file":f"{source}.t{run:%Hz}.pgrb2a.0p50.f{lead:03d}"})
    ds.close()
    pd.DataFrame(citydays).to_csv(output / "nonus_city_days.csv", index=False)
    with (output / "gefs_tmax_nonus_plan.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--project-dir", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--timeout", type=int, default=240); args = ap.parse_args()
    rows = build_rows(args.project_dir, args.output_dir); manifest = args.output_dir / "download_manifest.csv"; fields = list(pd.read_csv(manifest, nrows=0).columns)
    def fetch(row): return download_one((row, str(args.output_dir), args.timeout))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(fetch, rows), 1):
            with manifest.open("a", newline="") as f: csv.DictWriter(f, fieldnames=fields).writerow({k:result.get(k, "") for k in fields})
            if i % 100 == 0 or i == len(rows): print(f"completed {i}/{len(rows)} {result['status']}", flush=True)
if __name__ == "__main__": main()
