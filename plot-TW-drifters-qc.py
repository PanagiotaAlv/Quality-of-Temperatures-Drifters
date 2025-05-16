from netCDF4 import Dataset
from datetime import datetime, timedelta
from copy import copy
import re
from glob import glob
import numpy as np
import xarray as xr
from matplotlib import pylab as plt
import matplotlib.lines as mlines
import sys
import os
import matplotlib.dates as mdates
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from netCDF4 import num2date
from pathlib import Path
import argparse

# Argument parsing 
parser = argparse.ArgumentParser(description="Generate Water Temperature plots at surface level from dataset")
parser.add_argument("-p", '--path', help='Path to the directory of interest')
parser.add_argument("-lat", "--latitude", type=float, nargs=2, required=True, help="Latitude range (min max)")
parser.add_argument("-lon", "--longitude", type=float, nargs=2, required=True, help="Longitude range (min max)")
parser.add_argument("-s", "--station_id", type=str, help="Station ID to filter (optional)")
parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder for saving plots")
args = parser.parse_args()

target_dir = args.path
output_folder = args.output_folder
lat_user = args.latitude
lon_user = args.longitude
station_id_requested = args.station_id
fnamepre = "Tw_trackmap"

# Read NetCDF files 
nc_files = sorted(glob(os.path.join(target_dir, "*.nc")))
if not nc_files:
    print("No NetCDF files found.")
    exit(1)

print(f"Reading {len(nc_files)} NetCDF files...")
ds = xr.open_mfdataset(nc_files, combine='by_coords', preprocess=lambda ds: ds.sortby('time'))

# Extract variables 
time = ds['time'].values
lat = ds['lat'].values
lon = ds['lon'].values
platform_id = ds['platform_id'].values.astype(str)
tw = ds['water_temperature'].values
cTw = ds['climatology_water_temperature'].values
quality_level = ds['quality_level'].values

# Apply geo mask 
geo_mask = (
    (lat >= lat_user[0]) & (lat <= lat_user[1]) &
    (lon >= lon_user[0]) & (lon <= lon_user[1])
)
lat, lon, time = lat[geo_mask], lon[geo_mask], time[geo_mask]
tw, cTw, platform_id = tw[geo_mask], cTw[geo_mask], platform_id[geo_mask]

# Surface level 
if tw.ndim > 1:
    tw = tw[:, 0]

# Station list 
stList = np.unique(platform_id)
if station_id_requested:
    stList = [station_id_requested]

# Loop over stations 
for station_id in stList:
    print(f"Processing station {station_id}...")
    subset = platform_id == station_id
    if not np.any(subset):
        print(f"Skipping station {station_id} (no data found).")
        continue

    valid_lon = lon[subset]
    valid_lat = lat[subset]
    valid_time = time[subset]
    valid_tw = tw[subset]
    valid_cTw = cTw[subset]

    # Mask out only NaN Tw values 
    not_nan = ~np.isnan(valid_tw)
    valid_tw = valid_tw[not_nan]
    valid_lat = valid_lat[not_nan]
    valid_lon = valid_lon[not_nan]
    valid_time = valid_time[not_nan]
    valid_cTw = valid_cTw[not_nan]

    if len(valid_tw) == 0:
        print(f"Skipping station {station_id} (no valid Tw observations).")
        continue

    has_valid_cTw = not np.all(np.isnan(valid_cTw))
    valid_diff = valid_tw - valid_cTw if has_valid_cTw else None

    if station_id_requested:
        print(f"Printing data for station {station_id}:")
        for i in range(len(valid_lat)):
            date_str = pd.to_datetime(valid_time[i]).strftime('%Y-%m-%d %H:%M')
            tw_val = f"{valid_tw[i]:.2f}"
            ctw_val = f"{valid_cTw[i]:.2f}" if not np.isnan(valid_cTw[i]) else "NaN"
            print(f"Lat: {valid_lat[i]:.2f}, Lon: {valid_lon[i]:.2f}, Date: {date_str}, Tw: {tw_val}, cTw: {ctw_val}")

    # Plotting 
    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)

    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
    ax1.set_title(f"Drifter Track - Station {station_id}", fontsize=14)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    xmin, xmax = valid_lon.min() - 3, valid_lon.max() + 3
    ymin, ymax = valid_lat.min() - 1, valid_lat.max() + 1
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.coastlines()

    # Track lines
    for i in range(len(valid_lon) - 1):
        ax1.plot(valid_lon[i:i+2], valid_lat[i:i+2],
                 color=plt.cm.coolwarm((valid_tw[i] + 5) / 10),
                 linewidth=2, alpha=0.8)

    # Scatter plot
    if has_valid_cTw:
        sc1 = ax1.scatter(valid_lon, valid_lat, c=valid_tw, cmap='coolwarm',
                          edgecolor='k', s=50, alpha=0.75)
        cbar1 = fig.colorbar(sc1, ax=ax1, orientation='vertical', pad=0.02)
        cbar1.set_label("Tw [°C]")
    else:
        sc1 = ax1.scatter(valid_lon, valid_lat, color='grey',
                          edgecolor='k', s=50, alpha=0.75)
        ax1.text(0.5, -0.15, "Climatology missing, Tw only (grey dots)",
                 ha='center', va='top', transform=ax1.transAxes,
                 fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # Stats box
    stats_text = (
        f"Lon: [{xmin:.2f}, {xmax:.2f}]   Lat: [{ymin:.2f}, {ymax:.2f}]\n"
        f"Num obs: {len(valid_tw)}"
    )
    ax1.text(0.5, -0.25, stats_text,
             ha='center', va='top', transform=ax1.transAxes,
             fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # Time series plot 
    ax2 = fig.add_subplot(122)
    ax2.set_title(f"Surface Temperature - Drifter {station_id}", fontsize=14)
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.tick_params(axis='x', rotation=45)

    if has_valid_cTw:
        sc2 = ax2.scatter(pd.to_datetime(valid_time), valid_tw, c=valid_diff,
                          cmap='coolwarm', edgecolor='k', alpha=0.75)
        ax2.set_ylim(valid_tw.min() - 1, valid_tw.max() + 1)
        cbar2 = fig.colorbar(sc2, ax=ax2)
        cbar2.set_label("Tw - cTw [°C]")
    else:
        sc2 = ax2.scatter(pd.to_datetime(valid_time), valid_tw,
                          color='grey', edgecolor='k', alpha=0.75)
        ax2.set_ylim(valid_tw.min() - 1, valid_tw.max() + 1)

    # Save figure
    output_path = os.path.join(output_folder, f"{fnamepre}_{station_id}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
