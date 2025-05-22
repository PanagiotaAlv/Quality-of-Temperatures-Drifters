# Import required libraries
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
import re
from glob import glob
import numpy as np
import xarray as xr
from matplotlib import pylab as plt
import os
import matplotlib.dates as mdates
import pandas as pd
import cartopy.crs as ccrs
from matplotlib.colors import Normalize
from pathlib import Path
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate Water Temperature plots at surface level from dataset")
parser.add_argument("-p", '--path', help='Path to the directory of interest')
parser.add_argument("-lat", "--latitude", type=float, nargs=2, required=True, help="Latitude range (min max)")
parser.add_argument("-lon", "--longitude", type=float, nargs=2, required=True, help="Longitude range (min max)")
parser.add_argument("-s", "--station_id", type=str, help="Station ID to filter (optional)")
parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder for saving plots")
args = parser.parse_args()

# Assign argument values to variables
target_dir = args.path
output_folder = args.output_folder
lat_user = args.latitude
lon_user = args.longitude
station_id_requested = args.station_id
fnamepre = "Tw_trackmap"  # Prefix for saved plot files

# Find all NetCDF files in the given directory
nc_files = sorted(glob(os.path.join(target_dir, "*.nc")))
if not nc_files:
    print("No NetCDF files found.")
    exit(1)

print(f"Reading {len(nc_files)} NetCDF files...")

# Load and combine NetCDF files, sorted by time
ds = xr.open_mfdataset(nc_files, combine='by_coords', preprocess=lambda ds: ds.sortby('time'))

# Extract relevant data variables
time = ds['time'].values
lat = ds['lat'].values
lon = ds['lon'].values
platform_id = ds['platform_id'].values.astype(str)
tw_all = ds['water_temperature'].values
cTw = ds['climatology_water_temperature'].values
quality_level = ds['quality_level'].values

# Create a geographical mask to include only data within user-specified bounds
geo_mask = (
    (lat >= lat_user[0]) & (lat <= lat_user[1]) &
    (lon >= lon_user[0]) & (lon <= lon_user[1])
)

# Apply the mask to all variables
lat, lon, time = lat[geo_mask], lon[geo_mask], time[geo_mask]
tw_all, cTw, platform_id = tw_all[geo_mask], cTw[geo_mask], platform_id[geo_mask]

# Use temperature data from variable water_temperature that has 5 columns for 5 different depths.
# Only use data from the two first columns and combine them. Give priority to 1st column and then
# add data from 2nd column where 1st column is empty.
tw0 = tw_all[:, 0]
tw1 = tw_all[:, 1]
#sub0 = (tw0 > -40) & np.isnan(tw1) # Not used at present
sub1 = (tw1 > -40) & np.isnan(tw0)

# Make new variable tw based on 1st column of tw_all, and add data from 2nd column
tw = tw_all[:, 0]
tw[sub1] = tw1[sub1]

# Print temperature arrays for debugging
print(tw_all)
print(tw0)
print(tw1)
print(tw)

# Get unique platform (station) IDs
stList = np.unique(platform_id)
if station_id_requested:
    stList = [station_id_requested]

# Loop through each station to create individual plots
for station_id in stList:
    print(f"Processing station {station_id}...")
    subset = platform_id == station_id
    if not np.any(subset):
        print(f"Skipping station {station_id} (no data found).")
        continue

    # Extract data for the current station
    valid_lon = lon[subset]
    valid_lat = lat[subset]
    valid_time = time[subset]
    valid_tw = tw[subset]
    valid_cTw = cTw[subset]

    print(valid_tw)

    # Remove NaN temperature entries
    not_nan = ~np.isnan(valid_tw)
    valid_tw = valid_tw[not_nan]
    valid_lat = valid_lat[not_nan]
    valid_lon = valid_lon[not_nan]
    valid_time = valid_time[not_nan]
    valid_cTw = valid_cTw[not_nan]

    if len(valid_tw) == 0:
        print(f"Skipping station {station_id} (no valid Tw observations).")
        continue
    
    # Calculate difference regarding valid cTw
    has_valid_cTw = not np.all(np.isnan(valid_cTw))
    valid_diff = valid_tw - valid_cTw if has_valid_cTw else None

    # Print detailed info for requested station
    if station_id_requested:
        print(f"Printing data for station {station_id}:")
        for i in range(len(valid_lat)):
            date_str = pd.to_datetime(valid_time[i]).strftime('%Y-%m-%d %H:%M')
            tw_val = f"{valid_tw[i]:.2f}"
            ctw_val = f"{valid_cTw[i]:.2f}" if not np.isnan(valid_cTw[i]) else "NaN"
            print(f"Lat: {valid_lat[i]:.2f}, Lon: {valid_lon[i]:.2f}, Date: {date_str}, Tw: {tw_val}, cTw: {ctw_val}")

    # Create figure for the track map and time series
    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)

    # Map subplot: track of the drifter
    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
    ax1.set_title(f"Drifter Track - Station {station_id}", fontsize=14)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # Set map extent with buffer
    xmin, xmax = valid_lon.min() - 3, valid_lon.max() + 3
    ymin, ymax = valid_lat.min() - 1, valid_lat.max() + 1
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.coastlines()

    # Plot connecting lines colored by temperature
    for i in range(len(valid_lon) - 1):
        ax1.plot(valid_lon[i:i+2], valid_lat[i:i+2],
                 color=plt.cm.coolwarm((valid_tw[i] + 5) / 10),
                 linewidth=2, alpha=0.8)

    # Scatter plot of individual observations
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

    # Display stats on map
    stats_text = (
        f"Lon: [{xmin:.2f}, {xmax:.2f}]   Lat: [{ymin:.2f}, {ymax:.2f}]\n"
        f"Num obs: {len(valid_tw)}"
    )
    ax1.text(0.5, -0.25, stats_text,
             ha='center', va='top', transform=ax1.transAxes,
             fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # Time series subplot
    ax2 = fig.add_subplot(122)
    ax2.set_title(f"Surface Temperature - Drifter {station_id}", fontsize=14)
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.tick_params(axis='x', rotation=45)

    # Scatter plot of Tw over time, colored by Tw - cTw difference
    # If cTw is defined, give point the color depending on cTw. Else, plot as gray.
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

    # Save the figure to the output directory
    output_path = os.path.join(output_folder, f"{fnamepre}_{station_id}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
