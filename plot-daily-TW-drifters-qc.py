from netCDF4 import Dataset
from datetime import datetime, timedelta
from copy import copy
import re
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

# Set up argument parsing
parser = argparse.ArgumentParser(description="Generate Water Temperature plots at surface level from dataset")
parser.add_argument("-p", '--path', help='Path to the directory of interest')
parser.add_argument("-i", "--input_file", type=str, help="Path to the input NetCDF file (optional)")
parser.add_argument("-lat", "--latitude", type=float, nargs=2, required=True, help="Latitude range (min max)")
parser.add_argument("-lon", "--longitude", type=float, nargs=2, required=True, help="Longitude range (min max)")
parser.add_argument("-s", "--station_id", type=str, help="Station ID to filter (optional)")
parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder for saving plots")
args = parser.parse_args()

# Define input file and output folder from arguments
target_dir = args.path
output_folder = args.output_folder
lat_user = args.latitude
lon_user = args.longitude
station_id_requested = args.station_id
label = "Temperature water [°C]"
fnamepre = "Tw_trackmap"

# List all NetCDF files in the specified directory
nc_files = [f for f in os.listdir(target_dir) if f.endswith(".nc")]
if not nc_files:
    print("No NetCDF files found in the specified directory.")
    exit(1)


# Loop through each file in the directory
for filename in nc_files:
    file_path = os.path.join(target_dir, filename)
    
    # Open the dataset
    ds = xr.open_dataset(file_path)

    # Read variables from the file
    time = ds['time'].values
    lat = ds['lat'].values
    lon = ds['lon'].values
    platform_id = ds['platform_id'].values.astype(str)
    tw = ds['water_temperature'].values  # Expecting [n_obs, depth]
    cTw = ds['climatology_water_temperature'].values
    quality_level = ds['quality_level'].values

    # Use only surface level (depth index 0)
    if tw.ndim > 1:
        tw = tw[:, 0]
        


# Find unique platform IDs
stList = np.unique(platform_id)
print(f"Total number of stations: {len(stList)}")

if station_id_requested:
    stList = [station_id_requested]
    print(station_id_requested)
# Process each station
for station_id in stList:
    print(f"Processing station {station_id}...")

    
    # Filter by station ID
    subset = platform_id == station_id
    if not np.any(subset):
        print(f"Skipping station {station_id} (no data found).")
        continue
    
    # Extract data for the station
    valid_lon = lon[subset]
    valid_lat = lat[subset]
    valid_time = time[subset]
    valid_tw = tw[subset]
    valid_cTw = cTw[subset]
    valid_diff = valid_tw - valid_cTw  
    
    num_obs = len(valid_lat)
    
    # Skip station if any NaNs are present in Tw or cTw
    if np.isnan(valid_tw).any() or np.isnan(valid_cTw).any():
        print(f"Skipping station {station_id} (NaNs in water temperature or climatology).")
        continue

    
    # If requested, print raw data
    if station_id_requested:
        print(f"Printing data for station {station_id}:")
        for i in range(len(valid_lat)):
            observation_date = pd.to_datetime(valid_time[i]).strftime('%Y-%m-%d %H:%M')
            print(f"Lat: {valid_lat[i]:.2f}, Lon: {valid_lon[i]:.2f}, Date: {observation_date}, Tw: {valid_tw[i]:.2f}, cTw: {valid_cTw[i]:.2f}")
    
    # Plotting
    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)
    
    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
    ax1.set_title(f"Drifter Track - Station {station_id}", fontsize=14)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    
    xmin = valid_lon.min() - 3
    xmax = valid_lon.max() + 3
    ymin = valid_lat.min() - 1
    ymax = valid_lat.max() + 1
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
  
    ax1.coastlines()
    
    # Plot track
    for i in range(len(valid_lon) - 1):
        ax1.plot(valid_lon[i:i+2], valid_lat[i:i+2],
                 color=plt.cm.coolwarm((valid_tw[i] + 5) / 10),
                 linewidth=2, alpha=0.8)
    
    sc1 = ax1.scatter(valid_lon, valid_lat, c=valid_tw, cmap='coolwarm',
                      edgecolor='k', s=50, alpha=0.75)
    cbar1 = fig.colorbar(sc1, ax=ax1, orientation='vertical', pad=0.02)
    cbar1.set_label("Tw [°C]")
    
    # Add a box with statistics below the map nicely
    stats_text = (
        f"Lon: [{xmin:.2f}, {xmax:.2f}]   Lat: [{ymin:.2f}, {ymax:.2f}]\n"
        f"Num obs: {num_obs}"
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
    
    sc2 = ax2.scatter(pd.to_datetime(valid_time), valid_tw, c=valid_diff,
                      cmap='coolwarm', edgecolor='k', alpha=0.75)
    ax2.set_ylim(valid_tw.min() - 1, valid_tw.max() + 1)  # dynamic y-axis
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label("Tw - cTw [°C]")
    
    # Save figure
    output_path = os.path.join(output_folder, f"{fnamepre}_{station_id}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()