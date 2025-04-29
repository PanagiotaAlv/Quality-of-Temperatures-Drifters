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
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Generate SST or IST difference plots from dataset")
parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input NetCDF file")
parser.add_argument("-v", "--input_var", type=str, default='sst', choices=['sst','ist'], help="Temperature variable to be plotted")
parser.add_argument("-s", "--station_id", type=str, help="Station ID to plot (optional)")
parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to the output folder for saving plots")
args = parser.parse_args()

# Define input file and output folder from arguments
input_file = args.input_file
output_folder = args.output_folder
station_id_requested = args.station_id
temp_var = args.input_var

colormap = "coolwarm"

# Open the dataset
ds = xr.open_dataset(input_file)

# Read all necessary variables
years = ds.variables['yys'].values
months = ds.variables['mms'].values
days = ds.variables['dds'].values
hours = ds.variables['hhs'].values
minutes = ds.variables['mis'].values
stType = ds.variables['stType'].values.astype(str)
stID = ds.variables['stID'].values.astype(str)
tw = ds.variables['Tw'].values[:, 0]
sst = ds.variables['sea_surface_temperature'].values[:, 2, 2]
ist = ds.variables['surface_temperature'].values[:, 2, 2]
lon = ds.variables['longitude'].values  
lat = ds.variables['latitude'].values  


hours_fixed = np.where((hours >= 0) & (hours < 24), hours, 0)
minutes_fixed = np.where((minutes >= 0) & (minutes < 60), minutes, 0)

# Create time dataframe
time_df = pd.DataFrame({
    "year": years,
    "month": months,
    "day": days,
    "hour": hours_fixed,
    "minute": minutes_fixed
})

time = pd.to_datetime(time_df, errors='coerce')


# Define the choises 
if temp_var == 'ist':
    diff = ist - tw
    tempvar = ist
    title1 = 'ISTsat - ISTinsitu for drifter'
    label = "ISTsat - ISTinsitu [°C]"
    fnamepre = "ISTdiff_trackmap"
else:
    diff = sst - tw
    tempvar = sst
    title1 = 'SSTsat - SSTinsitu for drifter'
    label = "SSTsat - SSTinsitu [°C]"
    fnamepre = "SSTdiff_trackmap"


stList = np.unique(stID[np.logical_or(stType == 'SYNOP',stType == 'drifter')])
print(f"Total number of stations: {len(stList)}")

# If a specific station ID is requested, filter to only that station
if station_id_requested:
    stList = [station_id_requested]

# Loop through each station
for station_id in stList:
    print(f"Processing station {station_id}...")

    # Select valid data
    subset = np.logical_and.reduce((stID == station_id,abs(diff) < 299,time.notna()))

    if not np.any(subset):
        print(f"Skipping station {station_id} (no valid data).")
        continue

    # Extract valid data
    valid_lon = lon[subset]
    valid_lat = lat[subset]
    valid_time = time[subset]
    valid_tw = tw[subset]
    valid_diff = diff[subset]
    
    
    # Calculate the statistics
    diff_mean =valid_diff.mean()
    diff_std = valid_diff.std()
    num_obs = len(valid_lat)

    
   # Print data for the specific station if -s flag is used
    if station_id_requested:
        print(f"Printing data for station {station_id}:")
        for i in range(len(valid_lat)):
            observation_date = valid_time.iloc[i].strftime('%Y-%m-%d %H:%M')  # Use iloc to access Series element
            print(f"Lat: {valid_lat[i]:.2f}, Lon: {valid_lon[i]:.2f}, Date: {observation_date}, Tsat: {tempvar[subset][i]:.2f}, Tw: {valid_tw[i]:.2f}")
          

    # Create a new figure
    fig = plt.figure(figsize=(12, 6))
    
    # Adjust subplot layout
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)
    
    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())  # Left subplot
    ax2 = fig.add_subplot(122)  # Right subplot
    
    # Set up the left subplot: Drifter Track on Map
    ax1.set_title(f"Drifter Track - Station {station_id}", fontsize=14)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    xmin = valid_lon.min() - 8
    xmax = valid_lon.max() + 8
    ymin = valid_lat.min() - 3
    ymax = valid_lat.max() + 3
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
  
    ax1.coastlines()
    
    # Plot track colored by SST difference
    for i in range(len(valid_lon) - 1):
        ax1.plot(valid_lon[i:i+2], valid_lat[i:i+2], color=plt.cm.coolwarm((valid_diff[i] + 5) / 10),
                 linewidth=2, alpha=0.8)

    # Scatter plot to show individual points
    sc1 = ax1.scatter(valid_lon, valid_lat, c=valid_diff, cmap='coolwarm', edgecolor='k', s=50, alpha=0.75)


    # Add colorbar for SST difference
    cbar1 = fig.colorbar(sc1, ax=ax1, orientation='vertical', pad=0.02)


    # Add a box with statistics below the map nicely
    stats_text = (
        f"Lon: [{xmin:.2f}, {xmax:.2f}]   Lat: [{ymin:.2f}, {ymax:.2f}]\n"
        f"Mean diff: {diff_mean:.2f}  Std: {diff_std:.2f}  Num: {num_obs}"
    )
    ax1.text(0.5, -0.25, stats_text,
             ha='center', va='top', transform=ax1.transAxes,
             fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    
    ax2.set_title(f"{title1} {station_id}", fontsize=14)
    ax2.set_xlabel("Date")
    ax2.set_ylabel(label)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.tick_params(axis='x', rotation=45)
    
    # Scatter plot for SST/IST difference over time
    sc2 = ax2.scatter(valid_time, valid_diff, c=valid_tw, cmap='coolwarm', edgecolor='k', alpha=0.75)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    
    # Add colorbar for Tw
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label("Tw [K]")


    # Save plot
    output_path = os.path.join(output_folder, f"{fnamepre}_{station_id}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
   
    # close plt before making a new figure
    plt.close()
