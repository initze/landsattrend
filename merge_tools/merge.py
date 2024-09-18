# Merge the two lake change datasets from Ingmar Nitze
# Author: Juliet Cohen
# Date: 07/26/23
# use conda environment "geospatial"

import geopandas as gpd
import xarray as xar
import pandas as pd
import os

# Lake area time series data GeoPackage
path_to_gpkg= os.path.join(os.getcwd(), 'merge_tools', "lake_change_cleaned_Z004.gpkg")
path_to_test_gpkg = os.path.join(os.getcwd(), 'merge_tools', "INitze_Lakes_merged_v3_PDG_Testset.gpkg")
gdf = gpd.read_file(path_to_test_gpkg)
# Lake area time series data NetCDF
path_to_area= os.path.join(os.getcwd(), 'merge_tools', "lake_change_cleaned_Z004.nc")
path_to_test_area= os.path.join(os.getcwd(), 'merge_tools', "Lakes_IngmarPDG_annual_area.nc")
area = xar.open_dataset(path_to_test_area)

# convert the NetCDF file into a dataframe, with columns for:
# ID_merged, year, permanent_water, seasonal_water
area_df = area.to_dataframe().reset_index()

# merge the dataframe, retaining only ID_merged values (lakes) that exist in both files
# the spatial dataframe must be the left argument to retain the geometries
merged_data = gdf.merge(right = area_df,
                        how = 'inner', # only retain lakes with data for the measurements for permanent water and seasonal water
                        on = 'ID_merged')

# Save as a gpkg for input into the viz-workflow
path_to_new_gpkg = os.path.join(os.getcwd(), 'merge_tools',"merged_lakes_area.gpkg" )
merged_data.to_file(path_to_new_gpkg, driver = "GPKG")