#Landsattrend

###Python package to process robust trends of Landsat image stacks
This package contains several modules 
* to preprocess downloaded and zipped ready to use (e.g. TOA or SR) Landsat data
* to calculate robust trends (Theil Sen) of multispectral indices
* to export the data to raster files
* to mosaic produced tiles to larger maps

###The type of files processeed

When run as an extractor, the files processed are of this form

trendimage_{year1-year2}_{sitename}_{lat}_{lon}.tif

This extractor runs on the dataset level - all files will be processed

