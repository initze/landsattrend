# FOR RUNNING 07-02_Lake_Analysis_Z056_local_ray.py

this is way to run multiple sites for a time period locally using ray

install dependencies in environment_py38_v2_ray.yml

you can start ray locally using 'ray start'

or if you are connecting to another cluster, follow instructions in kuberay folder 
(note this option does not work yet on the remote cluster)

the file (site_file_list.txt) is included with 2 zones. More can be added to run more zones
this is a tentative solution for running multiple sites. 


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

