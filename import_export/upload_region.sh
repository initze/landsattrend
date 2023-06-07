#!/bin/bash

conda activate landsattrend2

python /scratch/bbou/toddn/landsat-delta/landsattrend/import_export/upload_input_regions.py $1 $2 $3 $4 $5
