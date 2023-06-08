#!/bin/bash

clowderUrl=$1
key=$2
currentRegion=$3
pathToProcessDelta=$4
clowderSpaceId=$5

conda activate landsattrend2

python /scratch/bbou/toddn/landsat-delta/landsattrend/import_export/upload_output_regions.py $clowderUrl $key $currentRegion $pathToProcessDelta $clowderSpaceId
