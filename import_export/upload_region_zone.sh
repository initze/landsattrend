#!/bin/bash

clowderUrl=$1
key=$2
pathToProcessDelta=$3
clowderSpaceId=$4
currentRegion=$5

eval "$(conda shell.bash hook)"
conda activate landsattrend2

python /scratch/bbou/toddn/landsat-delta/landsattrend/import_export/upload_output_zone.py $clowderUrl $key $pathToProcessDelta $clowderSpaceId $zone
