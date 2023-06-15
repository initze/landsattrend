#!/bin/bash

clowderUrl=$1
key=$2
currentRegion=$3
pathToProcessDelta=$4
clowderSpaceId=$5

eval "$(conda shell.bash hook)"
conda activate landsattrend2

python /scratch/bbou/toddn/landsat-delta/landsattrend/import_export/upload_output_zone.py $clowderUrl $key $zone $pathToProcessDelta $clowderSpaceId
