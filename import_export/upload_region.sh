#!/bin/bash

#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate landsattrend2

clowderUrl=$1
key=$2
currentRegion=$3
pathToInputDelta=$4
clowderSpaceId=$5


python /scratch/bbou/toddn/landsat-delta/landsattrend/import_export/upload_input_regions.py  $clowderUrl $key $currentRegion $pathToInputDelta $clowderSpaceId
