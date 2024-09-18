#!/bin/bash
region="$1"
STARTYEAR="$2"
ENDYEAR="$3"
region=$(echo $region | tr 'a-z' 'A-Z')
echo ${region}
echo ${PWD}
for zone in $(cat ${PWD}/${region}_zones.txt);
do
  echo ${zone}
  python /scratch/bbou/toddn/landsat-delta/landsattrend/07-02_LakeAnalysis_Z056_local.py --current_site_name=${zone} --startyear=${STARTYEAR} --endyear=${ENDYEAR} --process_root=/scratch/bbou/toddn/landsat-delta/landsattrend
done