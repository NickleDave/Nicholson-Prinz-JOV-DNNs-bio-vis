#!/bin/bash

SCRIPT=src/scripts/experiment-2-VSD/convert-results-gz-to-csv.py

TARGETS=('car'
'cat'
'chair'
'dog'
'person')

for TARGET in "${TARGETS[@]}"; do
  echo "generating source .csv from results.gz files for target: ${TARGET}"

  python $SCRIPT \
    --results_gz_root results/VSD/results_gz/${TARGET} \
    --source_data_root results/VSD/source_data/${TARGET} \
    --methods transfer --modes classify \
    --alexnet_split_csv_path data/Visual_Search_Difficulty_v1.0/VSD-dataset-split-searchstims-${TARGET}.csv \
    --VGG16_split_csv_path data/Visual_Search_Difficulty_v1.0/VSD-dataset-split-searchstims-${TARGET}.csv

done
