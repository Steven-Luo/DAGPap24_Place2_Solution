#!/bin/bash

for VERSION in v34 v36 v39
do
  export WINDOW_LENGTHS=10,20
  export VERSION=$VERSION
  python postprocessing/post_process.py
done

for VERSION in v48 v50 v51
do
  export WINDOW_LENGTHS=10,20,30
  export VERSION=$VERSION
  python postprocessing/post_process.py
done

python postprocessing/e3_ensemble.py
python postprocessing/e5_ensemble.py

export VERSION=e5
export WINDOW_LENGTHS=5
python postprocessing/post_process.py