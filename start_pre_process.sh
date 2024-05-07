#!/bin/bash

# preprocess data for v34, v36 and v39
python preprocessing/convert_parquet_to_json_seqlen_240.py
# preprocess data for v48
python preprocessing/convert_parquet_to_json_deberta_v3_subword_500.py
# preprocess data for v50
python preprocessing/convert_parquet_to_json_deberta_v2_subword_500.py
# preprocess data for v51
python preprocessing/convert_parquet_to_json_albert_subword_500.py