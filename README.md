# Environments

- Python Version: 3.11.5
- CUDA version: 11.7

# Steps

## Download data

Download the data from [here](https://www.codabench.org/competitions/2431/#/pages-tab) and unzip the data into `./data` folder.

There should be `train_data.parquet` and `test_test.parquet` files in data folder after unzipping.

## Installing Packagess

```bash
pip install -r requirements.txt
```

## Preprocessing

Tokenizer of `albert/albert-xxlarge-v2`, `microsoft/deberta-v3-large` and `microsoft/deberta-v2-xxlarge`  will be used, you can modify the local path of the tokenizer in `preprocessing/convert_parquet_to_json_albert_subword_500.py`, `preprocessing/convert_parquet_to_json_deberta_v2_subword_500.py` and `preprocessing/convert_parquet_to_json_deberta_v3_subword_500.py` resrespectively.

In the project root, run the following command to preprocess the data:

```bash
sh start_pre_process.sh
```

After the script finished, following files will be generated:

- data/data_gen_content_train_240.json
- data/data_gen_content_val_240.json
- data/data_gen_content_test_240.json
- data/data_gen_content_train_subword_deberta_v2_500.json
- data/data_gen_content_val_subword_deberta_v2_500.json
- data/data_gen_content_test_subword_deberta_v2_500.json
- data/data_gen_content_train_subword_deberta_v3_500.json
- data/data_gen_content_val_subword_deberta_v3_500.json
- data/data_gen_content_test_subword_deberta_v3_500.json
- data/data_gen_content_train_subword_albert_500.json
- data/data_gen_content_val_subword_albert_500.json
- data/data_gen_content_test_subword_albert_500.json

## Train models

In the project root, run `start_train.sh` followed by model version to start training. For example, to train v34 model:

```bash
sh start_train.sh v34
```

After the script finished, model weights and test predictions will be saved in `./output/v34` folder.

## Postprocessing

In this step, the corresponding version of the `test_prediction.txt` file will be read and postprocessed to generate the final result.

In the project root, run the following command to postprocess the data and generate the final result:

```bash
sh start_post_process.sh
```
