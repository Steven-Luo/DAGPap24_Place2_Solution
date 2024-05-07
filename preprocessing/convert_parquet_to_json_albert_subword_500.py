# %%
from tqdm.auto import tqdm, trange
import json

import os
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
import yaml

from collections import Counter

from transformers import AutoTokenizer 

model_name_or_path = "albert/albert-xxlarge-v2"
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=True,
    add_prefix_space=True,
    # revision=model_args.model_revision,
    use_auth_token=True
)

def chunk_tokens_labels(df: pd.DataFrame, max_length: int):
    """
    This function chunks tokens and their respective labels to
    max_length token length
    """
    index_list = []
    tokens_list = []
    labels_list = []
    max_subword_length = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        subword_length_now = 0
        l, r = 0, 0
        total_subword_length, total_token_length = 0, 0
        for i, token in enumerate(row["tokens"]):
            now = len(tokenizer(token)["input_ids"][1:-1])
            if subword_length_now+now >= max_length:   
                max_subword_length = max(max_subword_length, subword_length_now)
                l, r = r, i       
                index_list.append(index)
                tokens_list.append(row["tokens"][l:r])
                labels_list.append(row["token_label_ids"][l:r])
                total_token_length += len(row["tokens"][l:r])
                total_subword_length += subword_length_now
                subword_length_now = 0
            subword_length_now += now
        
        # Append last chunk
        total_subword_length += subword_length_now
        total_token_length += len(row["tokens"][r:])
        index_list.append(index)
        tokens_list.append(row["tokens"][r:])
        labels_list.append(row["token_label_ids"][r:])
        assert len(row["tokens"]) == total_token_length
        assert len(tokenizer(row["tokens"], is_split_into_words=True)["input_ids"][1:-1]) == total_subword_length
    print("max_subword_length", max_subword_length)

    return pd.DataFrame(
        {"index": index_list, "tokens": tokens_list, "labels": labels_list}
    )

def write_df_to_json(df: pd.DataFrame, path_to_json: str):
    """
    This function writes pandas dataframes into a compatible json format
    to be used by hf_token_classification.py
    """
    index_list = df["index"].values.tolist()
    tokens_list = df["tokens"].values.tolist()
    labels_list = df["labels"].values.tolist()
    data_list = []
    for i in tqdm(range(len(tokens_list)), total=len(tokens_list)):
        index = index_list[i]
        tokens = tokens_list[i]
        labels = [str(el) for el in labels_list[i]]
        data_list.append(
            {"index": index, "tokens": tokens, "ner_tags": labels}
        )
    with open(path_to_json, "w") as f:
        f.write(json.dumps(data_list))


def prep_data(path_to_file: str, max_length: int, test: bool = False):
    if test == False:
        dataset = "train"
    else:
        dataset = "test"

    print(f"Loading {dataset} dataset from file")
    df = pd.read_parquet(path_to_file, engine="fastparquet")
    if df.index.name != "index":
        df.set_index("index", inplace=True)

    # the external NER Classification script needs a target column
    # for the test set as well, which is not available.
    # Therefore, we're subsidizing this column with a fake label column
    # Which we're not using anyway, since we're only using the test set
    # for predictions.
    if "token_label_ids" not in df.columns:
        df["token_label_ids"] = df["tokens"].apply(
            lambda x: np.zeros(len(x), dtype=int)
        )
    df = df[["tokens", "token_label_ids"]]

    print(f"Initial {dataset} data length: {len(df)}")
    df = chunk_tokens_labels(df, max_length=max_length)
    print(
        f"{dataset} data length after chunking to max {max_length} tokens: {len(df)}"
    )

    return df

def convert_parquet_data_to_json(
    input_folder_path: str,
    input_train_file_name: str,
    input_test_file_name: str,
    max_length: int,
    val_size: float = 0.1,
    output_train_file_name: str = "",
    output_val_file_name: str = "",
    output_test_file_name: str = "",
    seed: int = 0,
):
    """
    This function takes a parquet file with (at least) the text split and the token
    label ids, and converts it to train, validation, and test data.
    Each chunk is saved as a separate json file

    param input_folder_path: path to data dir
    param input_train_file_name: the input train file name
    param input_test_file_name: the input test file name
    param max_length: max token length
    param val_size: validation size as a fraction of the total
    param output_train_file_name: json train file name
    param output_val_file_name: json validation file name
    param output_test_file_name: json test file name

    returns: None
    """

    print("Loading train and test datasets")

    # Loading and prepping train dataset
    train_df = prep_data(
        path_to_file=Path(input_folder_path) / Path(input_train_file_name),
        max_length=max_length,
        test=False,
    )
    # Loading and prepping train dataset
    test_df = prep_data(
        path_to_file=Path(input_folder_path) / Path(input_test_file_name),
        max_length=max_length,
        test=True,
    )

    print("Splitting train data into train and validation splits")
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=seed, shuffle=True
    )

    print(f"Final train size: {len(train_df)}")
    print(f"Final validation size: {len(val_df)}")
    print(f"Final test size: {len(test_df)}")

    print("Writing train df to json...")
    write_df_to_json(
        train_df,
        f"{input_folder_path}/{output_train_file_name}",
    )
    print("Writing val df to json...")
    write_df_to_json(val_df, f"{input_folder_path}/{output_val_file_name}")
    print("Writing test df to json...")
    write_df_to_json(
        test_df,
        f"{input_folder_path}/{output_test_file_name}",
    )

if __name__ == '__main__':
    convert_parquet_data_to_json(
        'data',
        input_train_file_name='train_data.parquet',
        input_test_file_name='test_data.parquet',
        max_length=500,
        output_train_file_name='data_gen_content_train_subword_albert_500.json',
        output_val_file_name='data_gen_content_val_subword_albert_500.json',
        output_test_file_name='data_gen_content_test_subword_albert_500.json'
    )
