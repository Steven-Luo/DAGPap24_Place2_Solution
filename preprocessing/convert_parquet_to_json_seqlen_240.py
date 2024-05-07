from tqdm.auto import tqdm, trange
import json

import os
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
import yaml

from collections import Counter

def chunk_tokens_labels(df: pd.DataFrame, max_length: int):
    """
    This function chunks tokens and their respective labels to
    max_length token length
    """
    index_list = []
    tokens_list = []
    labels_list = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if len(row["token_label_ids"]) > max_length:
            remaining_tokens = row["tokens"]
            remaining_labels = row["token_label_ids"]

            # While the remaining list is larger than max_length,
            # truncate and append
            while len(remaining_labels) > max_length:
                index_list.append(index)
                tokens_list.append(remaining_tokens[:max_length])
                labels_list.append(remaining_labels[:max_length])
                remaining_tokens = remaining_tokens[max_length:]
                remaining_labels = remaining_labels[max_length:]
            # Append last chunk
            index_list.append(index)
            tokens_list.append(remaining_tokens)
            labels_list.append(remaining_labels)
        else:
            index_list.append(index)
            tokens_list.append(row["tokens"])
            labels_list.append(row["token_label_ids"])

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

if __name__ == "__main__":
    max_length = 240

    convert_parquet_data_to_json(
        'data',
        input_train_file_name='train_data.parquet',
        input_test_file_name='test_data.parquet',
        max_length=max_length,
        output_train_file_name=f'data_gen_content_train_stage2_{max_length}.json',
        output_val_file_name=f'data_gen_content_val_stage2_{max_length}.json',
        output_test_file_name=f'data_gen_content_test_stage2_{max_length}.json'
    )
