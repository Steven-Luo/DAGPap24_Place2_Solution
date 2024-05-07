import pandas as pd
import os
from tqdm.auto import tqdm

version = os.environ['VERSION']
window_lengths = os.environ['WINDOW_LENGTHS']
window_lengths = [int(val) for val in window_lengths.split(',')]

print(f"version={version}, window_lengths={window_lengths}")

output_dir = (
    f"output_dev/{version}"
)

def prepare_submission():
    print("prepare submission...")
    
    path_to_test_data = 'data/test_data.parquet'
    path_to_test_trunc_data = 'data/data_gen_content_test_240.json'
    path_to_test_preds = os.path.join(output_dir, 'test_predictions.json')

    orig_test_data = pd.read_parquet(path_to_test_data, engine="fastparquet")
    if orig_test_data.index.name != "index":
        orig_test_data.set_index("index", inplace=True)
        
    txt_test_data = open(path_to_test_preds.replace('.json', '.txt')).readlines()
    txt_test_data = [[int(val) for val in line.split()] for line in txt_test_data if line.strip() != '']
    test_trunc_df = pd.read_json(path_to_test_trunc_data)
    
    assert len(test_trunc_df) == len(txt_test_data)
    all_predictions = []

    padding_zero_count = 0
    for idx, row in test_trunc_df.iterrows():
        if len(txt_test_data[idx]) != len(row['tokens']):
            padding_zero_count += 1
            txt_test_data[idx].extend([0] * (len(row['tokens']) - len(txt_test_data[idx])))
            # break
        assert len(txt_test_data[idx]) == len(row['ner_tags']) == len(row['tokens'])
        all_predictions.append(txt_test_data[idx])
    print(f"padding zero percentage: {padding_zero_count * 100 / len(test_trunc_df):.2f}%")
    test_trunc_df['predictions'] = all_predictions
    
    test_trunc_group_df = test_trunc_df.groupby('index')['predictions'].apply(lambda x: sum(x.tolist(), [])).reset_index()
    assert len(test_trunc_group_df) == len(orig_test_data)

    for _, row in test_trunc_group_df.iterrows():
        index = row['index']
        assert len(row['predictions']) == len(orig_test_data.loc[index]['tokens'])
    
    test_trunc_group_df = test_trunc_group_df.rename(columns={'predictions': 'preds'})
    test_trunc_group_df.index = test_trunc_group_df['index']
    _ = test_trunc_group_df.pop('index')
    
    for index, row in test_trunc_group_df.iterrows():
        if len(row["preds"]) > len(orig_test_data.loc[index, "tokens"]):
            test_trunc_group_df.at[index, "preds"] = row["preds"][
                : len(orig_test_data.loc[index, "tokens"])
            ]

        elif len(row["preds"]) < len(orig_test_data.loc[index, "tokens"]):
            test_trunc_group_df.loc[index, "preds"] = row["preds"] + 0 * (
                len(orig_test_data.loc[index, "tokens"]) - len(row["preds"])
            )

    for index, row in test_trunc_group_df.iterrows():
        assert len(row["preds"]) == len(orig_test_data.loc[index, "tokens"])

    pd.DataFrame(test_trunc_group_df["preds"]).to_parquet(os.path.join(output_dir, "predictions.parquet"))
    
    
def do_post_process():
    df = pd.read_parquet(f"{output_dir}/predictions.parquet", engine="fastparquet")
    assert len(df) == 20000

    for win_len in window_lengths:
        all_new_preds = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"window={win_len}"):
            now = row["preds"][0]
            pre = -1
            now_list = [now]
            new_preds = []

            for idx, label in enumerate(row["preds"][1:]):
                if label != now or idx == len(row["preds"][1:]) - 1:
                    if idx == len(row["preds"][1:]) - 1:
                        now_list.append(now_list[-1])
                    if len(now_list) < win_len:
                        if pre != -1:
                            now_list = len(now_list) * [pre]
                        else:
                            now_list = len(now_list) * [label]
                    pre = now_list[-1]
                    new_preds.extend(now_list)
                    now_list = [label]
                else:
                    now_list.append(label)
                now = label

            assert len(new_preds) == len(row["preds"])
            row["preds"] = new_preds
            all_new_preds.append(new_preds)

        df["preds"] = all_new_preds

    post_output_dir = f"{output_dir}/post_process_stage2"
    os.makedirs(post_output_dir, exist_ok=True)

    pd.DataFrame(df["preds"]).to_parquet(f"{post_output_dir}/predictions.parquet")


if __name__ ==  '__main__':
    prepare_submission()
    do_post_process()