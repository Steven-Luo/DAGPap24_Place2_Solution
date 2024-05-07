import pandas as pd
from collections import Counter
import os
from tqdm.auto import tqdm, trange

if __name__ == '__main__':
    output_root = 'outputs/stage2_predictions'
    pred_versions = [
        os.path.join('v34', 'post_process_stage2'),
        os.path.join('v36', 'post_process_stage2'),
        os.path.join('v39', 'post_process_stage2')
    ]
    prefered_version = os.path.join('v36', 'post_process_stage2')
    ensemble_version = 'e3'

    dfs = {}
    for version in tqdm(pred_versions):
        filepath = os.path.join(output_root, version, 'predictions.parquet')
        dfs[version] = pd.read_parquet(filepath, engine='fastparquet')
        
    vote_df = dfs[prefered_version].copy(deep=True)

    def vote(arr, prefer_val=None):
        freq = Counter(arr).most_common()
        if len(freq) == 1:
            return freq[0][0], False
        if freq[0][1] == freq[1][1]:
            return prefer_val if prefer_val is not None else freq[0][0], True
        return freq[0][0], False

    update_count = 0
    total_equal_freq = 0
    total_seq_len = 0

    for idx, row in tqdm(vote_df.iterrows(), total=len(vote_df)):
        preds = [dfs[version].loc[idx, 'preds'] for version in pred_versions]
        prefer_preds = dfs[prefered_version].loc[idx, 'preds']
        total_seq_len += len(prefer_preds)
        
        vote_result = []
        for token_idx, _ in enumerate(vote_df.at[idx, 'preds']):
            mojarity, has_equal_freq = vote([pred[token_idx] for pred in preds], prefer_val=prefer_preds[token_idx])
            total_equal_freq += int(has_equal_freq)
            vote_result.append(mojarity)
        assert len(vote_result) == len(vote_df.at[idx, 'preds'])
        vote_df.at[idx, 'preds'] = vote_result
        if vote_df.at[idx, 'preds'] != dfs[prefered_version].at[idx, 'preds']:
            update_count += 1

    print(f"update percentage: {update_count * 100 / len(vote_df):.2f}%, total equal freq count: {total_equal_freq}, equal freq percentage: {total_equal_freq * 100 / total_seq_len:.4f}%")

    output_dir = os.path.join(output_root, ensemble_version)
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(vote_df['preds']).to_parquet(os.path.join(output_dir, 'predictions.parquet'))

