import pandas as pd
import os
import pickle
from typing import Tuple


def load_data_cached(data_dir: str, sample_frac: float = 1.0, test_sample_frac: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                       pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_dir = os.path.join(data_dir, '..', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f'data_sample_{sample_frac}_test_{test_sample_frac}.pkl')
    
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Loading data...")
    
    print("  Reading train_source1...")
    s1_train = pd.read_csv(os.path.join(data_dir, 'train', 'train_source1.tsv'), sep='\t')
    print(f"  Read {len(s1_train)} rows")
    
    print("  Reading train_source2...")
    s2_train = pd.read_csv(os.path.join(data_dir, 'train', 'train_source2.tsv'), sep='\t')
    print(f"  Read {len(s2_train)} rows")
    
    print("  Reading train_source3...")
    s3_train = pd.read_csv(os.path.join(data_dir, 'train', 'train_source3.tsv'), sep='\t')
    print(f"  Read {len(s3_train)} rows")
    
    print("  Reading train_ground_truth...")
    gt_train = pd.read_csv(os.path.join(data_dir, 'train', 'train_ground_truth.tsv'), sep='\t')
    print(f"  Read {len(gt_train)} rows")
    
    print("  Reading test_source1...")
    s1_test = pd.read_csv(os.path.join(data_dir, 'test', 'test_source1.tsv'), sep='\t')
    print(f"  Read {len(s1_test)} rows")
    
    print("  Reading test_source2...")
    s2_test = pd.read_csv(os.path.join(data_dir, 'test', 'test_source2.tsv'), sep='\t')
    print(f"  Read {len(s2_test)} rows")
    
    print("  Reading test_source3...")
    s3_test = pd.read_csv(os.path.join(data_dir, 'test', 'test_source3.tsv'), sep='\t')
    print(f"  Read {len(s3_test)} rows")
    
    if sample_frac < 1.0:
        print(f"Sampling {sample_frac*100:.2f}% of TRAINING data only...")
        s1_train = s1_train.sample(frac=sample_frac, random_state=42)
        s2_train = s2_train.sample(frac=sample_frac, random_state=42)
        s3_train = s3_train.sample(frac=sample_frac, random_state=42)
        gt_train = gt_train[gt_train['source1_entity_id'].isin(s1_train['entity_id'])]
    
    if test_sample_frac < 1.0:
        print(f"Sampling {test_sample_frac*100:.2f}% of TEST data...")
        s1_test = s1_test.sample(frac=test_sample_frac, random_state=42)
        s2_test = s2_test.sample(frac=test_sample_frac, random_state=42)
        s3_test = s3_test.sample(frac=test_sample_frac, random_state=42)
    
    print(f"Train: S1={len(s1_train)}, S2={len(s2_train)}, S3={len(s3_train)}, GT={len(gt_train)}")
    print(f"Test: S1={len(s1_test)}, S2={len(s2_test)}, S3={len(s3_test)}")
    
    data = (s1_train, s2_train, s3_train, gt_train, s1_test, s2_test, s3_test)
    
    print(f"Saving cache to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    return data