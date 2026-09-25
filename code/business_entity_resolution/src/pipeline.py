import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

from .blocking import generate_candidates_blocking, merge_candidates
from .features import prepare_training_data, build_feature_matrix
from .model import train_model, find_optimal_threshold, predict_candidates, save_model, load_model
from .data_loader import load_data_cached


def load_data(data_dir: str, sample_frac: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                       pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    
    print(f"Train: S1={len(s1_train)}, S2={len(s2_train)}, S3={len(s3_train)}, GT={len(gt_train)}")
    print(f"Test: S1={len(s1_test)}, S2={len(s2_test)}, S3={len(s3_test)}")
    
    return s1_train, s2_train, s3_train, gt_train, s1_test, s2_test, s3_test


def generate_training_candidates(s1_train: pd.DataFrame, s2_train: pd.DataFrame, 
                                  s3_train: pd.DataFrame) -> Dict[str, List[str]]:
    print("Generating training candidates...")
    candidates = generate_candidates_blocking(s1_train, s2_train, s3_train)
    total_cands = sum(len(v) for v in candidates.values())
    print(f"Total training candidates: {total_cands}")
    print(f"Avg candidates per S1: {total_cands / len(candidates):.1f}")
    return candidates


def generate_test_candidates(s1_test: pd.DataFrame, s2_test: pd.DataFrame, 
                              s3_test: pd.DataFrame) -> Dict[str, List[str]]:
    print("Generating test candidates...")
    candidates = generate_candidates_blocking(s1_test, s2_test, s3_test)
    total_cands = sum(len(v) for v in candidates.values())
    print(f"Total test candidates: {total_cands}")
    print(f"Avg candidates per S1: {total_cands / len(candidates):.1f}")
    return candidates


def train_pipeline(s1_train: pd.DataFrame, s2_train: pd.DataFrame, 
                    s3_train: pd.DataFrame, gt_train: pd.DataFrame,
                    model_path: str) -> Tuple[object, float]:
    print("\n=== TRAINING PIPELINE ===")
    
    train_candidates = generate_training_candidates(s1_train, s2_train, s3_train)
    
    X, y = prepare_training_data(s1_train, s2_train, s3_train, gt_train, train_candidates)
    
    if X.empty:
        raise ValueError("No training data generated!")
    
    if y.sum() == 0:
        print("WARNING: No positive samples in training data! Creating dummy model.")
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy='constant', constant=0)
        model.fit(X, y)
        threshold = 0.5
        save_model(model, threshold, model_path)
        return model, threshold
    
    stratify = y if y.sum() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=stratify
    )
    
    model = train_model(X_train, y_train, X_val, y_val)
    
    threshold = find_optimal_threshold(model, X_val, y_val)
    
    save_model(model, threshold, model_path)
    
    return model, threshold


def inference_pipeline(s1_test: pd.DataFrame, s2_test: pd.DataFrame, 
                        s3_test: pd.DataFrame, model, threshold: float) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    print("\n=== INFERENCE PIPELINE ===")
    
    test_candidates = generate_test_candidates(s1_test, s2_test, s3_test)
    
    s23_test = pd.concat([s2_test, s3_test], ignore_index=True)
    
    feature_df, pairs = build_feature_matrix(s1_test, s23_test, test_candidates)
    
    if feature_df.empty:
        print("No candidates generated!")
        return {sid: [] for sid in s1_test['entity_id']}, test_candidates
    
    feature_cols = [c for c in feature_df.columns if c not in ['source1_entity_id', 'candidate_entity_id']]
    X_test = feature_df[feature_cols]
    
    matches = predict_candidates(model, X_test, pairs, threshold)
    
    all_s1_ids = s1_test['entity_id'].tolist()
    for sid in all_s1_ids:
        if sid not in matches:
            matches[sid] = []
    
    return matches, test_candidates


def write_output(matches: Dict[str, List[str]], candidates: Dict[str, List[str]], 
                  output_dir: str):
    print("\n=== WRITING OUTPUT ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    matching_rows = []
    for s1_id in sorted(matches.keys()):
        matched = sorted(matches[s1_id])
        matching_rows.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ','.join(matched)
        })
    
    matching_df = pd.DataFrame(matching_rows)
    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    matching_df.to_csv(matching_path, sep='\t', index=False)
    print(f"Written {len(matching_df)} rows to {matching_path}")
    
    candidate_rows = []
    for s1_id in sorted(candidates.keys()):
        cands = sorted(candidates[s1_id])
        candidate_rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(cands)
        })
    
    candidate_df = pd.DataFrame(candidate_rows)
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    candidate_df.to_csv(candidate_path, sep='\t', index=False)
    print(f"Written {len(candidate_df)} rows to {candidate_path}")


def run_validation(output_dir: str, test_dir: str):
    print("\n=== RUNNING VALIDATION ===")
    import subprocess
    script_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'student_resource', 'utils', 'validate_submission.py')
    result = subprocess.run([
        'python', script_path,
        '--matching', os.path.join(output_dir, 'matching_results.tsv'),
        '--candidate', os.path.join(output_dir, 'candidate_pairs.tsv'),
        '--test-dir', test_dir
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0


def main():
    import sys
    sample_frac = 0.001
    test_sample_frac = 0.001
    if len(sys.argv) > 1:
        sample_frac = float(sys.argv[1])
    if len(sys.argv) > 2:
        test_sample_frac = float(sys.argv[2])
    
    data_dir = '../../student_resource/dataset'
    output_dir = '../../output'
    model_path = '../../model.joblib'
    
    s1_train, s2_train, s3_train, gt_train, s1_test, s2_test, s3_test = load_data_cached(data_dir, sample_frac, test_sample_frac)
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        model, threshold = load_model(model_path)
    else:
        model, threshold = train_pipeline(s1_train, s2_train, s3_train, gt_train, model_path)
    
    matches, candidates = inference_pipeline(s1_test, s2_test, s3_test, model, threshold)
    
    write_output(matches, candidates, output_dir)
    
    run_validation(output_dir, os.path.join(data_dir, 'test'))
    
    print("\n=== DONE ===")


if __name__ == '__main__':
    main()