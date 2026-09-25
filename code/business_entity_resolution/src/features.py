import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re
from collections import Counter


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def get_char_ngrams(text: str, n: int = 3) -> List[str]:
    text = normalize_text(text)
    if len(text) < n:
        return [text]
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def cosine_similarity(vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
    if not vec1 or not vec2:
        return 0.0
    dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1) & set(vec2))
    norm1 = np.sqrt(sum(v*v for v in vec1.values()))
    norm2 = np.sqrt(sum(v*v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def extract_features_fast(s1_row: pd.Series, s23_row: pd.Series) -> Dict[str, float]:
    features = {}
    
    s1_name = normalize_text(s1_row['business_name'])
    s23_name = normalize_text(s23_row['business_name'])
    s1_addr = normalize_text(s1_row['business_address'])
    s23_addr = normalize_text(s23_row['business_address'])
    s1_country = str(s1_row['country']).strip().lower()
    s23_country = str(s23_row['country']).strip().lower()
    
    features['same_country'] = 1.0 if s1_country == s23_country else 0.0
    
    s1_name_tokens = set(tokenize(s1_name))
    s23_name_tokens = set(tokenize(s23_name))
    s1_addr_tokens = set(tokenize(s1_addr))
    s23_addr_tokens = set(tokenize(s23_addr))
    
    features['name_jaccard_word'] = jaccard_similarity(s1_name_tokens, s23_name_tokens)
    features['addr_jaccard_word'] = jaccard_similarity(s1_addr_tokens, s23_addr_tokens)
    
    s1_name_chars = set(get_char_ngrams(s1_name, 3))
    s23_name_chars = set(get_char_ngrams(s23_name, 3))
    s1_addr_chars = set(get_char_ngrams(s1_addr, 3))
    s23_addr_chars = set(get_char_ngrams(s23_addr, 3))
    
    features['name_jaccard_char3'] = jaccard_similarity(s1_name_chars, s23_name_chars)
    features['addr_jaccard_char3'] = jaccard_similarity(s1_addr_chars, s23_addr_chars)
    
    s1_name_len = len(s1_name)
    s23_name_len = len(s23_name)
    features['name_len_ratio'] = min(s1_name_len, s23_name_len) / max(s1_name_len, s23_name_len) if max(s1_name_len, s23_name_len) > 0 else 1.0
    
    s1_addr_len = len(s1_addr)
    s23_addr_len = len(s23_addr)
    features['addr_len_ratio'] = min(s1_addr_len, s23_addr_len) / max(s1_addr_len, s23_addr_len) if max(s1_addr_len, s23_addr_len) > 0 else 1.0
    
    common_suffixes = ['ltd', 'limited', 'pvt', 'private', 'corp', 'corporation', 'inc', 'incorporated', 'llc', 'llp', 'co', 'company']
    s1_has_suffix = any(s1_name.endswith(suf) or f' {suf}' in s1_name for suf in common_suffixes)
    s23_has_suffix = any(s23_name.endswith(suf) or f' {suf}' in s23_name for suf in common_suffixes)
    features['both_have_suffix'] = 1.0 if s1_has_suffix and s23_has_suffix else 0.0
    
    s1_digits = set(re.findall(r'\d+', s1_addr))
    s23_digits = set(re.findall(r'\d+', s23_addr))
    features['addr_digit_overlap'] = jaccard_similarity(s1_digits, s23_digits)
    
    features['name_prefix_match'] = 1.0 if s1_name[:3] == s23_name[:3] and len(s1_name) >= 3 else 0.0
    features['addr_prefix_match'] = 1.0 if s1_addr[:3] == s23_addr[:3] and len(s1_addr) >= 3 else 0.0
    
    return features


def build_feature_matrix(s1_df: pd.DataFrame, s23_df: pd.DataFrame, 
                          candidates: Dict[str, List[str]]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    s1_dict = s1_df.set_index('entity_id').to_dict('index')
    s23_dict = s23_df.set_index('entity_id').to_dict('index')
    
    rows = []
    pairs = []
    
    for s1_id, cand_ids in candidates.items():
        if s1_id not in s1_dict:
            continue
        s1_row = pd.Series(s1_dict[s1_id])
        
        for cand_id in cand_ids:
            if cand_id not in s23_dict:
                continue
            s23_row = pd.Series(s23_dict[cand_id])
            
            feats = extract_features_fast(s1_row, s23_row)
            feats['source1_entity_id'] = s1_id
            feats['candidate_entity_id'] = cand_id
            rows.append(feats)
            pairs.append((s1_id, cand_id))
    
    if not rows:
        return pd.DataFrame(), []
    
    feature_df = pd.DataFrame(rows)
    return feature_df, pairs


def prepare_training_data(s1_train: pd.DataFrame, s2_train: pd.DataFrame, 
                           s3_train: pd.DataFrame, ground_truth: pd.DataFrame,
                           candidates: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.Series]:
    print("Preparing training data...")
    
    s23_train = pd.concat([s2_train, s3_train], ignore_index=True)
    
    feature_df, pairs = build_feature_matrix(s1_train, s23_train, candidates)
    
    if feature_df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    
    gt_dict = {}
    for _, row in ground_truth.iterrows():
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids']
        if pd.isna(matches) or matches == '':
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(matches.split(','))
    
    labels = []
    for s1_id, cand_id in pairs:
        true_matches = gt_dict.get(s1_id, set())
        labels.append(1 if cand_id in true_matches else 0)
    
    feature_cols = [c for c in feature_df.columns if c not in ['source1_entity_id', 'candidate_entity_id']]
    X = feature_df[feature_cols]
    y = pd.Series(labels, name='label')
    
    print(f"Training data: {len(X)} samples")
    print(f"Positive rate: {y.mean():.4f}")
    
    return X, y