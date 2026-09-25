import pandas as pd
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz, process
from tqdm import tqdm
import re
from typing import Dict, List, Set, Tuple
import gc


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return text.split()


def get_name_tokens(name: str) -> List[str]:
    tokens = tokenize(name)
    stopwords = {'ltd', 'limited', 'pvt', 'private', 'corp', 'corporation', 'inc', 
                 'incorporated', 'llc', 'llp', 'co', 'company', 'and', 'the', 'of', 
                 'for', 'in', 'on', 'at', 'to', 'by', 'with', 'from', 'a', 'an'}
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def get_address_tokens(address: str) -> List[str]:
    tokens = tokenize(address)
    stopwords = {'near', 'opposite', 'behind', 'front', 'beside', 'next', 'to', 
                 'the', 'and', 'of', 'for', 'in', 'on', 'at', 'road', 'street', 
                 'avenue', 'lane', 'drive', 'circle', 'court', 'place', 'blvd', 
                 'boulevard', 'highway', 'expressway', 'pkwy', 'parkway'}
    return [t for t in tokens if t not in stopwords and (len(t) > 2 or t.isdigit())]


def build_blocking_keys_chunked(df: pd.DataFrame, chunk_size: int = 100000) -> pd.DataFrame:
    results = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        
        name_norm = chunk['business_name'].astype(str).str.lower().str.strip()
        name_norm = name_norm.str.replace(r'[^\w\s]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True)
        
        addr_norm = chunk['business_address'].astype(str).str.lower().str.strip()
        addr_norm = addr_norm.str.replace(r'[^\w\s]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True)
        
        country = chunk['country'].astype(str).str.strip().str.lower()
        
        name_first_token = name_norm.str.split().str[0].str[:3].fillna('')
        name_second_token = name_norm.str.split().str[1].str[:3].fillna('')
        addr_first_token = addr_norm.str.split().str[0].str[:3].fillna('')
        
        block_key = country + '_' + name_first_token
        block_key = block_key + '_' + name_second_token.replace('', '_')
        block_key = block_key + '_' + addr_first_token.replace('', '_')
        
        chunk['_block_key'] = block_key
        chunk['_country'] = country
        chunk['_name_prefix'] = country + '_' + name_first_token
        
        results.append(chunk[['entity_id', '_block_key', '_country', '_name_prefix']])
        del chunk
        gc.collect()
    
    return pd.concat(results, ignore_index=True)


def generate_candidates_blocking(source1: pd.DataFrame, source2: pd.DataFrame, 
                                   source3: pd.DataFrame) -> Dict[str, List[str]]:
    print("Building blocking keys for Source 2...")
    s2_keys = build_blocking_keys_chunked(source2)
    
    print("Building blocking keys for Source 3...")
    s3_keys = build_blocking_keys_chunked(source3)
    
    all_s23 = pd.concat([s2_keys[['entity_id', '_block_key']], 
                         s3_keys[['entity_id', '_block_key']]], ignore_index=True)
    
    del s2_keys, s3_keys
    gc.collect()
    
    print("Building block-to-ids mapping...")
    block_to_ids = all_s23.groupby('_block_key')['entity_id'].apply(list).to_dict()
    
    print("Building prefix index...")
    prefix_to_blocks = defaultdict(list)
    for block_key in block_to_ids.keys():
        parts = block_key.split('_')
        if len(parts) >= 2:
            prefix = parts[0] + '_' + parts[1]
            prefix_to_blocks[prefix].append(block_key)
    
    del all_s23
    gc.collect()
    
    print("Building blocking keys for Source 1...")
    s1_keys = build_blocking_keys_chunked(source1)
    
    print("Generating candidates...")
    candidates = {}
    
    s1_ids = s1_keys['entity_id'].values
    s1_block_keys = s1_keys['_block_key'].values
    s1_name_prefixes = s1_keys['_name_prefix'].values
    
    for i in range(len(s1_ids)):
        s1_id = s1_ids[i]
        block_key = s1_block_keys[i]
        name_prefix = s1_name_prefixes[i]
        
        candidate_ids = set()
        
        if block_key in block_to_ids:
            candidate_ids.update(block_to_ids[block_key])
        
        if name_prefix in prefix_to_blocks:
            for bk in prefix_to_blocks[name_prefix]:
                candidate_ids.update(block_to_ids[bk])
        
        if len(candidate_ids) > 500:
            candidate_ids = list(candidate_ids)[:500]
        
        candidates[s1_id] = list(candidate_ids)
        
        if i % 100000 == 0 and i > 0:
            print(f"  Processed {i}/{len(s1_ids)}")
    
    return candidates


def generate_candidates_fuzzy(source1: pd.DataFrame, source2: pd.DataFrame, 
                               source3: pd.DataFrame, threshold: int = 70) -> Dict[str, List[str]]:
    print("Building name indexes for fuzzy matching...")
    
    s2_names = dict(zip(source2['entity_id'], source2['business_name'].apply(normalize_text)))
    s3_names = dict(zip(source3['entity_id'], source3['business_name'].apply(normalize_text)))
    s2_countries = dict(zip(source2['entity_id'], source2['country']))
    s3_countries = dict(zip(source3['entity_id'], source3['country']))
    
    all_s23_names = {**s2_names, **s3_names}
    all_s23_countries = {**s2_countries, **s3_countries}
    
    print("Running fuzzy matching...")
    candidates = {}
    
    for _, row in tqdm(source1.iterrows(), total=len(source1), desc="Fuzzy matching"):
        s1_id = row['entity_id']
        country = str(row['country']).strip()
        name = normalize_text(row['business_name'])
        
        if not name:
            candidates[s1_id] = []
            continue
        
        same_country_ids = [eid for eid, c in all_s23_countries.items() if c == country]
        
        if not same_country_ids:
            candidates[s1_id] = []
            continue
        
        name_choices = {eid: all_s23_names[eid] for eid in same_country_ids if eid in all_s23_names}
        
        if not name_choices:
            candidates[s1_id] = []
            continue
        
        matches = process.extract(name, name_choices, scorer=fuzz.token_sort_ratio, 
                                   limit=100, score_cutoff=threshold)
        
        candidate_ids = [match[2] for match in matches]
        candidates[s1_id] = candidate_ids
    
    return candidates


def merge_candidates(candidates1: Dict[str, List[str]], candidates2: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged = {}
    all_keys = set(candidates1.keys()) | set(candidates2.keys())
    for key in all_keys:
        merged[key] = list(set(candidates1.get(key, []) + candidates2.get(key, [])))
    return merged