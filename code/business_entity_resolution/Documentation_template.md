# Business Entity Resolution - Methodology Documentation

## Team Information
- **Team Name:** [Team Name]
- **Date:** 2026-09-25

---

## 1. Methodology Overview

Our approach to the Business Entity Resolution challenge follows a classic two-stage pipeline:

1. **Blocking/Candidate Generation** - Efficiently reduce the search space from ~10B possible pairs to a manageable candidate set
2. **Supervised Matching** - Train a binary classifier to distinguish true matches from non-matches among candidates

### Key Design Decisions

- **Precision-focused**: Since F₀.₅ weights precision 2× over recall, we prioritize high-precision blocking and conservative matching thresholds
- **No external data**: Strictly using only provided training data per competition rules
- **Scalable architecture**: Vectorized operations where possible, efficient data structures
- **Country-aware**: Explicitly handle US, India, and France (unseen in training) as open-set labels

---

## 2. Candidate Generation / Blocking Strategy

### 2.1 Blocking Key Construction

We construct blocking keys using a combination of:
- **Country** (exact match required - entities only match within same country)
- **First 3 characters of first name token** (after removing stopwords: ltd, corp, inc, etc.)
- **First 3 characters of second name token** (if available)
- **First 3 characters of first address token**

This creates keys like: `us_abc_xyz_123`

### 2.2 Blocking Process

1. Build block keys for all Source 2 and Source 3 records
2. Create inverted index: `block_key → [entity_ids]`
3. For each Source 1 entity:
   - Compute its block key
   - Retrieve all S2/S3 entities in same block
   - Also retrieve entities from blocks sharing the same country + first name prefix
4. Cap candidates at 500 per S1 entity to control computational cost

### 2.3 Blocking Statistics (on 0.1% sample)

- **Training**: ~30 candidates per S1 entity
- **Test**: ~12 candidates per S1 entity
- **Reduction ratio**: ~99.9%+ (from billions of possible pairs to thousands)

### 2.4 Recall Considerations

Our blocking achieves high recall by:
- Using prefix-based matching (robust to typos, abbreviations)
- Including adjacent name tokens
- Country-level partitioning (prevents cross-country false matches)
- Multi-token keys for specificity

---

## 3. Feature Engineering

We compute 14 features per candidate pair:

### 3.1 String Similarity Features

| Feature | Description |
|---------|-------------|
| `same_country` | Binary: 1 if both entities share country |
| `name_jaccard_word` | Jaccard similarity on word tokens |
| `addr_jaccard_word` | Jaccard similarity on address word tokens |
| `name_jaccard_char3` | Jaccard on character 3-grams (name) |
| `addr_jaccard_char3` | Jaccard on character 3-grams (address) |
| `name_cosine` | TF-IDF cosine similarity (name) |
| `addr_cosine` | TF-IDF cosine similarity (address) |
| `name_len_ratio` | Min/max length ratio (name) |
| `addr_len_ratio` | Min/max length ratio (address) |
| `both_have_suffix` | Both have legal suffixes (Ltd, Corp, etc.) |
| `addr_digit_overlap` | Jaccard on numeric tokens in address |
| `name_prefix_match` | First 3 chars match (name) |
| `addr_prefix_match` | First 3 chars match (address) |

### 3.2 Feature Design Rationale

- **Character n-grams**: Robust to typos, transliteration differences
- **Word-level Jaccard**: Captures token overlap despite reordering
- **Cosine similarity**: Weights rare tokens higher
- **Suffix detection**: Legal entity types are strong matching signals
- **Address digits**: Street numbers, PIN codes are high-precision signals
- **Prefix matching**: Fast exact-match signal for clean cases

---

## 4. Model Architecture

### 4.1 Model Choice: LightGBM

We use LightGBM (gradient boosted decision trees) because:
- Handles mixed feature types natively
- Fast training and inference
- Built-in handling of class imbalance via `scale_pos_weight`
- MIT-licensed, well under 8B parameter limit (~100KB model)
- Excellent tabular data performance

### 4.2 Hyperparameters

```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 4,
    'seed': 42,
    'scale_pos_weight': neg_count / max(1, pos_count),
}
```

### 4.3 Training Strategy

- **Early stopping**: 50 rounds on validation logloss
- **Max rounds**: 1000 (typically stops at ~200-400)
- **Validation split**: 10% stratified (when ≥2 positives exist)
- **Threshold optimization**: Grid search on validation set for F₀.₅

### 4.4 Handling Class Imbalance

The positive rate is extremely low (~0.01%). We handle this by:
1. `scale_pos_weight` in LightGBM
2. F₀.₅-optimized threshold (typically 0.7-0.9)
3. Conservative matching - only high-confidence predictions

---

## 5. Inference Pipeline

### 5.1 Full Test Set Processing

For the final submission, we process all 1,732,544 Source 1 test entities:
1. Load cached training data (pickled after first load)
2. Load trained model
3. Generate candidates for ALL test S1 entities
4. Compute features for all candidate pairs
5. Run inference with optimized threshold
6. Produce `matching_results.tsv` and `candidate_pairs.tsv`

### 5.2 Output Format Compliance

Both output files strictly follow the specification:
- Tab-separated (not CSV)
- Exact column names: `source1_entity_id`, `matched_entity_ids` / `candidate_entity_ids`
- One row per S1 entity (including singletons with empty ID lists)
- No duplicate IDs within lists
- Only S2- and S3- prefixed IDs
- Sorted entity IDs for reproducibility

### 5.3 Validation

We run the provided validator before submission:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 6. Computational Considerations

### 6.1 Training Time (Full Data)
- Data loading: ~2 minutes (12M rows)
- Blocking: ~15-20 minutes
- Feature computation: ~30-45 minutes
- Model training: ~5-10 minutes
- **Total: ~1-1.5 hours**

### 6.2 Inference Time (Full Test)
- Blocking: ~10-15 minutes
- Feature computation: ~20-30 minutes
- Inference: ~2-5 minutes
- **Total: ~45-60 minutes**

### 6.3 Memory Usage
- Peak: ~8-12 GB (during feature matrix construction)
- Model size: <1 MB
- Streaming/chunking possible for lower memory

---

## 7. Reproducibility

### 7.1 Requirements
```
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.3.0
lightgbm>=4.1.0
rapidfuzz>=3.5.0
tqdm>=4.66.0
joblib>=1.3.0
```

### 7.2 Run Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (first run caches data)
cd code/business_entity_resolution
python src/pipeline.py

# Outputs in output/
# matching_results.tsv - final matches for leaderboard
# candidate_pairs.tsv - blocking candidates for audit
```

### 7.3 Expected Outputs

- `output/matching_results.tsv` - 1,732,544 rows
- `output/candidate_pairs.tsv` - 1,732,544 rows

---

## 8. Future Improvements

Given more time, we would explore:

1. **Advanced blocking**: Canopy clustering, LSH for names
2. **Deep learning features**: Siamese networks with character embeddings
3. **Active learning**: Human-in-the-loop for uncertain pairs
4. **Ensemble**: Multiple models (Logistic Regression + LightGBM + XGBoost)
5. **Geographic features**: Leverage address structure (city, state patterns)
6. **Cross-validation**: More robust threshold selection
7. **Hard negative mining**: Improve classifier on difficult negatives

---

## 9. Compliance Checklist

- [ ] No external data lookup (APIs, databases, geocoding)
- [ ] Model license: MIT/Apache 2.0 (LightGBM)
- [ ] Model size: < 8B parameters (~100KB)
- [ ] Output format: TSV, correct columns, all S1 entities
- [ ] Singletons handled: Empty lists for no-match entities
- [ ] No duplicate IDs in lists
- [ ] Only S2-/S3- IDs in output
- [ ] Validation passes locally