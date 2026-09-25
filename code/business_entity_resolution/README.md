# Business Entity Resolution Pipeline

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
cd code/business_entity_resolution
python src/pipeline.py
```

## Pipeline Stages

1. **Blocking/Candidate Generation** - Country + name prefix + address prefix blocking
2. **Feature Engineering** - 14 string similarity features (Jaccard, cosine, prefix match, etc.)
3. **Model Training** - LightGBM binary classifier with F₀.₅-optimized threshold
4. **Inference** - Apply model to test candidates
5. **Output Generation** - `matching_results.tsv` and `candidate_pairs.tsv`

## Output Files

- `output/matching_results.tsv` - Final matches (scored on leaderboard)
- `output/candidate_pairs.tsv` - Candidate pairs from blocking stage

## Validation

```bash
python ../../student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```

## Data Caching

First run caches data to `student_resource/cache/` (~2 min load time). Subsequent runs use cached data.

## Model

- LightGBM binary classifier
- Trained on candidate pairs from training data
- Optimized for F₀.₅ score (precision-weighted)
- MIT/Apache 2.0 compatible
- Under 8B parameter limit