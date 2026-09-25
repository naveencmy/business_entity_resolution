# Business Entity Resolution Pipeline (ML Challenge 2026)

## Overview
This package implements an end-to-end, high-performance Entity Resolution system designed to match noisy, heterogeneous business records from three independent sources ($\mathcal{S}_1, \mathcal{S}_2, \mathcal{S}_3$) with reference deduplicated entities in $\mathcal{S}_1$.

The solution is specifically engineered to optimize the **Macro $F_{0.5}$** metric (targeting $\ge 0.998$), which heavily penalizes false merges (false positives) on singletons and matches.

---

## Directory Structure
```
code/business_entity_resolution/
├── requirements.txt         # Pinned Python dependencies
├── README.md                # End-to-end execution guide
└── src/
    ├── __init__.py
    ├── config.py            # Paths, parameters, thresholds
    ├── normalizer.py        # Multilingual Indic transliteration & clean-up
    ├── blocking.py          # Scalable dual-channel inverted index blocking
    ├── features.py          # Fast pairwise similarity vectorizer
    ├── model.py             # Precision-calibrated XGBoost & threshold optimizer
    ├── ingestion.py         # Streaming TSV parser
    └── pipeline.py          # Main entry point (train + test inference)
```

---

## Requirements & Environment Setup
Python 3.10+ or 3.11+ is recommended. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Reproduction & Pipeline Execution

### Option A: Running on Kaggle / Cloud GPU (Recommended for High Speed)
1. Upload this package or `kaggle_pipeline.py` / `kaggle_business_entity_resolution.ipynb` to Kaggle.
2. Attach the competition dataset as input data.
3. Turn on GPU accelerator (T4 or P100) in Kaggle Notebook settings.
4. Execute:
```bash
python kaggle_pipeline.py
```
The script will auto-detect CUDA GPU, train with XGBoost GPU `hist` engine, and stream predictions across all 1.73M test entities in minutes.

---

### Option B: Local Pipeline Execution
```bash
cd code/business_entity_resolution/src
python pipeline.py
```

This script:
1. Streams and normalizes training entities and ground truth pairs.
2. Constructs the country-partitioned dual-channel inverted index.
3. Mines hard negative candidates and extracts composite string/address similarity features.
4. Trains the precision-calibrated XGBoost classifier and optimizes the decision threshold on validation data.
5. Ingests `test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv` across all open-set country partitions (`US`, `India`, `France`).
6. Generates candidate pairs and predicts matches.
7. Strictly formats and exports:
   - `output/candidate_pairs.tsv`
   - `output/matching_results.tsv`

---

## Submission Validation
To verify strict formatting compliance and run the competition validator:
```bash
python Deputy_pipe/Datasets/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir Deputy_pipe/Datasets/student_resource/dataset/test
```
A valid run exits with code `0` and outputs `PASS`.
