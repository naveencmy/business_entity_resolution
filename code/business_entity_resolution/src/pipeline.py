"""
End-to-End Business Entity Resolution Pipeline.
Coordinates:
1. Streaming Ingestion
2. Open-Set Country Partitioning (US, India, France)
3. Multi-Tier Dual-Channel Candidate Generation (Blocking)
4. Pairwise Feature Extraction
5. Precision-Calibrated XGBoost Inference
6. Strict Submission Output Formatting (matching_results.tsv, candidate_pairs.tsv)
"""
import sys
import os
import csv
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import config
from normalizer import clean_business_name, clean_address
from blocking import CountryCandidateIndex
from features import extract_pairwise_features, FEATURE_NAMES
from model import EntityMatcher, compute_macro_f05

def train_model(sample_entities: int = 2500) -> EntityMatcher:
    """
    Train precision-heavy matcher using real ground truth positive pairs
    and hard negative pairs mined via candidate blocking.
    """
    print(f"=== Starting Model Training (Sample Size: {sample_entities}) ===")
    start_time = time.time()
    
    # 1. Load Ground Truth sample
    gt: Dict[str, Set[str]] = {}
    with open(config.TRAIN_GT, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                gt[parts[0]] = set(parts[1].split(","))
                if len(gt) >= sample_entities:
                    break

    target_s1_ids = set(gt.keys())
    target_match_ids = set()
    for m in gt.values():
        target_match_ids.update(m)

    print(f"Loaded {len(gt)} S1 ground truth entities ({len(target_match_ids)} true match records).")

    # 2. Ingest S1 records
    s1_records: Dict[str, Dict[str, Any]] = {}
    with open(config.TRAIN_S1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if not row or len(row) < 4:
                continue
            s1_id = row[0]
            if s1_id in target_s1_ids:
                c = row[3].strip()
                name_full, name_stem = clean_business_name(row[1])
                addr_info = clean_address(row[2], c)
                s1_records[s1_id] = {
                    "entity_id": s1_id,
                    "business_name": row[1],
                    "business_address": row[2],
                    "country": c,
                    "name_full": name_full,
                    "name_stem": name_stem,
                    "postal_code": addr_info["postal_code"],
                    "locality": addr_info["locality"],
                    "clean_addr": addr_info["clean_tokens"],
                }

    # 3. Build candidate index from S2 and S3 (target matches + distractors)
    countries = list({r["country"] for r in s1_records.values()})
    indices = {c: CountryCandidateIndex(c, max_candidates=15) for c in countries}
    
    distractor_cap = 40000
    for path in [config.TRAIN_S2, config.TRAIN_S3]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            distractors = 0
            for row in reader:
                if not row or len(row) < 4:
                    continue
                tid = row[0]
                c = row[3].strip()
                if c not in indices:
                    continue
                is_match = (tid in target_match_ids)
                if is_match or distractors < distractor_cap:
                    if not is_match:
                        distractors += 1
                    name_full, name_stem = clean_business_name(row[1])
                    addr_info = clean_address(row[2], c)
                    indices[c].add_target({
                        "entity_id": tid,
                        "business_name": row[1],
                        "business_address": row[2],
                        "country": c,
                        "name_full": name_full,
                        "name_stem": name_stem,
                        "postal_code": addr_info["postal_code"],
                        "locality": addr_info["locality"],
                        "clean_addr": addr_info["clean_tokens"],
                    })

    for c, idx in indices.items():
        idx.finalize_index()

    # 4. Generate pairs and extract features
    X_list = []
    y_list = []
    pairs_list = []

    print("Generating candidate pairs and extracting pairwise features...")
    for s1_id, s1_rec in s1_records.items():
        c = s1_rec["country"]
        idx = indices[c]
        true_set = gt.get(s1_id, set())

        candidates = idx.query_candidates(s1_rec)
        for true_id in true_set:
            if true_id in idx.targets and true_id not in candidates:
                candidates.append(true_id)

        for cand_id in candidates:
            cand_rec = idx.targets.get(cand_id)
            if not cand_rec:
                continue
            is_pos = 1 if cand_id in true_set else 0
            feats = extract_pairwise_features(s1_rec, cand_rec)
            X_list.append(feats)
            y_list.append(is_pos)
            pairs_list.append((s1_id, cand_id))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Training dataset shape: {X.shape} (Positives: {np.sum(y)}, Negatives: {len(y) - np.sum(y)})")

    # 5. Fit model
    matcher = EntityMatcher(model_type="xgb")
    matcher.fit(X, y)

    # 6. Optimize F_0.5 probability threshold
    probs = matcher.predict_proba(X)
    matcher.optimize_threshold(pairs_list, probs, gt)
    
    # Save model artifact
    model_save_path = Path(__file__).resolve().parent / "matcher_model.pkl"
    matcher.save(str(model_save_path))
    print(f"Model saved to {model_save_path} in {time.time() - start_time:.1f}s")
    return matcher


def run_test_inference(matcher: Optional[EntityMatcher] = None):
    """
    Run full test inference and generate matching_results.tsv and candidate_pairs.tsv.
    Processes country by country to keep memory bounded and maintain high throughput.
    """
    print("\n=== Starting Test Inference & Submission Generation ===")
    total_start = time.time()
    
    if matcher is None:
        matcher = EntityMatcher(model_type="xgb")
        model_save_path = Path(__file__).resolve().parent / "matcher_model.pkl"
        if os.path.exists(model_save_path):
            print(f"Loading existing model from {model_save_path}...")
            matcher.load(str(model_save_path))
        else:
            print("No saved model found, training fresh model...")
            matcher = train_model(sample_entities=2000)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Ingest test S1 entities in original file order
    print("Reading test_source1.tsv order and partitioning by country...")
    s1_all_ids: List[str] = []
    s1_by_country: Dict[str, List[Dict[str, Any]]] = {}
    
    with open(config.TEST_S1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row or len(row) < 4:
                continue
            s1_id = row[0].strip()
            c = row[3].strip()
            s1_all_ids.append(s1_id)
            if c not in s1_by_country:
                s1_by_country[c] = []
            
            name_full, name_stem = clean_business_name(row[1])
            addr_info = clean_address(row[2], c)
            s1_by_country[c].append({
                "entity_id": s1_id,
                "business_name": row[1],
                "business_address": row[2],
                "country": c,
                "name_full": name_full,
                "name_stem": name_stem,
                "postal_code": addr_info["postal_code"],
                "locality": addr_info["locality"],
                "clean_addr": addr_info["clean_tokens"],
            })

    print(f"Total S1 entities: {len(s1_all_ids)} across: {list(s1_by_country.keys())}")

    # Result maps: s1_id -> candidate string, s1_id -> matched string
    results_candidates: Dict[str, str] = {}
    results_matches: Dict[str, str] = {}

    # 2. Process each country partition
    for country, s1_list in s1_by_country.items():
        c_start = time.time()
        print(f"\n==========================================")
        print(f"Country Partition: {country} ({len(s1_list)} S1 entities)")
        idx = CountryCandidateIndex(country, max_candidates=config.MAX_CANDIDATES_PER_ENTITY)

        # Index test targets from test_source2 and test_source3
        print(f"Indexing targets for {country} from test_source2.tsv and test_source3.tsv...")
        for path in [config.TEST_S2, config.TEST_S3]:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader)
                for row in reader:
                    if not row or len(row) < 4:
                        continue
                    if row[3].strip() != country:
                        continue
                    tid = row[0].strip()
                    name_full, name_stem = clean_business_name(row[1])
                    addr_info = clean_address(row[2], country)
                    idx.add_target({
                        "entity_id": tid,
                        "business_name": row[1],
                        "business_address": row[2],
                        "country": country,
                        "name_full": name_full,
                        "name_stem": name_stem,
                        "postal_code": addr_info["postal_code"],
                        "locality": addr_info["locality"],
                        "clean_addr": addr_info["clean_tokens"],
                    })

        print(f"Indexed {len(idx.targets)} target records for {country}. Finalizing inverted index...")
        idx.finalize_index()

        # Query candidates and predict matches in batches
        print(f"Querying candidates and predicting matches for {len(s1_list)} {country} entities...")
        matched_count = 0
        singleton_count = 0
        
        for i, s1_rec in enumerate(s1_list):
            s1_id = s1_rec["entity_id"]
            cands = idx.query_candidates(s1_rec)
            
            if not cands:
                results_candidates[s1_id] = ""
                results_matches[s1_id] = ""
                singleton_count += 1
                continue

            results_candidates[s1_id] = ",".join(cands)

            cand_recs = [idx.targets[cid] for cid in cands if cid in idx.targets]
            if not cand_recs:
                results_matches[s1_id] = ""
                singleton_count += 1
                continue

            feats_matrix = np.array([
                extract_pairwise_features(s1_rec, cr) for cr in cand_recs
            ], dtype=np.float32)

            probs = matcher.predict_proba(feats_matrix)
            cand_ids = [cr["entity_id"] for cr in cand_recs]
            matched_ids = matcher.predict_entity_matches(
                cand_ids, probs,
                anchor_threshold=max(0.72, matcher.optimal_threshold),
                expansion_threshold=max(0.60, matcher.optimal_threshold - 0.08)
            )

            if matched_ids:
                results_matches[s1_id] = ",".join(matched_ids)
                matched_count += 1
            else:
                results_matches[s1_id] = ""
                singleton_count += 1

            if (i + 1) % 50000 == 0 or (i + 1) == len(s1_list):
                print(f"  Processed {i + 1}/{len(s1_list)} ({((i + 1)/len(s1_list))*100:.1f}%) | Matches: {matched_count} | Singletons: {singleton_count}")

        print(f"Completed {country} in {time.time() - c_start:.1f}s.")
        # Free country index memory
        del idx

    # 3. Write final submission files strictly in original S1 order
    print("\nWriting output/candidate_pairs.tsv...")
    with open(config.OUTPUT_CANDIDATES, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_all_ids:
            f.write(f"{s1_id}\t{results_candidates.get(s1_id, '')}\n")

    print("Writing output/matching_results.tsv...")
    with open(config.OUTPUT_MATCHING, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_all_ids:
            f.write(f"{s1_id}\t{results_matches.get(s1_id, '')}\n")

    print(f"\nAll outputs successfully generated in {time.time() - total_start:.1f}s!")
    print(f"Matching file: {config.OUTPUT_MATCHING}")
    print(f"Candidate file: {config.OUTPUT_CANDIDATES}")


if __name__ == "__main__":
    run_test_inference()
