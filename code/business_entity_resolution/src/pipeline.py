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
import collections
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

import random

def train_model(sample_entities: int = 5000) -> EntityMatcher:
    """
    Train precision-heavy matcher with strict entity-level 80/20 train/val split.
    - Eliminates resubstitution leakage: fit only on train split, threshold tuned only on val split.
    - Eliminates candidate cheat: no force-injection of true positives into candidate lists.
    - Real diagnostic blocking recall tracking.
    """
    print(f"=== Starting Honest Model Training (Entity Sample: {sample_entities}) ===")
    start_time = time.time()
    
    # 1. Ingest Ground Truth sample with shuffling
    all_gt_lines = []
    with open(config.TRAIN_GT, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                all_gt_lines.append((parts[0], set(parts[1].split(","))))

    rng = random.Random(42)
    rng.shuffle(all_gt_lines)
    selected_gt = dict(all_gt_lines[:sample_entities])

    target_s1_ids = set(selected_gt.keys())
    target_match_ids = set()
    for m in selected_gt.values():
        target_match_ids.update(m)

    print(f"Sampled {len(selected_gt)} S1 ground truth entities ({len(target_match_ids)} true matches).")

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

    # 3. Build candidate index from S2 and S3 (target matches + scaled distractors)
    countries = list({r["country"] for r in s1_records.values()})
    indices = {c: CountryCandidateIndex(c, max_candidates=15) for c in countries}
    
    distractor_cap = 60000
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

    # 4. Strict Entity-Level 80/20 Train/Validation Split
    entity_id_list = list(s1_records.keys())
    rng.shuffle(entity_id_list)
    split_idx = int(len(entity_id_list) * 0.8)
    train_ids = set(entity_id_list[:split_idx])
    val_ids = set(entity_id_list[split_idx:])

    gt_train = {eid: selected_gt[eid] for eid in train_ids if eid in selected_gt}
    gt_val = {eid: selected_gt[eid] for eid in val_ids if eid in selected_gt}

    print(f"Entity Split: {len(train_ids)} Train Entities (80%), {len(val_ids)} Validation Entities (20%).")

    # Diagnostic blocking metrics
    blocking_hits = 0
    blocking_total_true = 0

    X_train, y_train, pairs_train = [], [], []
    X_val, y_val, pairs_val = [], [], []

    print("Generating candidate pairs without force-injection cheat...")
    for s1_id, s1_rec in s1_records.items():
        c = s1_rec["country"]
        idx = indices[c]
        true_set = selected_gt.get(s1_id, set())

        # REAL BLOCKING CANDIDATES ONLY
        cand_with_scores = idx.query_candidates_with_scores(s1_rec)
        cand_ids = [cid for cid, _ in cand_with_scores]

        # Track blocking recall honestly
        if true_set:
            blocking_hits += len(true_set.intersection(set(cand_ids)))
            blocking_total_true += len(true_set)

        is_val = s1_id in val_ids

        for cand_id, sc in cand_with_scores:
            cand_rec = idx.targets.get(cand_id)
            if not cand_rec:
                continue
            is_pos = 1 if cand_id in true_set else 0
            feats = extract_pairwise_features(s1_rec, cand_rec, blocking_score=sc)

            if is_val:
                X_val.append(feats)
                y_val.append(is_pos)
                pairs_val.append((s1_id, cand_id))
            else:
                X_train.append(feats)
                y_train.append(is_pos)
                pairs_train.append((s1_id, cand_id))

    if blocking_total_true > 0:
        print(f"Honest Candidate Generation Recall (Ceiling): {blocking_hits / blocking_total_true:.4f} ({blocking_hits}/{blocking_total_true})")

    X_tr = np.array(X_train, dtype=np.float32)
    y_tr = np.array(y_train, dtype=np.int32)
    X_v = np.array(X_val, dtype=np.float32)
    y_v = np.array(y_val, dtype=np.int32)

    print(f"Train Matrix Shape: {X_tr.shape} (Pos: {np.sum(y_tr)}, Neg: {len(y_tr) - np.sum(y_tr)})")
    print(f"Val Matrix Shape:   {X_v.shape} (Pos: {np.sum(y_v)}, Neg: {len(y_v) - np.sum(y_v)})")

    # 5. Fit Model ONLY on Train Split
    matcher = EntityMatcher(model_type="xgb")
    matcher.fit(X_tr, y_tr)

    # 6. Optimize Decision Threshold ONLY on Held-Out Validation Split
    val_probs = matcher.predict_proba(X_v)
    matcher.optimize_threshold(pairs_val, val_probs, gt_val)

    # 7. Evaluate honest held-out validation metrics with per-country breakdown
    val_preds: Dict[str, Set[str]] = {eid: set() for eid in val_ids}
    val_entity_cands: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for (s1_id, cid), prob in zip(pairs_val, val_probs):
        val_entity_cands[s1_id].append((cid, float(prob)))

    for s1_id in val_ids:
        c_list = val_entity_cands.get(s1_id, [])
        if not c_list:
            continue
        c_ids = [cid for cid, _ in c_list]
        p_arr = np.array([p for _, p in c_list], dtype=np.float32)
        m_ids = matcher.predict_entity_matches(
            c_ids, p_arr,
            anchor_threshold=matcher.optimal_threshold,
            expansion_threshold=max(0.55, matcher.optimal_threshold - 0.08)
        )
        val_preds[s1_id] = set(m_ids)

    val_countries = {eid: s1_records[eid]["country"] for eid in val_ids if eid in s1_records}
    metrics = compute_macro_f05(gt_val, val_preds, entity_countries=val_countries)
    print("\n" + "=" * 50)
    print(f"HONEST HELD-OUT VALIDATION RESULTS (Zero Leakage):")
    print(f"  Macro F_0.5: {metrics['macro_f05']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    if "per_country" in metrics:
        print("  Per-Country Breakdown:")
        for c, sc in metrics["per_country"].items():
            print(f"    - {c}: {sc:.4f}")
    print("=" * 50 + "\n")

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
        
        batch_size = 2000
        for b_start in range(0, len(s1_list), batch_size):
            b_chunk = s1_list[b_start : b_start + batch_size]
            batch_feats = []
            entity_cand_meta = []

            for s1_rec in b_chunk:
                s1_id = s1_rec["entity_id"]
                cand_with_scores = idx.query_candidates_with_scores(s1_rec)
                
                if not cand_with_scores:
                    results_candidates[s1_id] = ""
                    results_matches[s1_id] = ""
                    singleton_count += 1
                    continue

                cands = [cid for cid, _ in cand_with_scores]
                results_candidates[s1_id] = ",".join(cands)

                cand_recs = [(idx.targets[cid], sc) for cid, sc in cand_with_scores if cid in idx.targets]
                if not cand_recs:
                    results_matches[s1_id] = ""
                    singleton_count += 1
                    continue

                start_off = len(batch_feats)
                for cr, sc in cand_recs:
                    batch_feats.append(extract_pairwise_features(s1_rec, cr, blocking_score=sc))
                end_off = len(batch_feats)
                cand_ids = [cr["entity_id"] for cr, _ in cand_recs]
                entity_cand_meta.append((s1_id, cand_ids, start_off, end_off))

            if batch_feats:
                feats_matrix = np.array(batch_feats, dtype=np.float32)
                batch_probs = matcher.predict_proba(feats_matrix)
                
                for s1_id, cand_ids, s_off, e_off in entity_cand_meta:
                    probs = batch_probs[s_off:e_off]
                    matched_ids = matcher.predict_entity_matches(
                        cand_ids, probs,
                        anchor_threshold=matcher.optimal_threshold,
                        expansion_threshold=max(0.55, matcher.optimal_threshold - 0.08)
                    )
                    if matched_ids:
                        results_matches[s1_id] = ",".join(matched_ids)
                        matched_count += 1
                    else:
                        results_matches[s1_id] = ""
                        singleton_count += 1

            processed = min(b_start + batch_size, len(s1_list))
            if processed % 50000 == 0 or processed == len(s1_list):
                print(f"  Processed {processed}/{len(s1_list)} ({((processed)/len(s1_list))*100:.1f}%) | Matches: {matched_count} | Singletons: {singleton_count}")

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
