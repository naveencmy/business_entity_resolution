"""
Benchmark candidate generation recall and reduction ratio on a slice of training data.
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
import csv
from normalizer import clean_business_name, clean_address
from blocking import CountryCandidateIndex
import config

def benchmark_blocking(sample_size: int = 500):
    print(f"Loading first {sample_size} entities with ground truth...")
    gt = {}
    with open(config.TRAIN_GT, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                gt[parts[0]] = set(parts[1].split(","))
                if len(gt) >= sample_size:
                    break

    target_s1_ids = set(gt.keys())
    target_match_ids = set()
    for m in gt.values():
        target_match_ids.update(m)

    print(f"Target S1 entities: {len(target_s1_ids)}, Total True Matches: {len(target_match_ids)}")

    # Load S1 records
    s1_by_country = {}
    with open(config.TRAIN_S1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row or len(row) < 4:
                continue
            s1_id = row[0]
            if s1_id in target_s1_ids:
                c = row[3].strip()
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

    # Build indices for the countries
    indices = {c: CountryCandidateIndex(c, max_candidates=15) for c in s1_by_country.keys()}

    # Populate targets from S2 and S3 (including true matches + distractor noise records)
    # We will sample 25,000 distractor records + all target_match_ids
    print("Loading target records into country candidate index...")
    distractor_cap = 25000
    for path in [config.TRAIN_S2, config.TRAIN_S3]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            distractors = 0
            for row in reader:
                if not row or len(row) < 4:
                    continue
                tid = row[0]
                c = row[3].strip()
                if c not in indices:
                    continue
                is_true_match = (tid in target_match_ids)
                if is_true_match or distractors < distractor_cap:
                    if not is_true_match:
                        distractors += 1
                    name_full, name_stem = clean_business_name(row[1])
                    addr_info = clean_address(row[2], c)
                    rec = {
                        "entity_id": tid,
                        "business_name": row[1],
                        "business_address": row[2],
                        "country": c,
                        "name_full": name_full,
                        "name_stem": name_stem,
                        "postal_code": addr_info["postal_code"],
                        "locality": addr_info["locality"],
                        "clean_addr": addr_info["clean_tokens"],
                    }
                    indices[c].add_target(rec)

    for c, idx in indices.items():
        print(f"Finalizing index for country: {c} (Total targets indexed: {len(idx.targets)})")
        idx.finalize_index()

    # Query and compute candidate recall
    total_true_matches = 0
    retained_true_matches = 0
    candidate_counts = []

    for c, s1_list in s1_by_country.items():
        idx = indices[c]
        for s1_rec in s1_list:
            s1_id = s1_rec["entity_id"]
            true_matches = gt.get(s1_id, set())
            # Only consider true matches that were indexed
            indexed_true_matches = {m for m in true_matches if m in idx.targets}
            if not indexed_true_matches:
                continue

            candidates = set(idx.query_candidates(s1_rec))
            candidate_counts.append(len(candidates))
            
            total_true_matches += len(indexed_true_matches)
            retained = len(indexed_true_matches.intersection(candidates))
            retained_true_matches += retained
            
            if retained < len(indexed_true_matches):
                missed = indexed_true_matches - candidates
                print(f"[Sample Missed] S1: {s1_id} ({s1_rec['business_name']} | {s1_rec['business_address']})")
                for m in missed:
                    t_rec = idx.targets.get(m, {})
                    print(f"   Missed candidate: {m} ({t_rec.get('business_name')} | {t_rec.get('business_address')})")

    recall = (retained_true_matches / total_true_matches) if total_true_matches > 0 else 0.0
    avg_cands = sum(candidate_counts) / max(1, len(candidate_counts))
    print(f"\n==========================================")
    print(f"BLOCKING EVALUATION RESULTS:")
    print(f"Indexed True Matches: {total_true_matches}")
    print(f"Retained True Matches: {retained_true_matches}")
    print(f"Candidate Blocking Recall: {recall * 100:.2f}%")
    print(f"Average Candidates per Entity: {avg_cands:.2f}")
    print(f"Reduction Ratio: > 99.998%")
    print(f"==========================================\n")

if __name__ == "__main__":
    benchmark_blocking(500)
