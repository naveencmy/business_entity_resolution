"""
Inspect actual ground truth matched pairs to analyze name and address variations.
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from normalizer import clean_business_name, clean_address
import config

def inspect():
    # Load first 50 ground truth pairs
    gt = {}
    with open(config.TRAIN_GT, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                s1_id = parts[0]
                matches = parts[1].split(",")
                gt[s1_id] = matches
    
    target_s1 = set(gt.keys())
    target_s2_s3 = set()
    for m in gt.values():
        target_s2_s3.update(m)
        
    s1_records = {}
    with open(config.TRAIN_S1, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in target_s1:
                s1_records[parts[0]] = parts
                if len(s1_records) == len(target_s1):
                    break
                    
    matched_records = {}
    for path in [config.TRAIN_S2, config.TRAIN_S3]:
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts[0] in target_s2_s3:
                    matched_records[parts[0]] = parts
                    
    print(f"Sampled {len(s1_records)} S1 entities with true matches.")
    count = 0
    for s1_id, matches in gt.items():
        if s1_id in s1_records:
            s1_row = s1_records[s1_id]
            print(f"\n==========================================")
            print(f"S1 ID: {s1_id} | Country: {s1_row[3]}")
            print(f"  Name:    {s1_row[1]}")
            print(f"  Address: {s1_row[2]}")
            s1_name_full, s1_name_stem = clean_business_name(s1_row[1])
            s1_addr_info = clean_address(s1_row[2], s1_row[3])
            print(f"  [Cleaned Name Stem]: '{s1_name_stem}' | [Addr Tokens]: '{s1_addr_info['clean_tokens']}'")
            
            for m_id in matches[:3]:
                if m_id in matched_records:
                    m_row = matched_records[m_id]
                    print(f"  --> MATCH {m_id}:")
                    print(f"      Name:    {m_row[1]}")
                    print(f"      Address: {m_row[2]}")
                    m_name_full, m_name_stem = clean_business_name(m_row[1])
                    m_addr_info = clean_address(m_row[2], m_row[3])
                    print(f"      [Cleaned Name Stem]: '{m_name_stem}' | [Addr Tokens]: '{m_addr_info['clean_tokens']}'")
            count += 1
            if count >= 5:
                break

if __name__ == "__main__":
    inspect()
