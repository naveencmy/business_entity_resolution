"""
================================================================================
KAGGLE / CLOUD GPU READY BUSINESS ENTITY RESOLUTION PIPELINE
Target Metric: Macro F_0.5 >= 0.998 - 0.9998
================================================================================
Features:
- Self-contained: runs on Kaggle GPU (T4 / P100 / A100) or Colab with zero extra setup.
- Auto-detects Kaggle input datasets (/kaggle/input/**/dataset or local paths).
- Auto-detects CUDA GPU for accelerated XGBoost hist training & scoring.
- Multilingual Indic Brahmic script transliteration (Hindi, Tamil, Kannada, Gujarati, Odia, Bengali, Telugu).
- Dual-channel inverted index blocking (Name 3-grams + Address Keys & Distinct Tokens).
- Strict format compliance with ML Challenge 2026 validator.
================================================================================
"""
import sys
import os
import re
import csv
import time
import math
import unicodedata
import collections
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier

# -----------------------------------------------------------------------------
# 1. PATH RESOLUTION (Auto-detects Kaggle, Local, or Custom Environment)
# -----------------------------------------------------------------------------
def locate_dataset_dir() -> Path:
    candidates = [
        Path("/kaggle/input"),
        Path("./dataset"),
        Path("../dataset"),
        Path("e:/Projects/Active/Business_pipeline/Deputy_pipe/Datasets/student_resource/dataset"),
        Path("./Deputy_pipe/Datasets/student_resource/dataset"),
    ]
    # Check candidates
    for p in candidates:
        if p.exists():
            # If /kaggle/input, search recursively for test_source1.tsv
            if p == Path("/kaggle/input"):
                for root, dirs, files in os.walk(p):
                    if "test_source1.tsv" in files and "train_source1.tsv" in files:
                        print(f"[Dataset Detector] Found dataset at: {root}")
                        return Path(root)
            elif (p / "test" / "test_source1.tsv").exists() or (p / "test_source1.tsv").exists():
                print(f"[Dataset Detector] Found dataset at: {p}")
                return p
    # Fallback to local default
    return Path("e:/Projects/Active/Business_pipeline/Deputy_pipe/Datasets/student_resource/dataset")

DATASET_DIR = locate_dataset_dir()
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Detect train and test directories
TRAIN_DIR = DATASET_DIR / "train" if (DATASET_DIR / "train").exists() else DATASET_DIR
TEST_DIR = DATASET_DIR / "test" if (DATASET_DIR / "test").exists() else DATASET_DIR

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

OUTPUT_MATCHING = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"

# -----------------------------------------------------------------------------
# 2. NORMALIZATION & MULTILINGUAL TRANSLITERATION
# -----------------------------------------------------------------------------
INDIC_BASE_MAP = {
    0x05: 'a', 0x06: 'aa', 0x07: 'i', 0x08: 'ee', 0x09: 'u', 0x0A: 'oo',
    0x0F: 'e', 0x10: 'ai', 0x13: 'o', 0x14: 'au', 0x15: 'k', 0x16: 'kh',
    0x17: 'g', 0x18: 'gh', 0x19: 'ng', 0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j',
    0x1D: 'jh', 0x1E: 'ny', 0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh',
    0x23: 'n', 0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n',
    0x2A: 'p', 0x2B: 'ph', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm', 0x2F: 'y',
    0x30: 'r', 0x31: 'r', 0x32: 'l', 0x33: 'l', 0x34: 'zh', 0x35: 'v',
    0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h', 0x3E: 'aa', 0x3F: 'i',
    0x40: 'ee', 0x41: 'u', 0x42: 'oo', 0x46: 'e', 0x47: 'e', 0x48: 'ai',
    0x4A: 'o', 0x4B: 'o', 0x4C: 'au', 0x02: 'n', 0x4D: ''
}

def get_indic_latin(code_point: int) -> Optional[str]:
    if 0x0900 <= code_point <= 0x0D7F:
        offset = code_point % 0x0080
        return INDIC_BASE_MAP.get(offset)
    return None

LEGAL_TERMS = [
    r'\bprivate limited\b', r'\bpvt ltd\b', r'\bpvt\.?\s*ltd\.?\b', r'\bprivate ltd\b',
    r'\blimited\b', r'\bltd\.?\b', r'\bllp\b', r'\bllc\b', r'\bl\.l\.c\.?\b',
    r'\bincorporated\b', r'\binc\.?\b', r'\bcorporation\b', r'\bcorp\.?\b',
    r'\bco\.?\b', r'\bcompany\b', r'\bsarl\b', r'\bs\.a\.r\.l\.?\b',
    r'\bsasu\b', r'\bsas\b', r'\bsci\b', r'\beurl\b', r'\bsa\b',
    r'\bgmbh\b', r'\btrust\b', r'\bsociety\b', r'\bassociates\b',
    r'\benterprises\b', r'\bholding company\b', r'\bholdings\b',
    r'\bgroup\b', r'\bfils\b', r'\b& fils\b', r'\bet fils\b'
]
LEGAL_PATTERN = re.compile('|'.join(LEGAL_TERMS), re.IGNORECASE)
URL_PATTERN = re.compile(r'(?:https?://|www\.)?([a-zA-Z0-9-]+)\.(?:com|org|net|in|co|fr|io|biz|info)', re.IGNORECASE)
PUNCT_PATTERN = re.compile(r'[^a-zA-Z0-9\s]')
INDIA_PIN_PATTERN = re.compile(r'\b([1-9][0-9]{5})\b')
US_ZIP_PATTERN = re.compile(r'\b([0-9]{5})(?:-[0-9]{4})?\b')
FRANCE_CP_PATTERN = re.compile(r'\b(0[1-9]|[1-8][0-9]|9[0-5]|97|98)[0-9]{3}\b')

US_STATES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
}

def transliterate_to_latin(text: str) -> str:
    if not text:
        return ""
    if text.isascii():
        return text
    res = []
    for ch in text:
        cp = ord(ch)
        indic = get_indic_latin(cp)
        if indic is not None:
            res.append(indic)
        else:
            res.append(ch)
    combined = "".join(res)
    nfkd = unicodedata.normalize('NFKD', combined)
    ascii_bytes = nfkd.encode('ASCII', 'ignore')
    return ascii_bytes.decode('utf-8')

def clean_business_name(name: str) -> Tuple[str, str]:
    if not name or not isinstance(name, str):
        return "", ""
    if " dba " in f" {name.lower()} " or " dba: " in f" {name.lower()} ":
        name = re.sub(r'\b(?:dba|doing business as)[:\s]+', ' ', name, flags=re.IGNORECASE)
    text = transliterate_to_latin(name)
    text = URL_PATTERN.sub(r' \1 ', text)
    text = text.replace('&', ' and ')
    text = PUNCT_PATTERN.sub(' ', text).lower()
    tokens = text.split()
    deduped_tokens = []
    for tok in tokens:
        clean_tok = re.sub(r'^0+([0-9])', r'\1', tok) if tok.isdigit() else tok
        if not deduped_tokens or clean_tok != deduped_tokens[-1]:
            deduped_tokens.append(clean_tok)
    cleaned_full = " ".join(deduped_tokens)
    stem = LEGAL_PATTERN.sub('', cleaned_full)
    stem = " ".join(stem.split())
    if not stem and cleaned_full:
        stem = cleaned_full
    return cleaned_full, stem

def clean_address(address: str, country: str) -> Dict[str, str]:
    if not address or not isinstance(address, str):
        return {"postal_code": "", "locality": "", "clean_tokens": ""}
    text = transliterate_to_latin(address)
    postal_code = ""
    country_upper = (country or "").strip().upper()
    if country_upper == "INDIA":
        m = INDIA_PIN_PATTERN.search(text)
        if m: postal_code = m.group(0)
    elif country_upper == "US":
        m = US_ZIP_PATTERN.search(text)
        if m: postal_code = m.group(0)
    elif country_upper == "FRANCE":
        m = FRANCE_CP_PATTERN.search(text)
        if m: postal_code = m.group(0)
    else:
        m = re.search(r'\b([0-9]{5,6})\b', text)
        if m: postal_code = m.group(0)

    locality = ""
    if country_upper == "US":
        words = set(re.findall(r'\b[A-Za-z]{2}\b', text.upper()))
        found_states = words.intersection(US_STATES)
        if found_states:
            locality = sorted(list(found_states))[0]

    text = re.sub(r'\b0+([1-9][0-9]*)\b', r'\1', text)
    text = re.sub(r'([A-Za-z]+)-0+([1-9][0-9]*)', r'\1-\2', text)
    clean_addr = PUNCT_PATTERN.sub(' ', text).lower()
    tokens = [tok for tok in clean_addr.split() if len(tok) > 1]
    addr_stopwords = {
        'near', 'opp', 'opposite', 'behind', 'bh', 'floor', 'flat', 'unit',
        'plot', 'building', 'bldg', 'house', 'no', 'hno', 'road', 'rd',
        'street', 'st', 'lane', 'avenue', 'ave', 'boulevard', 'blvd', 'rue', 'null'
    }
    filtered_tokens = [tok for tok in tokens if tok not in addr_stopwords]
    return {
        "postal_code": postal_code,
        "locality": locality,
        "clean_tokens": " ".join(filtered_tokens)
    }

# -----------------------------------------------------------------------------
# 3. MULTI-TIER BLOCKING & CANDIDATE GENERATION
# -----------------------------------------------------------------------------
def extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    if not text or len(text) < n:
        return {text} if text else set()
    cleaned = f"^{text.strip()}$"
    return {cleaned[i:i+n] for i in range(len(cleaned) - n + 1)}

def extract_address_keys(clean_addr: str, postal_code: str, locality: str) -> List[str]:
    keys = []
    tokens = [tok for tok in clean_addr.split() if len(tok) > 1]
    numbers = [tok for tok in tokens if any(c.isdigit() for c in tok)]
    words = [tok for tok in tokens if not any(c.isdigit() for c in tok)]
    num_key = numbers[0] if numbers else ""
    first_word = words[0] if words else ""
    second_word = words[1] if len(words) > 1 else ""
    if postal_code:
        keys.append(f"P#{postal_code}")
        if num_key: keys.append(f"P#{postal_code}#{num_key}")
        if first_word: keys.append(f"P#{postal_code}#{first_word}")
    if locality and num_key:
        keys.append(f"L#{locality}#{num_key}")
    if num_key and first_word:
        keys.append(f"N#{num_key}#{first_word}")
        if second_word: keys.append(f"N#{num_key}#{first_word}#{second_word}")
    elif len(words) >= 2:
        keys.append(f"W#{first_word}#{second_word}")
    return keys

class CountryCandidateIndex:
    def __init__(self, country: str, max_candidates: int = 15):
        self.country = country
        self.max_candidates = max_candidates
        self.targets: Dict[str, Dict[str, Any]] = {}
        self.stem_index: Dict[str, List[str]] = collections.defaultdict(list)
        self.addr_key_index: Dict[str, List[str]] = collections.defaultdict(list)
        self.gram_index: Dict[str, List[str]] = collections.defaultdict(list)
        self.gram_df: Dict[str, int] = collections.defaultdict(int)

    def add_target(self, rec: Dict[str, Any]):
        tid = rec["entity_id"]
        self.targets[tid] = rec
        stem = rec.get("name_stem", "").strip().lower()
        if stem:
            self.stem_index[stem].append(tid)
            words = stem.split()
            if len(words) >= 2:
                prefix_key = " ".join(words[:2])
                self.stem_index[prefix_key].append(tid)
            grams = extract_char_ngrams(stem, n=3)
            for g in grams:
                self.gram_index[g].append(tid)
                self.gram_df[g] += 1
        addr_keys = extract_address_keys(
            rec.get("clean_addr", ""),
            rec.get("postal_code", ""),
            rec.get("locality", "")
        )
        for k in addr_keys:
            self.addr_key_index[k].append(tid)

    def finalize_index(self):
        total = len(self.targets)
        if total == 0: return
        max_df = max(500, int(total * 0.05))
        pruned_grams = [g for g, count in self.gram_df.items() if count > max_df]
        for g in pruned_grams:
            del self.gram_index[g]
        pruned_addr = [k for k, tids in self.addr_key_index.items() if len(tids) > 150]
        for k in pruned_addr:
            del self.addr_key_index[k]

    def query_candidates(self, s1_rec: Dict[str, Any]) -> List[str]:
        scores: Dict[str, float] = collections.defaultdict(float)
        s1_stem = s1_rec.get("name_stem", "").strip().lower()
        if s1_stem:
            for tid in self.stem_index.get(s1_stem, []):
                scores[tid] += 10.0
            words = s1_stem.split()
            if len(words) >= 2:
                prefix_key = " ".join(words[:2])
                for tid in self.stem_index.get(prefix_key, []):
                    scores[tid] += 5.0
        s1_addr_keys = extract_address_keys(
            s1_rec.get("clean_addr", ""),
            s1_rec.get("postal_code", ""),
            s1_rec.get("locality", "")
        )
        for k in s1_addr_keys:
            for tid in self.addr_key_index.get(k, []):
                scores[tid] += 8.0
        if s1_stem:
            s1_grams = extract_char_ngrams(s1_stem, n=3)
            num_s1_grams = len(s1_grams)
            if num_s1_grams > 0:
                gram_hits: Dict[str, int] = collections.defaultdict(int)
                for g in s1_grams:
                    for tid in self.gram_index.get(g, []):
                        gram_hits[tid] += 1
                for tid, hits in gram_hits.items():
                    target_rec = self.targets.get(tid)
                    if target_rec:
                        t_stem = target_rec.get("name_stem", "")
                        t_grams_count = max(1, len(t_stem) - 2)
                        overlap = hits / (num_s1_grams + t_grams_count - hits + 1e-5)
                        if overlap >= 0.25:
                            scores[tid] += overlap * 6.0
        if not scores: return []
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in ranked[:self.max_candidates]]

# -----------------------------------------------------------------------------
# 4. PAIRWISE FEATURE EXTRACTION
# -----------------------------------------------------------------------------
def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b: return 1.0
    if not set_a or not set_b: return 0.0
    u = len(set_a.union(set_b))
    return len(set_a.intersection(set_b)) / u if u > 0 else 0.0

def containment_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b: return 0.0
    m = min(len(set_a), len(set_b))
    return len(set_a.intersection(set_b)) / m if m > 0 else 0.0

def levenshtein_similarity(s1: str, s2: str) -> float:
    if s1 == s2: return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0: return 0.0
    if abs(len1 - len2) > max(len1, len2) * 0.7: return 0.0
    s1_t, s2_t = s1[:60], s2[:60]
    m, n = len(s1_t), len(s2_t)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if s1_t[i - 1] == s2_t[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return max(0.0, 1.0 - (dp[n] / max(m, n)))

def extract_pairwise_features(s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]) -> List[float]:
    stem1 = s1_rec.get("name_stem", "").strip().lower()
    stem2 = cand_rec.get("name_stem", "").strip().lower()
    full1 = s1_rec.get("name_full", "").strip().lower()
    full2 = cand_rec.get("name_full", "").strip().lower()

    exact_stem = 1.0 if (stem1 and stem1 == stem2) else 0.0
    exact_full = 1.0 if (full1 and full1 == full2) else 0.0

    grams1 = extract_char_ngrams(stem1, 3)
    grams2 = extract_char_ngrams(stem2, 3)
    name_gram_jaccard = jaccard_similarity(grams1, grams2)

    tokens1 = set(stem1.split())
    tokens2 = set(stem2.split())
    name_token_jaccard = jaccard_similarity(tokens1, tokens2)
    name_token_containment = containment_similarity(tokens1, tokens2)

    name_lev = levenshtein_similarity(stem1, stem2)
    w1 = stem1.split()[0] if stem1 else ""
    w2 = stem2.split()[0] if stem2 else ""
    first_word_match = 1.0 if (w1 and w1 == w2) else 0.0

    len1, len2 = len(stem1), len(stem2)
    max_len = max(len1, len2)
    name_len_ratio = (min(len1, len2) / max_len) if max_len > 0 else 1.0

    p1 = s1_rec.get("postal_code", "").strip()
    p2 = cand_rec.get("postal_code", "").strip()
    if not p1 and not p2: postal_match = -0.5
    elif p1 and p2: postal_match = 1.0 if p1 == p2 else -1.0
    else: postal_match = 0.0

    loc1 = s1_rec.get("locality", "").strip().upper()
    loc2 = cand_rec.get("locality", "").strip().upper()
    locality_match = 1.0 if (loc1 and loc2 and loc1 == loc2) else 0.0

    addr1 = s1_rec.get("clean_addr", "").strip().lower()
    addr2 = cand_rec.get("clean_addr", "").strip().lower()
    addr_toks1 = set(addr1.split())
    addr_toks2 = set(addr2.split())
    addr_jaccard = jaccard_similarity(addr_toks1, addr_toks2)
    addr_containment = containment_similarity(addr_toks1, addr_toks2)

    nums1 = {t for t in addr_toks1 if any(c.isdigit() for c in t)}
    nums2 = {t for t in addr_toks2 if any(c.isdigit() for c in t)}
    addr_num_overlap = jaccard_similarity(nums1, nums2) if (nums1 or nums2) else 0.5

    name_x_addr = name_gram_jaccard * max(addr_jaccard, addr_containment)
    name_or_addr_max = max(name_gram_jaccard, max(addr_jaccard, addr_containment))
    cand_id = cand_rec.get("entity_id", "")
    is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    return [
        exact_stem, exact_full, name_gram_jaccard, name_token_jaccard,
        name_token_containment, name_lev, first_word_match, name_len_ratio,
        postal_match, locality_match, addr_jaccard, addr_containment,
        addr_num_overlap, name_x_addr, name_or_addr_max, is_s3
    ]

# -----------------------------------------------------------------------------
# 5. METRIC & MATCHING MODEL (GPU ACCELERATED)
# -----------------------------------------------------------------------------
def compute_macro_f05(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> Dict[str, float]:
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue
        tp = len(true_set.intersection(pred_set))
        p = tp / len(pred_set)
        r = tp / len(true_set)
        denom = (0.25 * p) + r
        scores.append((1.25 * p * r) / denom if denom > 0 else 0.0)
    return {"macro_f05": float(np.mean(scores)) if scores else 0.0}

class EntityMatcher:
    def __init__(self):
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"[Device Detector] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        except ImportError:
            pass

        self.clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            device=device,
            scale_pos_weight=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        self.optimal_threshold: float = 0.65

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def optimize_threshold(self, candidate_pairs: List[Tuple[str, str]], probs: np.ndarray, gt: Dict[str, Set[str]]):
        best_t, best_score = 0.65, -1.0
        for tau in np.arange(0.40, 0.95, 0.02):
            preds = {s1: set() for s1 in gt.keys()}
            for (s1_id, cid), p in zip(candidate_pairs, probs):
                if p >= tau: preds[s1_id].add(cid)
            sc = compute_macro_f05(gt, preds)["macro_f05"]
            if sc > best_score:
                best_score = sc
                best_t = float(tau)
        print(f"Optimal Threshold: {best_t:.2f} (Macro F_0.5: {best_score:.4f})")
        self.optimal_threshold = best_t

# -----------------------------------------------------------------------------
# 6. END-TO-END PIPELINE EXECUTION
# -----------------------------------------------------------------------------
def run():
    print("=" * 60)
    print("RUNNING KAGGLE BUSINESS ENTITY RESOLUTION PIPELINE")
    print(f"Dataset root: {DATASET_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # Step A: Train model on ground truth
    print("\n[Phase 1] Ingesting training ground truth...")
    gt = {}
    with open(TRAIN_GT, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                gt[parts[0]] = set(parts[1].split(","))
                if len(gt) >= 2500: break

    target_s1 = set(gt.keys())
    target_matches = set()
    for m in gt.values(): target_matches.update(m)

    s1_train = {}
    with open(TRAIN_S1, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row and row[0] in target_s1:
                name_full, name_stem = clean_business_name(row[1])
                addr = clean_address(row[2], row[3])
                s1_train[row[0]] = {
                    "entity_id": row[0], "name_full": name_full, "name_stem": name_stem,
                    "postal_code": addr["postal_code"], "locality": addr["locality"],
                    "clean_addr": addr["clean_tokens"], "country": row[3].strip()
                }

    train_indices = {c: CountryCandidateIndex(c, max_candidates=15) for c in {r["country"] for r in s1_train.values()}}
    distractor_cap = 40000
    for path in [TRAIN_S2, TRAIN_S3]:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            distractors = 0
            for row in r:
                if not row or len(row) < 4: continue
                c = row[3].strip()
                if c not in train_indices: continue
                tid = row[0]
                is_m = tid in target_matches
                if is_m or distractors < distractor_cap:
                    if not is_m: distractors += 1
                    name_full, name_stem = clean_business_name(row[1])
                    addr = clean_address(row[2], c)
                    train_indices[c].add_target({
                        "entity_id": tid, "name_full": name_full, "name_stem": name_stem,
                        "postal_code": addr["postal_code"], "locality": addr["locality"],
                        "clean_addr": addr["clean_tokens"], "country": c
                    })

    for c, idx in train_indices.items(): idx.finalize_index()

    X_list, y_list, pairs_list = [], [], []
    for s1_id, s1_rec in s1_train.items():
        c = s1_rec["country"]
        idx = train_indices[c]
        true_set = gt.get(s1_id, set())
        cands = idx.query_candidates(s1_rec)
        for t_id in true_set:
            if t_id in idx.targets and t_id not in cands: cands.append(t_id)
        for cid in cands:
            cand_rec = idx.targets.get(cid)
            if not cand_rec: continue
            X_list.append(extract_pairwise_features(s1_rec, cand_rec))
            y_list.append(1 if cid in true_set else 0)
            pairs_list.append((s1_id, cid))

    matcher = EntityMatcher()
    matcher.fit(np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32))
    probs = matcher.predict_proba(np.array(X_list, dtype=np.float32))
    matcher.optimize_threshold(pairs_list, probs, gt)
    del train_indices, s1_train

    # Step B: Test Set Inference
    print("\n[Phase 2] Loading test_source1.tsv...")
    s1_all_ids = []
    s1_by_country = collections.defaultdict(list)
    with open(TEST_S1, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if not row or len(row) < 4: continue
            s1_id, c = row[0].strip(), row[3].strip()
            s1_all_ids.append(s1_id)
            name_full, name_stem = clean_business_name(row[1])
            addr = clean_address(row[2], c)
            s1_by_country[c].append({
                "entity_id": s1_id, "name_full": name_full, "name_stem": name_stem,
                "postal_code": addr["postal_code"], "locality": addr["locality"],
                "clean_addr": addr["clean_tokens"], "country": c
            })

    results_cands = {}
    results_matches = {}

    for country, s1_list in s1_by_country.items():
        print(f"\nProcessing Country Partition: {country} ({len(s1_list)} entities)...")
        idx = CountryCandidateIndex(country, max_candidates=15)
        for path in [TEST_S2, TEST_S3]:
            with open(path, "r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                next(r)
                for row in r:
                    if not row or len(row) < 4 or row[3].strip() != country: continue
                    name_full, name_stem = clean_business_name(row[1])
                    addr = clean_address(row[2], country)
                    idx.add_target({
                        "entity_id": row[0].strip(), "name_full": name_full, "name_stem": name_stem,
                        "postal_code": addr["postal_code"], "locality": addr["locality"],
                        "clean_addr": addr["clean_tokens"], "country": country
                    })
        idx.finalize_index()
        print(f"Index built ({len(idx.targets)} targets). Running inference...")

        for i, s1_rec in enumerate(s1_list):
            s1_id = s1_rec["entity_id"]
            cands = idx.query_candidates(s1_rec)
            if not cands:
                results_cands[s1_id] = ""
                results_matches[s1_id] = ""
                continue
            results_cands[s1_id] = ",".join(cands)
            cand_recs = [idx.targets[cid] for cid in cands if cid in idx.targets]
            if not cand_recs:
                results_matches[s1_id] = ""
                continue
            feats = np.array([extract_pairwise_features(s1_rec, cr) for cr in cand_recs], dtype=np.float32)
            p = matcher.predict_proba(feats)
            m_ids = [cand_recs[j]["entity_id"] for j, prob in enumerate(p) if prob >= matcher.optimal_threshold]
            results_matches[s1_id] = ",".join(m_ids) if m_ids else ""

            if (i + 1) % 100000 == 0 or (i + 1) == len(s1_list):
                print(f"  [{country}] {i + 1}/{len(s1_list)} complete.")
        del idx

    # Step C: Write outputs
    print("\nWriting output/candidate_pairs.tsv...")
    with open(OUTPUT_CANDIDATES, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_all_ids:
            f.write(f"{s1}\t{results_cands.get(s1, '')}\n")

    print("Writing output/matching_results.tsv...")
    with open(OUTPUT_MATCHING, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_all_ids:
            f.write(f"{s1}\t{results_matches.get(s1, '')}\n")

    print("\nOutputs generated successfully!")
    print(f"Matching Results: {OUTPUT_MATCHING}")
    print(f"Candidate Pairs:  {OUTPUT_CANDIDATES}")

if __name__ == "__main__":
    run()
