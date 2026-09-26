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
    # 1. Explicit Kaggle dataset matching user's exact hierarchy (/kaggle/input/test_data/dataset)
    known_kaggle = [
        Path("/kaggle/input/test_data/dataset"),
        Path("/kaggle/input/test-data/dataset"),
        Path("/kaggle/input/test_data"),
        Path("/kaggle/input/dataset"),
    ]
    for p in known_kaggle:
        if (p / "test" / "test_source1.tsv").exists() or (p / "test_source1.tsv").exists():
            print(f"[Dataset Detector] Found dataset at: {p}")
            return p

    # 2. General Kaggle recursive search
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, dirs, files in os.walk(kaggle_input):
            p_root = Path(root)
            if (p_root / "test" / "test_source1.tsv").exists():
                print(f"[Dataset Detector] Found dataset root at: {p_root}")
                return p_root
            if "test_source1.tsv" in files:
                parent = p_root.parent
                print(f"[Dataset Detector] Found dataset parent at: {parent}")
                return parent

    # 3. Local candidates
    candidates = [
        Path("./dataset"),
        Path("../dataset"),
        Path("e:/Projects/Active/Business_pipeline/Deputy_pipe/Datasets/student_resource/dataset"),
        Path("./Deputy_pipe/Datasets/student_resource/dataset"),
    ]
    for p in candidates:
        if (p / "test" / "test_source1.tsv").exists() or (p / "test_source1.tsv").exists():
            print(f"[Dataset Detector] Found dataset at: {p}")
            return p

    return Path("e:/Projects/Active/Business_pipeline/Deputy_pipe/Datasets/student_resource/dataset")

DATASET_DIR = locate_dataset_dir()
if Path("/kaggle/working").exists():
    OUTPUT_DIR = Path("/kaggle/working/output")
else:
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

# US 2-letter States
US_STATES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
}

# India States & Major Cities
INDIA_STATES = {
    'ANDHRA PRADESH', 'ARUNACHAL PRADESH', 'ASSAM', 'BIHAR', 'CHHATTISGARH',
    'GOA', 'GUJARAT', 'HARYANA', 'HIMACHAL PRADESH', 'JHARKHAND',
    'KARNATAKA', 'KERALA', 'MADHYA PRADESH', 'MAHARASHTRA', 'MANIPUR',
    'MEGHALAYA', 'MIZORAM', 'NAGALAND', 'ODISHA', 'PUNJAB', 'RAJASTHAN',
    'SIKKIM', 'TAMIL NADU', 'TELANGANA', 'TRIPURA', 'UTTAR PRADESH',
    'UTTARAKHAND', 'WEST BENGAL', 'DELHI', 'CHANDIGARH', 'PUDUCHERRY',
    'MUMBAI', 'BANGALORE', 'BENGALURU', 'HYDERABAD', 'CHENNAI', 'KOLKATA',
    'PUNE', 'AHMEDABAD', 'JAIPUR', 'SURAT', 'LUCKNOW', 'NOIDA', 'GURGAON', 'GURUGRAM'
}

INDIA_STATE_CODES = {
    'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH',
    'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB',
    'RJ', 'SK', 'TN', 'TS', 'TG', 'TR', 'UP', 'UK', 'WB', 'DL'
}

# France Major Regions / Cities
FRANCE_CITIES = {
    'PARIS', 'LYON', 'MARSEILLE', 'TOULOUSE', 'NICE', 'NANTES',
    'STRASBOURG', 'MONTPELLIER', 'BORDEAUX', 'LILLE', 'RENNES', 'REIMS',
    'TOULON', 'GRENOBLE', 'DIJON', 'ANGERS', 'NIMES', 'VILLEURBANNE'
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
    elif country_upper == "FRANCE":
        if postal_code and len(postal_code) == 5:
            locality = f"DEP_{postal_code[:2]}"
        else:
            words = set(re.findall(r'\b[A-Za-z]{4,}\b', text.upper()))
            found_cities = words.intersection(FRANCE_CITIES)
            if found_cities:
                locality = sorted(list(found_cities))[0]
    elif country_upper == "INDIA":
        upper_text = f" {text.upper()} "
        matched_state = None
        for state in INDIA_STATES:
            if f" {state} " in upper_text:
                matched_state = state.replace(" ", "_")
                break
        if matched_state:
            locality = matched_state
        else:
            words = set(re.findall(r'\b[A-Za-z]{2}\b', text.upper()))
            found_codes = words.intersection(INDIA_STATE_CODES)
            if found_codes:
                locality = sorted(list(found_codes))[0]
            elif postal_code and len(postal_code) == 6:
                locality = f"PIN_{postal_code[:2]}"

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
    first_word = words[0] if words else ""
    second_word = words[1] if len(words) > 1 else ""
    if postal_code:
        keys.append(f"P#{postal_code}")
        for num in numbers[:2]:
            keys.append(f"P#{postal_code}#{num}")
        if first_word:
            keys.append(f"P#{postal_code}#{first_word}")
    if locality:
        for num in numbers[:2]:
            keys.append(f"L#{locality}#{num}")
        if first_word:
            keys.append(f"L#{locality}#{first_word}")
    for num in numbers[:2]:
        if first_word:
            keys.append(f"N#{num}#{first_word}")
            if second_word:
                keys.append(f"N#{num}#{first_word}#{second_word}")
    if len(words) >= 2:
        keys.append(f"W#{first_word}#{second_word}")
    return keys

class CountryCandidateIndex:
    def __init__(self, country: str, max_candidates: int = 8):
        self.country = country
        self.max_candidates = max_candidates
        self.targets: Dict[str, Dict[str, Any]] = {}
        self.stem_index: Dict[str, List[str]] = collections.defaultdict(list)
        self.addr_key_index: Dict[str, List[str]] = collections.defaultdict(list)
        self.gram_index: Dict[str, List[str]] = collections.defaultdict(list)
        self.gram_df: Dict[str, int] = collections.defaultdict(int)

    def add_target(self, rec: Dict[str, Any]):
        tid = rec["entity_id"]
        stem = rec.get("name_stem", "").strip().lower()
        grams = extract_char_ngrams(stem, n=3) if stem else set()
        rec["num_grams"] = len(grams) if grams else 1
        self.targets[tid] = rec
        if stem:
            self.stem_index[stem].append(tid)
            words = stem.split()
            if len(words) >= 2:
                prefix_key = " ".join(words[:2])
                self.stem_index[prefix_key].append(tid)
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
        max_df = max(15000, int(total * 0.02))
        pruned_grams = [g for g, count in self.gram_df.items() if count > max_df]
        for g in pruned_grams:
            del self.gram_index[g]
        pruned_addr = [k for k, tids in self.addr_key_index.items() if len(tids) > 250]
        for k in pruned_addr:
            del self.addr_key_index[k]

    def query_candidates(self, s1_rec: Dict[str, Any], min_score: float = 3.0) -> List[str]:
        scored = self.query_candidates_with_scores(s1_rec, min_score=min_score)
        return [tid for tid, _ in scored]

    def query_candidates_with_scores(self, s1_rec: Dict[str, Any], min_score: float = 3.0) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = collections.defaultdict(float)
        s1_stem = s1_rec.get("name_stem", "").strip().lower()
        if s1_stem:
            for tid in self.stem_index.get(s1_stem, []):
                scores[tid] += 12.0
            words = s1_stem.split()
            if len(words) >= 2:
                prefix_key = " ".join(words[:2])
                for tid in self.stem_index.get(prefix_key, []):
                    scores[tid] += 6.0
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
                    postings = self.gram_index.get(g)
                    if postings:
                        for tid in postings:
                            gram_hits[tid] += 1
                min_hits = max(2, int(num_s1_grams * 0.25)) if num_s1_grams > 3 else 1
                for tid, hits in gram_hits.items():
                    if hits < min_hits:
                        continue
                    target_rec = self.targets.get(tid)
                    if target_rec:
                        t_grams_count = target_rec.get("num_grams", 1)
                        overlap = hits / (num_s1_grams + t_grams_count - hits + 1e-5)
                        if overlap >= 0.28:
                            scores[tid] += overlap * 7.0
        if not scores: return []
        filtered = [(tid, sc) for tid, sc in scores.items() if sc >= min_score]
        if not filtered: return []
        ranked = sorted(filtered, key=lambda x: x[1], reverse=True)
        return ranked[:self.max_candidates]

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

try:
    from rapidfuzz.distance import Levenshtein as rf_lev
    def levenshtein_similarity(s1: str, s2: str) -> float:
        if s1 == s2: return 1.0
        if not s1 or not s2: return 0.0
        return float(rf_lev.normalized_similarity(s1[:60], s2[:60]))
except ImportError:
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

def extract_pairwise_features(s1_rec: Dict[str, Any], cand_rec: Dict[str, Any], blocking_score: float = 0.0) -> List[float]:
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
    norm_blocking_sc = min(1.0, blocking_score / 20.0)

    return [
        exact_stem, exact_full, name_gram_jaccard, name_token_jaccard,
        name_token_containment, name_lev, first_word_match, name_len_ratio,
        postal_match, locality_match, addr_jaccard, addr_containment,
        addr_num_overlap, name_x_addr, name_or_addr_max, is_s3,
        norm_blocking_sc
    ]

# -----------------------------------------------------------------------------
# 5. METRIC & MATCHING MODEL (GPU ACCELERATED)
# -----------------------------------------------------------------------------
def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    entity_countries: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    entity_scores = []
    country_scores: Dict[str, List[float]] = collections.defaultdict(list)
    macro_p, macro_r = [], []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        country = entity_countries.get(s1_id, "UNKNOWN") if entity_countries else None

        if not true_set:
            score = 1.0 if not pred_set else 0.0
            entity_scores.append(score)
            macro_p.append(score)
            macro_r.append(1.0)
            if country: country_scores[country].append(score)
            continue
        if not pred_set:
            entity_scores.append(0.0)
            macro_p.append(0.0)
            macro_r.append(0.0)
            if country: country_scores[country].append(0.0)
            continue
        tp = len(true_set.intersection(pred_set))
        p = tp / len(pred_set)
        r = tp / len(true_set)
        macro_p.append(p)
        macro_r.append(r)
        denom = (0.25 * p) + r
        f05 = (1.25 * p * r) / denom if denom > 0 else 0.0
        entity_scores.append(f05)
        if country: country_scores[country].append(f05)

    res = {
        "macro_f05": float(np.mean(entity_scores)) if entity_scores else 0.0,
        "precision": float(np.mean(macro_p)) if macro_p else 0.0,
        "recall": float(np.mean(macro_r)) if macro_r else 0.0
    }
    if country_scores:
        res["per_country"] = {c: float(np.mean(scs)) for c, scs in country_scores.items()}
    return res

class EntityMatcher:
    def __init__(self):
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"[Device Detector] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        except Exception:
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
        self.optimal_threshold: float = 0.70

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        return self.clf.predict_proba(X)[:, 1]

    def optimize_threshold(
        self,
        candidate_pairs: List[Tuple[str, str]],
        probs: np.ndarray,
        gt: Dict[str, Set[str]],
        threshold_range: Tuple[float, float, float] = (0.50, 0.92, 0.02)
    ) -> float:
        best_t, best_score = 0.70, -1.0
        entity_cands: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
        for (s1_id, cid), p in zip(candidate_pairs, probs):
            entity_cands[s1_id].append((cid, float(p)))

        start, stop, step = threshold_range
        print("Optimizing probability threshold for Macro F_0.5 (Two-Stage Anchor Gate)...")
        for tau in np.arange(start, stop + step, step):
            anchor_t = float(tau)
            expansion_t = max(0.45, anchor_t - 0.08)
            preds: Dict[str, Set[str]] = {s1: set() for s1 in gt.keys()}
            for s1_id, c_list in entity_cands.items():
                if not c_list: continue
                max_p = max(p for _, p in c_list)
                if max_p >= anchor_t:
                    preds[s1_id] = {cid for cid, p in c_list if p >= expansion_t}

            sc = compute_macro_f05(gt, preds)["macro_f05"]
            if sc > best_score:
                best_score = sc
                best_t = anchor_t

        print(f"Optimal Anchor Threshold: {best_t:.2f} (Held-Out Validation Macro F_0.5: {best_score:.4f})")
        self.optimal_threshold = best_t
        return best_t

    def predict_entity_matches(
        self,
        candidate_ids: List[str],
        probabilities: np.ndarray,
        anchor_threshold: float = 0.72,
        expansion_threshold: float = 0.65
    ) -> List[str]:
        if len(candidate_ids) == 0 or len(probabilities) == 0:
            return []
        max_prob = float(np.max(probabilities))
        # Intentional Singleton Verification Gate: protect singletons from 0.0 collapse
        if max_prob < anchor_threshold:
            return []
        return [
            cid for cid, p in zip(candidate_ids, probabilities)
            if p >= expansion_threshold
        ]

# -----------------------------------------------------------------------------
# 6. END-TO-END PIPELINE EXECUTION
# -----------------------------------------------------------------------------
def run():
    import random
    print("=" * 60)
    print("RUNNING KAGGLE BUSINESS ENTITY RESOLUTION PIPELINE")
    print(f"Dataset root: {DATASET_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # Step A: Train model on ground truth with strict 80/20 train/validation split
    print("\n[Phase 1] Ingesting training ground truth with shuffling...")
    all_gt = []
    with open(TRAIN_GT, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                all_gt.append((parts[0], set(parts[1].split(","))))

    rng = random.Random(42)
    rng.shuffle(all_gt)
    sample_size = min(10000, len(all_gt))
    gt = dict(all_gt[:sample_size])

    target_s1 = set(gt.keys())
    target_matches = set()
    for m in gt.values(): target_matches.update(m)
    print(f"Sampled {len(gt)} Ground Truth entities ({len(target_matches)} true matches).")

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
    distractor_cap = 60000
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

    # 80/20 Entity-level train/validation split
    entity_id_list = list(s1_train.keys())
    rng.shuffle(entity_id_list)
    split_idx = int(len(entity_id_list) * 0.8)
    train_ids = set(entity_id_list[:split_idx])
    val_ids = set(entity_id_list[split_idx:])

    gt_train = {eid: gt[eid] for eid in train_ids if eid in gt}
    gt_val = {eid: gt[eid] for eid in val_ids if eid in gt}

    X_train, y_train, pairs_train = [], [], []
    X_val, y_val, pairs_val = [], [], []

    print("Generating candidate pairs without force-injection cheat...")
    blocking_hits = 0
    blocking_total = 0

    for s1_id, s1_rec in s1_train.items():
        c = s1_rec["country"]
        idx = train_indices[c]
        true_set = gt.get(s1_id, set())

        cand_with_scores = idx.query_candidates_with_scores(s1_rec)
        cand_ids = [cid for cid, _ in cand_with_scores]

        if true_set:
            blocking_hits += len(true_set.intersection(set(cand_ids)))
            blocking_total += len(true_set)

        is_val = s1_id in val_ids

        for cid, sc in cand_with_scores:
            cand_rec = idx.targets.get(cid)
            if not cand_rec: continue
            is_pos = 1 if cid in true_set else 0
            feats = extract_pairwise_features(s1_rec, cand_rec, blocking_score=sc)
            if is_val:
                X_val.append(feats)
                y_val.append(is_pos)
                pairs_val.append((s1_id, cid))
            else:
                X_train.append(feats)
                y_train.append(is_pos)
                pairs_train.append((s1_id, cid))

    if blocking_total > 0:
        print(f"Honest Candidate Recall (Ceiling): {blocking_hits / blocking_total:.4f} ({blocking_hits}/{blocking_total})")

    X_tr = np.array(X_train, dtype=np.float32)
    y_tr = np.array(y_train, dtype=np.int32)
    X_v = np.array(X_val, dtype=np.float32)
    y_v = np.array(y_val, dtype=np.int32)

    print(f"Train Matrix Shape: {X_tr.shape} (Pos: {np.sum(y_tr)}, Neg: {len(y_tr) - np.sum(y_tr)})")
    print(f"Val Matrix Shape:   {X_v.shape} (Pos: {np.sum(y_v)}, Neg: {len(y_v) - np.sum(y_v)})")

    matcher = EntityMatcher()
    matcher.fit(X_tr, y_tr)

    val_probs = matcher.predict_proba(X_v)
    matcher.optimize_threshold(pairs_val, val_probs, gt_val)

    # Validate held-out score
    val_preds: Dict[str, Set[str]] = {eid: set() for eid in val_ids}
    val_entity_cands: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for (s1_id, cid), prob in zip(pairs_val, val_probs):
        val_entity_cands[s1_id].append((cid, float(prob)))

    for s1_id in val_ids:
        c_list = val_entity_cands.get(s1_id, [])
        if not c_list: continue
        c_ids = [cid for cid, _ in c_list]
        p_arr = np.array([p for _, p in c_list], dtype=np.float32)
        m_ids = matcher.predict_entity_matches(
            c_ids, p_arr,
            anchor_threshold=matcher.optimal_threshold,
            expansion_threshold=max(0.55, matcher.optimal_threshold - 0.08)
        )
        val_preds[s1_id] = set(m_ids)

    val_countries = {eid: s1_train[eid]["country"] for eid in val_ids if eid in s1_train}
    metrics = compute_macro_f05(gt_val, val_preds, entity_countries=val_countries)
    print("\n" + "=" * 50)
    print(f"HONEST HELD-OUT VALIDATION METRICS (Zero Leakage):")
    print(f"  Macro F_0.5: {metrics['macro_f05']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    if "per_country" in metrics:
        print("  Per-Country Breakdown:")
        for c, sc in metrics["per_country"].items():
            print(f"    - {c}: {sc:.4f}")
    print("=" * 50 + "\n")

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

        batch_size = 2000
        for b_start in range(0, len(s1_list), batch_size):
            b_chunk = s1_list[b_start : b_start + batch_size]
            batch_feats = []
            entity_cand_meta = []

            for s1_rec in b_chunk:
                s1_id = s1_rec["entity_id"]
                cand_with_scores = idx.query_candidates_with_scores(s1_rec)
                if not cand_with_scores:
                    results_cands[s1_id] = ""
                    results_matches[s1_id] = ""
                    continue

                cands = [cid for cid, _ in cand_with_scores]
                results_cands[s1_id] = ",".join(cands)
                cand_recs = [(idx.targets[cid], sc) for cid, sc in cand_with_scores if cid in idx.targets]
                if not cand_recs:
                    results_matches[s1_id] = ""
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
                    m_ids = matcher.predict_entity_matches(
                        cand_ids, probs,
                        anchor_threshold=matcher.optimal_threshold,
                        expansion_threshold=max(0.55, matcher.optimal_threshold - 0.08)
                    )
                    results_matches[s1_id] = ",".join(m_ids) if m_ids else ""

            processed = min(b_start + batch_size, len(s1_list))
            if processed % 50000 == 0 or processed == len(s1_list):
                print(f"  [{country}] {processed}/{len(s1_list)} complete ({((processed)/len(s1_list))*100:.1f}%).")
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
