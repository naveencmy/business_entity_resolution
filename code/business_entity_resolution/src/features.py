"""
Pairwise feature extraction for Entity Matching.
Country-agnostic string, token, phonetic, and address similarity metrics.
Handles open-set countries (US, India, France) uniformly.
"""
import math
import collections
from typing import Dict, List, Set, Any, Tuple

def char_ngrams(s: str, n: int = 3) -> Set[str]:
    if not s or len(s) < n:
        return {s} if s else set()
    padded = f"^{s}$"
    return {padded[i:i+n] for i in range(len(padded) - n + 1)}

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0

def containment_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    min_len = min(len(set_a), len(set_b))
    if min_len == 0:
        return 0.0
    return len(set_a.intersection(set_b)) / min_len

def levenshtein_similarity(s1: str, s2: str) -> float:
    """Fast bounded Levenshtein ratio."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    if abs(len1 - len2) > max(len1, len2) * 0.7:
        return 0.0

    # Truncate if exceedingly long to keep feature extraction fast
    s1_trunc, s2_trunc = s1[:60], s2[:60]
    m, n = len(s1_trunc), len(s2_trunc)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if s1_trunc[i - 1] == s2_trunc[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    dist = dp[n]
    max_l = max(m, n)
    return max(0.0, 1.0 - (dist / max_l))

def extract_pairwise_features(s1_rec: Dict[str, Any], cand_rec: Dict[str, Any], blocking_score: float = 0.0) -> List[float]:
    """
    Extract a compact vector of normalized similarity features between s1_rec and cand_rec.
    Features:
    0: exact_stem_match (0/1)
    1: exact_full_name_match (0/1)
    2: name_char_3gram_jaccard [0, 1]
    3: name_token_jaccard [0, 1]
    4: name_token_containment [0, 1]
    5: name_levenshtein_sim [0, 1]
    6: first_word_match (0/1)
    7: name_length_ratio [0, 1]
    8: postal_code_match (-1: both missing, 0: mismatch, 1: match)
    9: locality_match (0/1)
    10: addr_token_jaccard [0, 1]
    11: addr_token_containment [0, 1]
    12: addr_number_overlap [0, 1]
    13: name_x_addr_composite [0, 1]
    14: name_or_addr_max [0, 1]
    15: is_source_3 (0/1)
    16: normalized_blocking_score
    """
    # Name stems and full names
    stem1 = s1_rec.get("name_stem", "").strip().lower()
    stem2 = cand_rec.get("name_stem", "").strip().lower()
    full1 = s1_rec.get("name_full", "").strip().lower()
    full2 = cand_rec.get("name_full", "").strip().lower()

    exact_stem = 1.0 if (stem1 and stem1 == stem2) else 0.0
    exact_full = 1.0 if (full1 and full1 == full2) else 0.0

    # Character 3-grams
    grams1 = char_ngrams(stem1, 3)
    grams2 = char_ngrams(stem2, 3)
    name_gram_jaccard = jaccard_similarity(grams1, grams2)

    # Word tokens
    tokens1 = set(stem1.split())
    tokens2 = set(stem2.split())
    name_token_jaccard = jaccard_similarity(tokens1, tokens2)
    name_token_containment = containment_similarity(tokens1, tokens2)

    # Edit distance
    name_lev = levenshtein_similarity(stem1, stem2)

    # First word match
    w1 = stem1.split()[0] if stem1 else ""
    w2 = stem2.split()[0] if stem2 else ""
    first_word_match = 1.0 if (w1 and w1 == w2) else 0.0

    # Length ratio
    len1, len2 = len(stem1), len(stem2)
    max_len = max(len1, len2)
    name_len_ratio = (min(len1, len2) / max_len) if max_len > 0 else 1.0

    # Postal code
    p1 = s1_rec.get("postal_code", "").strip()
    p2 = cand_rec.get("postal_code", "").strip()
    if not p1 and not p2:
        postal_match = -0.5
    elif p1 and p2:
        postal_match = 1.0 if p1 == p2 else -1.0
    else:
        postal_match = 0.0

    # Locality / State
    loc1 = s1_rec.get("locality", "").strip().upper()
    loc2 = cand_rec.get("locality", "").strip().upper()
    locality_match = 1.0 if (loc1 and loc2 and loc1 == loc2) else 0.0

    # Address tokens
    addr1 = s1_rec.get("clean_addr", "").strip().lower()
    addr2 = cand_rec.get("clean_addr", "").strip().lower()
    addr_toks1 = set(addr1.split())
    addr_toks2 = set(addr2.split())
    addr_jaccard = jaccard_similarity(addr_toks1, addr_toks2)
    addr_containment = containment_similarity(addr_toks1, addr_toks2)

    # Numeric address tokens (e.g. house/street numbers)
    nums1 = {t for t in addr_toks1 if any(c.isdigit() for c in t)}
    nums2 = {t for t in addr_toks2 if any(c.isdigit() for c in t)}
    addr_num_overlap = jaccard_similarity(nums1, nums2) if (nums1 or nums2) else 0.5

    # Composite metrics
    name_x_addr = name_gram_jaccard * max(addr_jaccard, addr_containment)
    name_or_addr_max = max(name_gram_jaccard, max(addr_jaccard, addr_containment))

    # Source prefix (S2 vs S3)
    cand_id = cand_rec.get("entity_id", "")
    is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    norm_blocking_sc = min(1.0, blocking_score / 20.0)

    return [
        exact_stem,
        exact_full,
        name_gram_jaccard,
        name_token_jaccard,
        name_token_containment,
        name_lev,
        first_word_match,
        name_len_ratio,
        postal_match,
        locality_match,
        addr_jaccard,
        addr_containment,
        addr_num_overlap,
        name_x_addr,
        name_or_addr_max,
        is_s3,
        norm_blocking_sc
    ]

FEATURE_NAMES = [
    "exact_stem",
    "exact_full",
    "name_gram_jaccard",
    "name_token_jaccard",
    "name_token_containment",
    "name_lev",
    "first_word_match",
    "name_len_ratio",
    "postal_match",
    "locality_match",
    "addr_jaccard",
    "addr_containment",
    "addr_num_overlap",
    "name_x_addr",
    "name_or_addr_max",
    "is_s3",
    "norm_blocking_sc"
]
