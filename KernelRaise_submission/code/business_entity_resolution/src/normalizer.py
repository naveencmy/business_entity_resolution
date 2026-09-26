"""
Normalizer for business names, addresses, and country-specific entities.
Handles:
- Transliteration of Indic scripts (Devanagari, Tamil, Gujarati, etc.) to Latin
- Legal suffix standardizing / stripping
- Token de-duplication (e.g. 'LLC LLC', 'Vidyalaya Vidyalaya')
- Domain / URL cleaning (e.g. 'heassociates.com' -> 'heassociates')
- Landmark and address normalization
- Open-set country preservation (US, India, France, etc.)
"""
import re
import unicodedata
from typing import Dict, Tuple, Optional, Set

# Base Devanagari phonetics table
# In Unicode, Brahmic scripts (Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam)
# share identical relative code point offsets modulo 0x0080 from Devanagari (0x0900).
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
    """Map any Indic script character (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) to Latin."""
    if 0x0900 <= code_point <= 0x0D7F:
        offset = code_point % 0x0080
        return INDIC_BASE_MAP.get(offset)
    return None

# Regex for common legal entity suffix terms (case-insensitive)
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
WHITESPACE_PATTERN = re.compile(r'\s+')

# Indian Postal PIN: 6-digit number
INDIA_PIN_PATTERN = re.compile(r'\b([1-9][0-9]{5})\b')
# US ZIP Code: 5-digit number
US_ZIP_PATTERN = re.compile(r'\b([0-9]{5})(?:-[0-9]{4})?\b')
# France Code Postal: 5-digit number starting with 01-95 or 97-98
FRANCE_CP_PATTERN = re.compile(r'\b(0[1-9]|[1-8][0-9]|9[0-5]|97|98)[0-9]{3}\b')

# US 2-letter States
US_STATES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
}


def transliterate_to_latin(text: str) -> str:
    """
    Transliterate non-Latin scripts (Devanagari, Tamil, Gujarati) into Latin approximations.
    """
    if not text:
        return ""
    
    # Quick check if non-ascii characters are present
    if text.isascii():
        return text
    
    # Translate known Indic code points
    res = []
    for ch in text:
        cp = ord(ch)
        indic = get_indic_latin(cp)
        if indic is not None:
            res.append(indic)
        else:
            # Fall back to unicode decomposition
            res.append(ch)
    
    combined = "".join(res)
    # Normalize unicode (NFKD) and strip accents (e.g. French accents: é -> e, à -> a)
    nfkd = unicodedata.normalize('NFKD', combined)
    ascii_bytes = nfkd.encode('ASCII', 'ignore')
    return ascii_bytes.decode('utf-8')


def clean_business_name(name: str) -> Tuple[str, str]:
    """
    Normalize business name.
    Returns:
        (cleaned_full_name, core_stem_name)
    """
    if not name or not isinstance(name, str):
        return "", ""
    
    # Handle DBA / Trading As
    # If "DBA:" or "doing business as" is present, extract both parts
    if " dba " in f" {name.lower()} " or " dba: " in f" {name.lower()} ":
        name = re.sub(r'\b(?:dba|doing business as)[:\s]+', ' ', name, flags=re.IGNORECASE)
    
    # Step 1: Transliterate multi-lingual script & normalize accents
    text = transliterate_to_latin(name)
    
    # Step 2: Extract brand from embedded URLs (e.g. vinaytele.com -> vinaytele, www.abc.com -> abc)
    text = URL_PATTERN.sub(r' \1 ', text)
    
    # Step 3: Replace '&' with 'and'
    text = text.replace('&', ' and ')
    
    # Step 4: Remove punctuation
    text = PUNCT_PATTERN.sub(' ', text).lower()
    
    # Step 5: Remove redundant consecutive tokens (e.g., "llc llc", "vidyalaya vidyalaya")
    tokens = text.split()
    deduped_tokens = []
    for tok in tokens:
        # Strip leading zeros on numeric tokens
        clean_tok = re.sub(r'^0+([0-9])', r'\1', tok) if tok.isdigit() else tok
        if not deduped_tokens or clean_tok != deduped_tokens[-1]:
            deduped_tokens.append(clean_tok)
    cleaned_full = " ".join(deduped_tokens)
    
    # Step 6: Create core stem by stripping legal suffixes
    stem = LEGAL_PATTERN.sub('', cleaned_full)
    stem_tokens = stem.split()
    stem = " ".join(stem_tokens)
    
    if not stem and cleaned_full:
        stem = cleaned_full
        
    return cleaned_full, stem


def clean_address(address: str, country: str) -> Dict[str, str]:
    """
    Extract postal code, primary locality, and clean address tokens.
    Normalizes zero-padded numbers (e.g. 00478 -> 478, AF-0684 -> AF-684).
    """
    if not address or not isinstance(address, str):
        return {"postal_code": "", "locality": "", "clean_tokens": ""}
    
    # Transliterate accents/scripts
    text = transliterate_to_latin(address)
    
    # Country-specific postal code extraction
    postal_code = ""
    country_upper = (country or "").strip().upper()
    
    if country_upper == "INDIA":
        m = INDIA_PIN_PATTERN.search(text)
        if m:
            postal_code = m.group(0)
    elif country_upper == "US":
        m = US_ZIP_PATTERN.search(text)
        if m:
            postal_code = m.group(0)
    elif country_upper == "FRANCE":
        m = FRANCE_CP_PATTERN.search(text)
        if m:
            postal_code = m.group(0)
    else:
        # Fallback generic 5-6 digit finder
        m = re.search(r'\b([0-9]{5,6})\b', text)
        if m:
            postal_code = m.group(0)
            
    # Extract locality / state indicator for US
    locality = ""
    if country_upper == "US":
        words = set(re.findall(r'\b[A-Za-z]{2}\b', text.upper()))
        found_states = words.intersection(US_STATES)
        if found_states:
            locality = sorted(list(found_states))[0]
            
    # Normalize zero-padded numbers before punctuation stripping (e.g. AF-0684 -> AF-684, 00127/4 -> 127/4)
    text = re.sub(r'\b0+([1-9][0-9]*)\b', r'\1', text)
    text = re.sub(r'([A-Za-z]+)-0+([1-9][0-9]*)', r'\1-\2', text)
    
    # Remove punctuation and normalize whitespace
    clean_addr = PUNCT_PATTERN.sub(' ', text).lower()
    tokens = [tok for tok in clean_addr.split() if len(tok) > 1]
    
    # Remove generic address stopwords
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
