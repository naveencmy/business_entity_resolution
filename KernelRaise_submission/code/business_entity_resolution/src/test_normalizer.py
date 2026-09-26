"""
Unit test to verify the normalization functionality on real challenge examples.
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
from normalizer import clean_business_name, clean_address

def test_samples():
    cases = [
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "Devanagari"),
        ("SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com", "URL and token repetition"),
        ("குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்", "Tamil"),
        ("Olszewski Holding Company LLC LLC", "Duplicate LLC"),
        ("SCI Ptit Àmicale", "French Accents and SCI"),
        ("Marina Ecole France Sarl", "French SARL"),
    ]
    
    print("Testing Business Name Normalization:")
    for name, desc in cases:
        full, stem = clean_business_name(name)
        print(f"[{desc}]")
        print(f"  Raw:  {name}")
        print(f"  Full: {full}")
        print(f"  Stem: {stem}\n")

    addr_cases = [
        ("1795 Westchester Drive, High Point, NC", "US"),
        ("Plot No.53, Sainikpuri, Hyderabad 500094, Telangana", "India"),
        ("175 Boulevard du Président Franklin Roosevelt, 33000 Bordeaux", "France"),
        ("Near Fortis Hospital, Mulund Link Road, Mumbai", "India"),
    ]
    
    print("Testing Address Normalization:")
    for addr, country in addr_cases:
        res = clean_address(addr, country)
        print(f"[{country}] {addr}")
        print(f"  Postal: {res['postal_code']}, Locality: {res['locality']}")
        print(f"  Tokens: {res['clean_tokens']}\n")

if __name__ == "__main__":
    test_samples()
