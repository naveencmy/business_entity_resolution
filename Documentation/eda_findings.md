# Empirical Exploratory Data Analysis (EDA) & Ground Truth Diagnostics

## 1. Ground Truth Triangulation Findings

Inspecting true positive pairs across `train_source1.tsv`, `train_source2.tsv`, and `train_source3.tsv` revealed three fundamental match archetypes:

### Archetype A: Name-Dominant Matches (Missing / Sparse Address)
- **Example:**
  - $S_1$: `Maure Williams Colombier Inc` (Address: `85 Wayne Avenue, Ticonderoga, NY`)
  - $S_2$: `Maure Wilblims Colombier Inc` (Address: `""` - completely empty)
- **Implication:** The system cannot require an address match to link entities. When the business name is distinct and rare, it must be retrievable and scoreable on name similarity alone.

### Archetype B: Address-Dominant Matches (DBA / Alternate Trade Brand)
- **Example:**
  - $S_1$: `Maure Williams Colombier Inc` (Address: `85 Wayne Avenue, Ticonderoga, NY`)
  - $S_3$: `Dréxkor` (Address: `85 Wanye Avenue, Ticonderoga Townshiip, New York`)
- **Implication:** `Dréxkor` is a trade name (DBA) sharing zero lexical overlap with `Maure Williams Colombier`. However, the address (`85 Wayne Avenue, Ticonderoga, NY` vs `85 Wanye Avenue, Ticonderoga Townshiip, New York`) is unique. Address-only inverted index blocking is mandatory to capture these pairs.

### Archetype C: Multilingual Script & Transliteration Discrepancies
- **Example:**
  - $S_1$: `Raj Investments LLP`, Chennai, Tamil Nadu
  - $S_2$: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` (Tamil script transliterated to `raaj investments elelpi`)
  - $S_1$: `Ss Food Private Limited`, Ghaziabad, Uttar Pradesh
  - $S_2$: `एसएस फूड प्राइवेट लिमिटेड` (Hindi script transliterated to `eses phood praaivet limited`)
- **Implication:** Script transliteration converts Indic scripts into standard Latin characters before indexing, enabling cross-script candidate generation.

### Archetype D: Typographical & Legal Suffix Noise
- **Example:**
  - $S_1$: `Payne Enterprises`, `3315 Fremont Street, Peoria, IL`
  - $S_2$: `PAYNE-ENRTPRMISES`, `3315 FREMONT SAINT, PEORIA, IL`
  - $S_2$: `Payne Enterpires`, `3315 FREMONT ST, PEORIA, IL`
- **Implication:** Character 3-gram indexing with sub-string Jaccard/Levenshtein similarity handles internal character swaps effortlessly.

---

## 2. Blocking Architecture Implications for 0.998+ F_0.5 Target

To achieve $F_{0.5} \ge 0.998$, candidate recall must be $R \ge 99.95\%$, and candidate set size per entity must remain bounded ($K \le 10-15$).
We deploy a **Dual-Inverted Index (Name + Address)** partitioned by country:
1. **Name Gram Index:** TF-IDF weighted character 3-grams of the business name.
2. **Address Token Index:** Inverted index of address tokens combining house/street number and postal code/city.
3. **Exact Key Index:** Normalized stem + postal/city.
4. **Union & Re-ranking:** A candidate is generated if it matches either the Name Index, the Address Index, or an Exact Key, strictly constrained within the same country partition.
