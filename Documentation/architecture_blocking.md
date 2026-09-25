# Phase 2 Architecture: Scalable Multi-Tier Blocking & Candidate Generation
## Business Entity Resolution Challenge (Target: F_0.5 ≥ 0.998 - 0.9998)

---

### 1. Mathematical Formulation & Scale Constraints

Let:
- $\mathcal{S}_1$ be the deduplicated reference source with $|\mathcal{S}_1| \approx 2.2 \times 10^6$ records (train) and $1.73 \times 10^6$ records (test).
- $\mathcal{S}_2$ and $\mathcal{S}_3$ be noisy candidate sources with $|\mathcal{S}_2| \approx 5.0 \times 10^6$ and $|\mathcal{S}_3| \approx 5.0 \times 10^6$ records.
- Total Cartesian product space $|\mathcal{S}_1 \times (\mathcal{S}_2 \cup \mathcal{S}_3)| \approx 1.73 \times 10^6 \times 10^7 \approx 1.73 \times 10^{13}$ pairs.

Exhaustive pairwise scoring over $1.73 \times 10^{13}$ pairs is computationally intractable. 

#### Metric Objective:
Macro $F_{0.5}$ over all $e \in \mathcal{S}_1$:
$$F_{0.5}(e) = \frac{(1 + 0.5^2) \cdot P(e) \cdot R(e)}{0.5^2 \cdot P(e) + R(e)} = \frac{1.25 \cdot P(e) \cdot R(e)}{0.25 \cdot P(e) + R(e)}$$
Singletons (entities where true match set $\mathcal{M}(e) = \emptyset$):
$$\widehat{\mathcal{M}}(e) = \emptyset \implies F_{0.5}(e) = 1.0; \quad \widehat{\mathcal{M}}(e) \neq \emptyset \implies F_{0.5}(e) = 0.0$$

#### The 0.998+ Imperative for Blocking:
1. **Recall Ceiling ($R_{\text{block}} \ge 0.9999$):** If a true match $(e_1, e_{2/3}) \notin \mathcal{C}$, it is permanently lost. To reach overall $F_{0.5} \ge 0.998$, the recall of candidate generation must approach 99.99%.
2. **Extreme Candidate Concentration (Reduction Ratio $\ge 99.999\%$):** The average candidate size per $e_1$ must be kept small ($K \approx 1 \text{ to } 10$ candidates per entity on average), eliminating spurious noise that causes false positive merges on singletons.
3. **Open-Set Country Partition Guarantee:** Match candidates are strictly partitioned by normalized country. Records with `country == 'US'`, `country == 'India'`, or `country == 'France'` are grouped independently. Cross-country comparisons are prohibited ($R_{\text{cross-country}} = 0$).

---

### 2. Multi-Tier Hierarchical Blocking Architecture

To balance ultra-high recall (≥99.99%) and minimal candidate volume, we deploy a **3-Tier Cascaded Inverted Index & Blocking Engine**:

```
[ Raw Records (S1, S2, S3) ]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│  Tier 0: Country Partition & Preprocessing Normalizer   │
│  - Open-set country isolation (US, India, France, etc) │
│  - Transliteration / Unicode normalization             │
│  - Suffix standardizer (Pvt Ltd, LLC, Corp, SARL)     │
│  - Clean token & n-gram generation                     │
└────────────────────────────────────────────────────────┘
           │
     ┌─────┴─────────────────────────────────────┐
     ▼                                           ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│ Tier 1: High-Precision        │   │ Tier 2: Fault-Tolerant        │
│ Deterministic Keys            │   │ Multi-Key Inverted Index      │
│ - Exact cleaned stem + city   │   │ - Name 3-gram minhash / IDF   │
│ - Phonetic Double Metaphone   │   │ - Address token overlap       │
│ - Postal/PIN code + stem      │   │ - Top-K BM25 / Cosine Sim     │
└───────────────────────────────┘   └───────────────────────────────┘
     │                                           │
     └─────────────────────┬─────────────────────┘
                           │ Union & Fast Pruning
                           ▼
┌────────────────────────────────────────────────────────┐
│ Tier 3: Adaptive Candidate Ranker & Size Limiter       │
│ - Union candidate sets per S1 entity                   │
│ - Max candidate cap: Top-N candidates (e.g. N ≤ 15)    │
│ - Outputs candidate_pairs.tsv (Format Compliant)       │
└────────────────────────────────────────────────────────┘
```

#### Tier 1: Deterministic Multi-Key Blocking (High Precision, High Speed)
1. **Key 1 (Normalized Name Stem + Primary Locality):**
   - Name normalized: lowercased, punctuation stripped, legal entity tokens stripped (`pvt`, `ltd`, `private`, `limited`, `llc`, `inc`, `corp`, `sarl`, `sasu`, `gmbh`, `society`, `trust`), whitespace compacted.
   - Primary Locality: Extracted postal code / PIN code if available, else canonical state/city abbreviation.
2. **Key 2 (Phonetic Encoding + First Address Token):**
   - Double Metaphone / Soundex of the dominant name tokens.
3. **Key 3 (Exact Substring / Domain Match):**
   - For records containing URL/domain noise (`www.xyz.com`), extract the root domain (`xyz`) as an exact blocking anchor.

#### Tier 2: High-Recall Approximate Inverted Index (IDF-Weighted Token / N-Gram)
For pairs with heavy typographical errors, transliterations (e.g. Devanagari/Tamil to Latin or vice versa), or missing address tokens:
1. **Character 3-grams of cleaned business name** filtered to keep medium-to-high IDF grams.
2. **Inverted Index lookup with BM25 / TF-IDF Cosine scoring:**
   - Pre-filter postings lists by dropping ultra-high frequency stopwords (e.g. `enterprises`, `services`, `solutions`, `associates`).
   - Sparse matrix multiplication using batched CSR matrices to achieve high throughput across millions of rows with low RAM overhead.
3. **Candidate Selection:** Retrieve top-$K$ candidates per $S_1$ entity ($K \le 10$).

#### Tier 3: Candidate Union & Open-Set Validation
- For each $e_1 \in \mathcal{S}_1$:
  $$\mathcal{C}(e_1) = \left( \mathcal{C}_{\text{Tier 1}}(e_1) \cup \mathcal{C}_{\text{Tier 2}}(e_1) \right) \cap \{ e \in \mathcal{S}_2 \cup \mathcal{S}_3 \mid \text{country}(e) = \text{country}(e_1) \}$$
- Candidates are strictly deduplicated and capped to avoid memory explosion.
- Ensures all singletons without plausible candidate matches are kept empty.

---

### 3. Handling Multi-Lingual & Regional Noise Patterns

#### A. Indian Entity Nuances:
- **Transliteration:** Indian scripts (Devanagari, Tamil, Telugu, Gujarati, Bengali, etc.) appear in $S_2$ and $S_3$ (e.g. `राम मार्केटिंग प्राइवेट लिमिटेड`, `குளோபल...`, `ગુજરાત`).
- **Solution:** Apply Unicode transliteration to Latin (e.g., using `unicodedata` / multi-lingual romanization mapping) during normalization so that script discrepancies map directly to the corresponding Latin phonetic/English stems in $S_1$.
- **Landmark-based Addresses:** Addresses containing "Near SBI ATM", "Opposite Railway Station", "Behind Petrol Pump". The normalizer extracts distinct spatial tokens (PIN codes, cities, states) while downweighting generic landmark qualifiers.

#### B. French Entity Nuances (Open-Set Test Country):
- **Legal Form Variations:** `SARL`, `SASU`, `EURL`, `SCI`, `Fils`, `Association`, `École`.
- **Address Components:** `Rue`, `Avenue`, `Boulevard`, `Allée`, `Place`, `Chemin`, French department/region names (e.g. `Nouvelle-Aquitaine`, `Hauts-de-France`).
- **Solution:** Country-agnostic tokenizer that does not assume US state codes or 6-digit Indian PIN codes, dynamically adapting address token parsing per country.

#### C. US Entity Nuances:
- **State Codes:** Standard 2-letter postal abbreviations (`NC`, `OK`, `AZ`, `MD`, `TX`, `VA`, etc.) appearing either prefix or suffix in the address.
- **Suite/Unit Formats:** `Unit 11`, `Apt G`, `Suite 300`, `Building 3030`.

---

### 4. Memory and Scalability Strategy (Streaming & Chunking)

Given the dataset size ($> 12 \text{ million}$ records across train/test):
1. **Per-Country Processing:** Process records partition by partition (e.g., `India`, `US`, `France`). This bounds memory footprint to fit easily in standard workstation RAM.
2. **Chunked Inverted Index Construction:** Inverted indices are constructed per country partition.
3. **Parallel Fast Caching:** Intermediate features and candidate lists are streamed directly to disk in `.tsv` format to avoid maintaining massive intermediate Python objects in memory.

---

### 5. Verification & Quality Gates for Phase 2

1. **Recall Check on Train Set:** Evaluate candidate generation against `train_ground_truth.tsv`. Target: Recall $\ge 99.95\%$.
2. **Reduction Ratio:**
   $$\text{RR} = 1 - \frac{\sum_{e_1} |\mathcal{C}(e_1)|}{|\mathcal{S}_1| \times (|\mathcal{S}_2| + |\mathcal{S}_3|)} > 99.999\%$$
3. **Format & Integrity Check:** Validate that `candidate_pairs.tsv` strictly conforms to the challenge validator `utils/validate_submission.py`.
