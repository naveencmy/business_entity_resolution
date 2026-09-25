# Engineering Strategy: Tight Blocking, Precision-Driven Scoring & Singleton Gating

## 1. Empirical Ground Truth Profiling

Analysis of all **2,206,821 Source 1 entities** in `train_ground_truth.tsv` revealed the exact underlying distribution:

| Cluster Size (Matches) | Count  | Percentage | Cumulative % |
| :--- | :--- | :--- | :--- |
| **0 matches (Singletons)** | **123,247** | **5.58%**   | **5.58%** |
| 1 match   | 119,157 | 5.40%   | 10.98% |
| 2 matches | 375,212 | 17.00%  | 27.98% |
| 3 matches | 530,841 | 24.05%  | 52.03% |
| 4 matches | 484,115 | 21.94%  | 73.97% |
| 5 matches | 321,957 |  14.59% | 88.56% |
| 6 matches | 164,868 |   7.47% | 96.03% |
| 7 matches | 63,968  |   2.90% | 98.93% |
| 8 matches | 18,680  |   0.85% | 99.78% |
| 9 matches | 4,205   |   0.19% | 99.97% |
| $\ge 10$ matches | 571   |   0.03% | 100.00% |

### Key Takeaways:
1. **The $K \le 8$ Empirical Ceiling:** Over **99.78%** of all entities have 8 or fewer matches. Capping candidate retrieval at $K = 8$ eliminates candidate bloat while retaining almost 100% of genuine match clusters.
2. **5.58% Singletons:** Approximately **97,000 entities in the test set** are singletons with zero true matches.

---

## 2. Pillar I: Tight Candidate Generation

Because `candidate_pairs.tsv` is audited for reduction ratio and blocking efficiency, we avoid naive high-recall dumping:
- **Minimum Blocking Score Filter (`min_score = 3.0`):** Inverted index character 3-gram hits that do not cross the minimum TF-IDF overlap threshold are discarded immediately.
- **Immediate Singleton Filtering:** For true singletons with no genuine matches in $S_2/S_3$, the candidate set resolves to completely **EMPTY** ($|\mathcal{C}(e)| = 0$). This achieves two critical objectives:
  1. Keeps average candidate set size minimal ($\approx 2.5 - 4.0$ candidates per entity across the dataset).
  2. Guarantees that the matching model cannot make false positive errors on these entities, securing an automatic $1.0$ score.

---

## 3. Pillar II: Precision-Driven Scoring ($F_{0.5}$ Metric Sensitivity)

The competition evaluation metric is the macro-averaged $F_{0.5}$ score:

$$F_{0.5} = \frac{(1 + 0.5^2) \cdot P \cdot R}{0.5^2 \cdot P + R} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

### Mathematical Sensitivity Comparison:
Consider an entity with 3 true matches:

| Decision Case | Predicted | True Positives | False Positives | Missed | Precision | Recall | Entity $F_{0.5}$ | Delta vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ideal** | 3 matches | 3 | 0 | 0 | 1.00 | 1.00 | **1.000** | — |
| **Conservative (Miss 1 match)** | 2 matches | 2 | 0 | 1 | 1.00 | 0.67 | **0.909** | **-0.091** |
| **Aggressive (1 false merge)** | 4 matches | 3 | 1 | 0 | 0.75 | 1.00 | **0.789** | **-0.211** |
| **Aggressive on Singleton** | 1 match | 0 | 1 | 0 | 0.00 | 1.00 | **0.000** | **-1.000** |

**Conclusion:** 
A false merge hurts **2.3× more** than a missed match on an active entity, and causes an immediate **100% failure (0.0)** on a singleton. Therefore, our model strictly rejects aggressive thresholding.

---

## 4. Pillar III: Intentional Singleton Verification Gate

To safeguard against false merges on the ~97,000 test singletons, we implement a **Dual-Stage Anchor + Expansion Gate**:

```
                       [ Candidate Set C(e1) ]
                                  │
                                  ▼
               Is max candidate probability P >= 0.72?
                                 / \
                                /   \
                        YES    /     \   NO
                              /       \
                             ▼         ▼
                 [ Anchor Confirmed ]  [ Confirmed Singleton ]
                 Include all c with    Return EMPTY []
                 P(c) >= 0.60          (Score: 1.0)
```

1. **Stage 1 (Anchor Verification):**
   - Condition: $\max_{c \in \mathcal{C}(e_1)} P(c) \ge \tau_{\text{anchor}} \approx 0.72$.
   - If not met, the model refuses to hypothesize a match. The entity is confirmed as a singleton and predicted as `""` (empty string).
2. **Stage 2 (Cluster Expansion):**
   - Once an anchor match is verified with high confidence, secondary cluster members are admitted if $P(c) \ge \tau_{\text{expansion}} \approx 0.60$.
   - This captures subtle address/name variants without exposing singletons to false positive risk.
