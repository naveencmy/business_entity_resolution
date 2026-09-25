## 📋 Description of Changes
<!-- Provide a brief summary of the changes made and the motivation behind them. -->

## 🎯 Target Impact
- [ ] Candidate Generation (Phase 2 Blocking / Recall improvement)
- [ ] Feature Engineering (Phase 3 Pairwise similarities)
- [ ] Model Training / Threshold Tuning (F_0.5 optimization)
- [ ] Normalization / Multilingual Transliteration
- [ ] Documentation / Pipeline Reproducibility

## 🧪 Verification & Benchmark Results
<!-- Describe what tests were run and include validation metrics if applicable -->
- [ ] Candidate Blocking Recall: `___%`
- [ ] Validation Macro $F_{0.5}$: `___`
- [ ] Passed `validate_submission.py` locally:
  ```bash
  python Deputy_pipe/Datasets/student_resource/utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir Deputy_pipe/Datasets/student_resource/dataset/test
  ```

## 🔍 Checklist
- [ ] Code follows project coding standards (strict tab-separation, country-isolation).
- [ ] No external lookups or unauthorized API calls (Academic Integrity).
- [ ] Documentation updated in `Documentation_template.md` / `README.md`.
- [ ] All commits follow Conventional Commits format (`feat:`, `fix:`, `docs:`, etc.).
