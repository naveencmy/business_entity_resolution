#!/usr/bin/env python3
"""
Create Fresh, Canonical Competition Submission Package.
Produces:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches
│   └── candidate_pairs.tsv         # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code
│       ├── README.md               # reproduction guide
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # filled methodology write-up
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def find_file(filename: str, search_roots: list) -> Path:
    for root in search_roots:
        p = Path(root)
        if not p.exists():
            continue
        direct = p / filename
        if direct.exists() and direct.is_file():
            return direct
        matches = list(p.glob(f"**/{filename}"))
        if matches:
            return matches[0]
    return None

def create_fresh_submission(team_name: str = "alpha_resolvers"):
    print("=" * 65)
    print(f"[+] CREATING FRESH SUBMISSION FOLDER: {team_name}_submission")
    print("=" * 65)

    base_dir = Path(".").resolve()
    target_dir = base_dir / f"{team_name}_submission"
    
    # 1. Clean & recreate destination folder
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. Setup output/ folder
    out_dir = target_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_output_paths = [
        Path("/kaggle/working/output"),
        Path("/kaggle/working/business_entity_resolution/output"),
        Path("/kaggle/working"),
        base_dir / "output",
        base_dir / "code" / "business_entity_resolution" / "output",
    ]

    match_tsv, cand_tsv = None, None
    for p in candidate_output_paths:
        m = p / "matching_results.tsv"
        c = p / "candidate_pairs.tsv"
        if m.exists() and c.exists():
            match_tsv, cand_tsv = m, c
            break

    if match_tsv and cand_tsv:
        shutil.copy(match_tsv, out_dir / "matching_results.tsv")
        shutil.copy(cand_tsv, out_dir / "candidate_pairs.tsv")
        print(f"  [OK] Copied output/matching_results.tsv ({match_tsv.stat().st_size / 1e6:.1f} MB)")
        print(f"  [OK] Copied output/candidate_pairs.tsv  ({cand_tsv.stat().st_size / 1e6:.1f} MB)")
    else:
        print("  [INFO] Output TSVs not found locally yet. Created placeholder files.")
        print("         (When run in Kaggle, the real generated TSVs will be automatically copied).")
        (out_dir / "matching_results.tsv").touch()
        (out_dir / "candidate_pairs.tsv").touch()

    # 3. Setup code/business_entity_resolution/
    code_dest = target_dir / "code" / "business_entity_resolution"
    code_dest.mkdir(parents=True, exist_ok=True)

    src_dir = base_dir / "code" / "business_entity_resolution" / "src"
    if src_dir.exists():
        shutil.copytree(src_dir, code_dest / "src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        print(f"  [OK] Copied code/business_entity_resolution/src/ ({len(list(src_dir.glob('*.py')))} Python modules)")

    readme_file = base_dir / "code" / "business_entity_resolution" / "README.md"
    if readme_file.exists():
        shutil.copy(readme_file, code_dest / "README.md")
        print("  [OK] Copied code/business_entity_resolution/README.md")

    req_file = base_dir / "code" / "business_entity_resolution" / "requirements.txt"
    if req_file.exists():
        shutil.copy(req_file, code_dest / "requirements.txt")
        print("  [OK] Copied code/business_entity_resolution/requirements.txt")

    kaggle_pipe = base_dir / "code" / "business_entity_resolution" / "kaggle_pipeline.py"
    if kaggle_pipe.exists():
        shutil.copy(kaggle_pipe, code_dest / "kaggle_pipeline.py")
        print("  [OK] Copied code/business_entity_resolution/kaggle_pipeline.py")

    # 4. Copy Documentation_template.md
    doc_file = base_dir / "Documentation_template.md"
    if doc_file.exists():
        shutil.copy(doc_file, target_dir / "Documentation_template.md")
        print("  [OK] Copied Documentation_template.md (Filled official methodology write-up)")

    # 5. Build ZIP archive
    zip_path = base_dir / f"{team_name}_submission"
    archive_file = shutil.make_archive(str(zip_path), 'zip', target_dir)

    print("-" * 65)
    print("ARCHIVE FILE STRUCTURE VERIFICATION:")
    print("-" * 65)
    with zipfile.ZipFile(archive_file, 'r') as z:
        for entry in sorted(z.namelist()):
            info = z.getinfo(entry)
            if not info.is_dir():
                print(f"  - {entry:<55} ({info.file_size / 1024:.1f} KB)")

    print("=" * 65)
    print(f"[+] CREATED FRESH: {archive_file}")
    print(f"[+] Total Size:   {os.path.getsize(archive_file) / (1024*1024):.2f} MB")
    print("=" * 65)
    return archive_file

if __name__ == "__main__":
    create_fresh_submission()
