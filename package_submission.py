#!/usr/bin/env python3
"""
Team Submission Packaging Utility for ML Challenge 2026: Business Entity Resolution.
Bundles outputs, code, and documentation into a compliant team_submission.zip.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

def find_file(filename: str, search_roots: list) -> Path:
    for root in search_roots:
        p = Path(root)
        if not p.exists():
            continue
        # Check direct path
        direct = p / filename
        if direct.exists() and direct.is_file():
            return direct
        # Recursive search
        matches = list(p.glob(f"**/{filename}"))
        if matches:
            return matches[0]
    return None

def build_package():
    print("=" * 65)
    print("📦 BUILDING COMPLETE OFFICIAL TEAM SUBMISSION PACKAGE")
    print("=" * 65)

    base_dir = Path(".").resolve()
    staging_dir = base_dir / "submission_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    search_roots = [
        Path("/kaggle/working"),
        Path("/kaggle/working/output"),
        Path("./output"),
        Path("../output"),
        base_dir
    ]

    # 1. Locate Output TSVs
    match_tsv = find_file("matching_results.tsv", search_roots)
    cand_tsv = find_file("candidate_pairs.tsv", search_roots)

    if not match_tsv:
        print("❌ Error: matching_results.tsv could not be found.")
        print("Please ensure kaggle_pipeline.py has generated output files.")
        return 1

    if not cand_tsv:
        print("❌ Error: candidate_pairs.tsv could not be found.")
        return 1

    stage_output = staging_dir / "output"
    stage_output.mkdir(parents=True, exist_ok=True)
    shutil.copy(match_tsv, stage_output / "matching_results.tsv")
    shutil.copy(cand_tsv, stage_output / "candidate_pairs.tsv")
    print(f"✅ Added output/matching_results.tsv ({match_tsv.stat().st_size / 1e6:.1f} MB)")
    print(f"✅ Added output/candidate_pairs.tsv  ({cand_tsv.stat().st_size / 1e6:.1f} MB)")

    # 2. Bundle Code
    code_dir = find_file("kaggle_pipeline.py", search_roots)
    if code_dir:
        src_code = code_dir.parent.parent
        stage_code = staging_dir / "code"
        shutil.copytree(src_code, stage_code, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git*"))
        print(f"✅ Added code/ directory from {src_code}")
    elif (base_dir / "code").exists():
        shutil.copytree(base_dir / "code", staging_dir / "code", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git*"))
        print(f"✅ Added code/ directory from {base_dir / 'code'}")

    # 3. Bundle Documentation & Methodology
    doc_template = find_file("Documentation_template.md", [base_dir, Path("/kaggle/working/business_entity_resolution")])
    if doc_template:
        shutil.copy(doc_template, staging_dir / "Documentation_template.md")
        print("✅ Added Documentation_template.md")

    if (base_dir / "Documentation").exists():
        shutil.copytree(base_dir / "Documentation", staging_dir / "Documentation", dirs_exist_ok=True)
        print("✅ Added Documentation/ architecture & strategy folder")

    for f_name in ["README.md", "LICENSE"]:
        doc_f = find_file(f_name, [base_dir, Path("/kaggle/working/business_entity_resolution")])
        if doc_f:
            shutil.copy(doc_f, staging_dir / f_name)
            print(f"✅ Added {f_name}")

    # 4. Create ZIP Archive
    zip_dest = base_dir / "team_submission"
    archive_path = shutil.make_archive(str(zip_dest), 'zip', staging_dir)
    shutil.rmtree(staging_dir)

    print("-" * 65)
    print("🔍 VERIFYING SUBMISSION ARCHIVE CONTENTS:")
    print("-" * 65)
    with zipfile.ZipFile(archive_path, 'r') as z:
        namelist = z.namelist()
        for name in sorted(namelist):
            info = z.getinfo(name)
            if not info.is_dir():
                print(f"  📄 {name:<45} ({info.file_size / 1024:.1f} KB)")

    print("=" * 65)
    print(f"🎉 FINAL SUBMISSION ZIP READY: {archive_path}")
    print(f"📊 Total Size: {os.path.getsize(archive_path) / (1024*1024):.2f} MB")
    print("=" * 65)
    return 0

if __name__ == "__main__":
    sys.exit(build_package())
