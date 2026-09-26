"""
Quick check of test entity distribution by country.
"""
import csv
from collections import Counter
import config

counts = Counter()
with open(config.TEST_S1, "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        if row and len(row) >= 4:
            counts[row[3].strip()] += 1

print("Test S1 Counts by Country:")
for c, cnt in counts.most_common():
    print(f"  {c}: {cnt}")
