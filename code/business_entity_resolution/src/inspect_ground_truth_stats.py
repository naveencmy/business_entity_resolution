"""
Statistical profiling of ground truth matches and singletons.
"""
import sys
from collections import Counter
import config

total_entities = 0
singleton_count = 0
match_counts = Counter()

with open(config.TRAIN_GT, "r", encoding="utf-8") as f:
    f.readline() # header
    for line in f:
        total_entities += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2 or not parts[1].strip():
            singleton_count += 1
            match_counts[0] += 1
        else:
            num_matches = len(parts[1].split(","))
            match_counts[num_matches] += 1

print(f"Total S1 entities in train_ground_truth: {total_entities}")
print(f"Singletons (0 matches): {singleton_count} ({singleton_count / total_entities * 100:.2f}%)")
print(f"Entities with matches: {total_entities - singleton_count} ({(total_entities - singleton_count) / total_entities * 100:.2f}%)")
print("\nDistribution of match counts:")
for k in sorted(match_counts.keys())[:10]:
    cnt = match_counts[k]
    print(f"  {k} matches: {cnt} ({cnt / total_entities * 100:.2f}%)")
