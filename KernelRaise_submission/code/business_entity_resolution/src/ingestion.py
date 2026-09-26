"""
High-throughput streaming ingestion pipeline for Business Entity Resolution.
Strictly enforces tab-separation (sep="\\t") and normalizes records in memory-efficient batches.
"""
import sys
import csv
from typing import Iterator, Dict, Any, List, Optional
from normalizer import clean_business_name, clean_address

def stream_records(file_path: str, chunk_size: int = 100000) -> Iterator[List[Dict[str, Any]]]:
    """
    Stream records from a TSV file in chunks to ensure bounded memory usage.
    Each record contains:
        entity_id: str
        business_name: str
        business_address: str
        country: str
        name_full: str
        name_stem: str
        postal_code: str
        locality: str
        clean_addr_tokens: str
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            return

        # Map header indices
        col_map = {col.strip().lower(): idx for idx, col in enumerate(header)}
        id_idx = col_map.get("entity_id", 0)
        name_idx = col_map.get("business_name", 1)
        addr_idx = col_map.get("business_address", 2)
        country_idx = col_map.get("country", 3)

        chunk = []
        for row in reader:
            if not row or len(row) < 4:
                continue
            entity_id = row[id_idx].strip()
            raw_name = row[name_idx]
            raw_addr = row[addr_idx]
            country = row[country_idx].strip()

            name_full, name_stem = clean_business_name(raw_name)
            addr_info = clean_address(raw_addr, country)

            rec = {
                "entity_id": entity_id,
                "business_name": raw_name,
                "business_address": raw_addr,
                "country": country,
                "name_full": name_full,
                "name_stem": name_stem,
                "postal_code": addr_info["postal_code"],
                "locality": addr_info["locality"],
                "clean_addr": addr_info["clean_tokens"],
            }
            chunk.append(rec)

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:
            yield chunk


def load_all_records_by_country(file_path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load all records partitioned by country:
    {
        "US": {entity_id: record, ...},
        "India": {entity_id: record, ...},
        "France": {entity_id: record, ...}
    }
    """
    by_country: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for chunk in stream_records(file_path):
        for rec in chunk:
            c = rec["country"]
            if c not in by_country:
                by_country[c] = {}
            by_country[c][rec["entity_id"]] = rec
    return by_country
