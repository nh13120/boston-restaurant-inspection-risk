"""Pull Boston food establishment inspections from Analyze Boston (CKAN SQL API).

The pull runs once and saves a snapshot. Everything downstream reads the snapshot,
so results do not change when the city updates the live table.
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://data.boston.gov/api/3/action/datastore_search_sql"
RESOURCE_ID = "4582bec6-2b4f-4f9e-bc55-cbaa73117f4c"
DATASET_PAGE = "https://data.boston.gov/dataset/food-establishment-inspections"
LICENSE = "Open Data Commons Public Domain Dedication and License (PDDL)"

# Only the columns we need. The owner-name columns (legalowner, namelast, namefirst)
# are never requested, and neither is "_full_text", which embeds those names.
COLUMNS = [
    "_id", "licenseno", "businessname", "descript", "licstatus", "result", "resultdttm",
    "viol_level", "viol_status", "violdesc", "comments", "issdttm", "zip", "city",
]
FORBIDDEN = {"legalowner", "namelast", "namefirst", "_full_text"}
assert not FORBIDDEN & set(COLUMNS)

# No lower date bound: inspections before 2016 are kept as HISTORY for the features
# (the model table itself starts in 2016, see features.MODEL_START). The upper bound
# freezes the snapshot, so pulling again later returns the same period.
SNAPSHOT_END = "2026-11-11"
WHERE = f""""resultdttm" < '{SNAPSHOT_END}' AND "descript" LIKE 'Eating & Drinking%'"""
PAGE_SIZE = 10_000


def build_sql(limit, offset):
    cols = ", ".join(f'"{c}"' for c in COLUMNS)  # CKAN needs names in double quotes
    return (f'SELECT {cols} FROM "{RESOURCE_ID}" WHERE {WHERE} '
            f'ORDER BY "_id" LIMIT {limit} OFFSET {offset}')


def fetch_page(sql, retries=4, timeout=120):
    """One API call with a timeout and retries (waits 2s, 4s, 8s between tries)."""
    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params={"sql": sql}, timeout=timeout)
            r.raise_for_status()
            body = r.json()
            if not body.get("success"):
                raise RuntimeError(f"API returned success=false: {body.get('error')}")
            return body["result"]["records"]
        except Exception as err:  # network errors, HTTP errors, bad JSON
            if attempt == retries - 1:
                raise RuntimeError(f"API failed after {retries} tries: {err}") from err
            time.sleep(2 ** (attempt + 1))


def content_hash(df):
    """A fingerprint of the table's content (independent of how the file is encoded)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


def _compare_with_reference(meta, reference):
    """Tell the user whether this snapshot is the one the committed results were built on."""
    reference = Path(reference)
    if not reference.exists():
        return False
    ref = json.loads(reference.read_text())
    if ref["content_sha256"] == meta["content_sha256"]:
        print("Snapshot matches the reference in docs/pull_metadata.json.")
        return True
    else:
        print(f"WARNING: snapshot differs from the reference ({meta['row_count']:,} rows here, "
              f"{ref['row_count']:,} in the reference). The city has probably edited the table "
              f"since {ref['pulled_at_utc']} (or this file is partial), so results may differ slightly from the committed ones.")
        return False


def pull_inspections(raw_dir="../data/raw", force=False, reference="../docs/pull_metadata.json"):
    """Return the snapshot DataFrame, pulling from the API only if needed."""
    raw_dir = Path(raw_dir)
    snapshot, meta_file = raw_dir / "inspections_snapshot.parquet", raw_dir / "pull_metadata.json"
    if snapshot.exists() and not force:
        print(f"Snapshot found, skipping the pull: {snapshot}")
        df = pd.read_parquet(snapshot)
        # Hash the file's content itself, so an edited or partial snapshot is noticed.
        same = _compare_with_reference({"content_sha256": content_hash(df), "row_count": len(df)}, reference)
        if same and not meta_file.exists():
            # A snapshot copied in without its metadata (the exact-files path in docs/setup.md):
            # it is the reference snapshot, so the reference metadata describes it.
            meta_file.write_text(Path(reference).read_text())
        return df

    raw_dir.mkdir(parents=True, exist_ok=True)
    pages, offset = [], 0
    while True:
        records = fetch_page(build_sql(PAGE_SIZE, offset))
        if not records:          # empty page = we have everything
            break
        pages.append(pd.DataFrame.from_records(records, columns=COLUMNS))
        offset += len(records)   # advance by rows actually returned
        if (offset // PAGE_SIZE) % 10 == 0:
            print(f"  pulled {offset:,} rows", flush=True)

    df = pd.concat(pages, ignore_index=True)
    assert df["_id"].is_unique, "pagination returned duplicate rows"
    assert not FORBIDDEN & set(df.columns)

    df.to_parquet(snapshot, index=False)
    meta = {
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": DATASET_PAGE,
        "license": LICENSE,
        "api_url": API_URL,
        "resource_id": RESOURCE_ID,
        "sql": build_sql(PAGE_SIZE, 0).replace(f"LIMIT {PAGE_SIZE} OFFSET 0", "LIMIT <page> OFFSET <n>"),
        "row_count": len(df),
        "content_sha256": content_hash(df),
        "columns": COLUMNS,
    }
    meta_file.write_text(json.dumps(meta, indent=2))
    print(f"Saved {len(df):,} rows to {snapshot}")
    _compare_with_reference(meta, reference)
    return df
