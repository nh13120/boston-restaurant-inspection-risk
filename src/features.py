"""Turn the raw snapshot (one row per violation) into the model table (one row per inspection).

Order matters for leakage:
  clean -> collapse to inspections -> history features on the FULL sequence
  -> only then flag re-inspections -> split by year.
Every history feature uses groupby("licenseno").shift(1), so a row only ever
sees inspections that happened before it.
"""
import re

import numpy as np
import pandas as pd

# --- Result codes (decided from the value counts, see docs/DECISIONS.md D1.1) ---
FAIL_CODES = {"HE_Fail", "HE_FailExt", "HE_Hearing", "HE_TSOP", "HE_Closure", "HE_VolClos", "HE_FAILNOR"}
PASS_CODES = {"HE_Pass", "HE_Filed", "Pass", "NoViol"}
ADMIN_CODES = {"HE_NotReq", "HE_OutBus", "HE_Misc", "Closed"}  # not real inspections: dropped

OWNER_COLUMNS = {"legalowner", "namelast", "namefirst", "_full_text"}
REINSPECTION_DAYS = 30

# --- Privacy masking -------------------------------------------------------
# Capitalized words that follow "spoke with ..." or "owner ..." but are not names.
_NOT_NAMES = {
    "Chef", "Owner", "Manager", "Principal", "Principle", "Assistant", "Asst", "Plumber", "Landlord",
    "Person", "Pic", "Field", "Health", "Heath", "Department", "Division", "Food", "Safety", "Allergen",
    "Awareness", "Pest", "Control", "Commissioner", "Licensed", "Discarded", "Product", "Boston", "City",
    "Fire", "Building", "Buiding", "Disposal", "Governor", "Mask", "Data", "Inspector", "Supervisor",
    "Staff", "Employee", "Employees", "Management", "The", "A", "An", "Operator", "Contact", "Is", "Was",
    "In", "Of", "For", "On", "At", "To", "And", "Not", "No", "Provide", "Clean", "Remove", "Repair",
    "Certified", "Protection", "Service", "Services", "State", "Code", "Kitchen", "Cook", "Server",
}
_HONORIFIC = re.compile(r"\b(Mr|Mrs|Ms|Miss|Dr)\b\.?\s+[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)?")
# "spoke with Chris", "per John": one capitalized word is enough to be a name.
_SPOKE = re.compile(
    r"(?i:\b(?:spoke|speak|spoken|talked|met|discussed)\s+(?:with|to)|\bper)"
    r"\s+([A-Z][a-z]+)(\s+[A-Z][a-z]{2,})?")
# "owner Jane Doe": after a role word we require two capitalized words, because a single
# one is usually the start of a sentence ("Manager Provide ...").
_ROLE = re.compile(
    r"(?i:\b(?:owner|manager|pic|chef|supervisor|operator|contact|person in charge)(?:\s+is|\s+was|\s+named)?)"
    r"\s+([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})")
_EMAIL = re.compile(r"[\w.\-]+@[\w\-]+(?:\.[\w\-]+)+")
_PHONE = re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")


def _mask_name(m):
    """Keep the context words ('spoke with', 'owner'), replace the name unless it is a role word."""
    first, second = m.group(1), (m.group(2) or "").strip()
    if first in _NOT_NAMES or second in _NOT_NAMES:
        return m.group(0)
    return m.group(0)[: m.start(1) - m.start(0)] + "[NAME]"


def mask_personal_info(text):
    """Mask personal names, emails and phone numbers in one comment."""
    if not isinstance(text, str) or not text:
        return text
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _HONORIFIC.sub("[NAME]", text)
    text = _SPOKE.sub(_mask_name, text)
    return _ROLE.sub(_mask_name, text)


# --- Step 1: clean ---------------------------------------------------------
def clean(raw):
    """Parse dates, standardize result codes, mask names, drop unusable and administrative rows."""
    assert not OWNER_COLUMNS & set(raw.columns), "owner-name columns must never be loaded"
    df = raw.copy()
    for col in ["licenseno", "businessname", "descript", "result", "comments", "zip", "city", "viol_level", "viol_status"]:
        df[col] = df[col].astype("string").str.strip()

    # Dates look like '2018-03-20 14:54:25+00' (some have fractional seconds):
    # the first 19 characters parse the same way for every row.
    df["date"] = pd.to_datetime(df["resultdttm"].str.slice(0, 19), errors="coerce").dt.normalize()
    df["license_issue_date"] = pd.to_datetime(df["issdttm"].str.slice(0, 19), errors="coerce").dt.normalize()

    n0 = len(df)
    df = df[df["licenseno"].notna() & (df["licenseno"] != "") & df["date"].notna()]
    print(f"clean: dropped {n0 - len(df):,} rows missing a license or date")

    unknown = set(df["result"].dropna().unique()) - FAIL_CODES - PASS_CODES - ADMIN_CODES
    assert not unknown, f"unmapped result codes: {unknown}"
    n1 = len(df)
    df = df[~df["result"].isin(ADMIN_CODES) & df["result"].notna()]
    print(f"clean: dropped {n1 - len(df):,} administrative rows ({', '.join(sorted(ADMIN_CODES))})")
    df["row_fail"] = df["result"].isin(FAIL_CODES)

    # Mask each distinct comment once (much faster than row by row).
    uniq = df["comments"].dropna().unique()
    masked = {c: mask_personal_info(c) for c in uniq}
    n_masked = sum(1 for c in uniq if masked[c] != c)
    df["comments"] = df["comments"].map(masked).fillna("")
    print(f"clean: masked personal info in {n_masked:,} of {len(uniq):,} distinct comments")

    # 'city' holds Boston neighborhood names; tidy the spelling variants ("Roslindale/" etc).
    hood = df["city"].str.rstrip("/").str.title().replace(
        {"Downtown/Financial District": "Financial District", "Dorchester Center": "Dorchester"})
    df["neighborhood"] = hood.fillna("Unknown")
    df["zip"] = df["zip"].fillna("Unknown")
    return df.sort_values("_id").reset_index(drop=True)


# --- Step 2: one row per inspection -----------------------------------------
def collapse_to_inspections(df):
    """One inspection = one license on one calendar date."""
    # A violation counts as "found" only when its status is Fail. Rows with status Pass
    # are earlier violations re-checked and found corrected.
    found = df["viol_status"].eq("Fail") & df["violdesc"].notna()
    work = df.assign(
        v=found,
        v1=found & df["viol_level"].eq("*"),
        v2=found & df["viol_level"].eq("**"),
        v3=found & df["viol_level"].eq("***"),
        rechecked=df["viol_status"].eq("Pass") & df["violdesc"].notna(),
        row_pass=~df["row_fail"],
    )
    keys = ["licenseno", "date"]
    g = work.groupby(keys, sort=True)
    insp = g.agg(
        label_fail=("row_fail", "max"),       # conflict rule: any fail code that day = fail
        any_pass=("row_pass", "max"),
        result_codes=("result", lambda s: "|".join(sorted(set(s)))),
        n_viol=("v", "sum"), n_lvl1=("v1", "sum"), n_lvl2=("v2", "sum"), n_lvl3=("v3", "sum"),
        n_rechecked_ok=("rechecked", "sum"),
        # dict.fromkeys keeps the first occurrence of each comment, in row order
        comments=("comments", lambda s: " | ".join(dict.fromkeys(c for c in s if c))),
        businessname=("businessname", "first"),
        descript=("descript", "first"),
        zip=("zip", "first"),
        neighborhood=("neighborhood", "first"),
        license_issue_date=("license_issue_date", "first"),
        licstatus=("licstatus", "first"),
    ).reset_index()

    conflicts = int((insp["label_fail"] & insp["any_pass"]).sum())
    print(f"collapse: {len(insp):,} inspections; {conflicts} had both a fail and a pass code "
          f"on the same day (labelled fail)")
    insp["label_fail"] = insp["label_fail"].astype(int)
    insp["result"] = np.where(insp["label_fail"] == 1, "fail", "pass")
    insp["inspection_id"] = insp["licenseno"] + "_" + insp["date"].dt.strftime("%Y%m%d")
    assert insp["inspection_id"].is_unique
    return insp.drop(columns="any_pass")


# --- Step 3: history features (full sequence, past only) ---------------------
def add_history_features(insp):
    df = insp.sort_values(["licenseno", "date"]).reset_index(drop=True)
    g = df.groupby("licenseno")

    df["prior_n"] = g.cumcount()                                   # inspections before this one
    df["is_first"] = (df["prior_n"] == 0).astype(int)
    # shift(1) first, so the mean only covers earlier inspections
    df["prior_fail_rate"] = g["label_fail"].transform(lambda s: s.shift(1).expanding().mean())
    df["last_fail"] = g["label_fail"].shift(1)
    df["last_n_viol"] = g["n_viol"].shift(1)
    df["last_n_lvl3"] = g["n_lvl3"].shift(1)
    df["days_since_last"] = (df["date"] - g["date"].shift(1)).dt.days
    df["prior_comments"] = g["comments"].shift(1).fillna("")

    # License age. The issue date in the snapshot is the CURRENT one and can be later than
    # an old inspection (re-issued licenses, pre-opening visits). To avoid using a future
    # date, the clock starts at the earlier of the issue date and the first inspection we see.
    first_seen = g["date"].transform("min")
    start = df["license_issue_date"].where(df["license_issue_date"] <= first_seen, first_seen)
    df["license_age_days"] = (df["date"] - start.fillna(first_seen)).dt.days

    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_takeout"] = df["descript"].str.contains("Take Out", na=False).astype(int)
    return df


# --- Step 4: re-inspections and split ----------------------------------------
def flag_reinspections(df, cutoff_days=REINSPECTION_DAYS):
    """A visit soon after a failed one is a follow-up, not a routine inspection."""
    df = df.copy()
    df["is_reinspection"] = ((df["last_fail"] == 1) & (df["days_since_last"] <= cutoff_days)).astype(int)
    return df


def add_split(df):
    """Fixed time split: train 2016-2022, validation 2023, test 2024 onward."""
    df = df.copy()
    df["split"] = np.select([df["year"] <= 2022, df["year"] == 2023], ["train", "val"], default="test")
    return df


def build_model_table(raw):
    """Run every step. Returns (model_table, full_sequence_before_filtering)."""
    full = add_split(flag_reinspections(add_history_features(collapse_to_inspections(clean(raw)))))
    table = full[full["is_reinspection"] == 0].reset_index(drop=True)
    print(f"re-inspections dropped: {len(full) - len(table):,} of {len(full):,} "
          f"({1 - len(table) / len(full):.1%}); model table has {len(table):,} rows")
    assert not OWNER_COLUMNS & set(table.columns)
    return table, full


def check_no_leakage(full, n_restaurants=3, seed=42):
    """Spot-check 3 random restaurants: each history feature must equal the previous row's value."""
    counts = full["licenseno"].value_counts()
    sample = counts[counts >= 4].sample(n_restaurants, random_state=seed).index
    for lic in sample:
        h = full[full["licenseno"] == lic].sort_values("date").reset_index(drop=True)
        assert (h["prior_n"] == np.arange(len(h))).all(), "prior_n != row position"
        assert h["last_fail"].iloc[1:].eq(h["label_fail"].shift(1).iloc[1:]).all(), "last_fail != previous label"
        assert (h["prior_comments"].iloc[1:].values == h["comments"].iloc[:-1].values).all(), \
            "prior_comments != previous comments"
        assert pd.isna(h["last_fail"].iloc[0]) and h["prior_comments"].iloc[0] == ""
        print(f"  license {lic}: {len(h)} inspections, history features only use earlier rows  OK")
