"""Turn the raw snapshot (one row per violation) into the model table (one row per inspection).

Order matters for leakage:
  clean -> collapse to inspections -> history features on the FULL sequence
  -> only then drop re-inspections and pre-2016 rows -> split by year.
Every history feature uses groupby("licenseno").shift(1), so a row only ever
sees inspections that happened before it.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --- Result codes (decided from the value counts, see docs/DECISIONS.md D1.2) ---
FAIL_CODES = {"HE_Fail", "HE_FailExt", "HE_Hearing", "HE_TSOP", "HE_Closure", "HE_VolClos", "HE_FAILNOR", "Fail"}
PASS_CODES = {"HE_Pass", "HE_Filed", "Pass", "NoViol"}
ADMIN_CODES = {"HE_NotReq", "HE_OutBus", "HE_Misc", "HE_Hold", "Closed", "DATAERR"}  # not real inspections: dropped

OWNER_COLUMNS = {"legalowner", "namelast", "namefirst", "_full_text"}
MODEL_START = "2016-01-01"    # the model table starts here; earlier inspections are history only
REINSPECTION_DAYS = 30
WINDOW_DAYS = 3 * 365         # look-back window for prior_n_3y

HISTORY_COLUMNS = ["prior_n", "prior_n_3y", "is_first", "prior_fail_rate", "prior_routine_fail_rate",
                   "last_fail", "last_n_viol", "last_n_lvl3",
                   "last_routine_fail", "last_routine_n_viol", "last_routine_n_lvl3",
                   "days_since_last", "is_reinspection"]

# --- Privacy masking -------------------------------------------------------
FIRST_NAMES = {line.strip() for line in (Path(__file__).parent / "first_names.txt").read_text().splitlines()
               if line.strip() and not line.startswith("#")}

# Common first names left out of first_names.txt because they are also ordinary words, brands
# or streets ("Mark all containers", "Charles Street"). They are only trusted in a name context
# such as "PIC Sam" or "stated by Scott Doe".
_CONTEXT_FIRST_NAMES = FIRST_NAMES | set("""Sam Scott Tom Jack Jim Ben Max Mark Bill Rob Pat Sue Ray Don Ken Ed Al Lee Kim
Charlie Charles Frank Victor Martin Jordan Taylor Morgan Wayne Sandy Stanley Howard Russell Lawrence Warren Kyle Dean
Grant Dale Guy Rich Wendy Grace Hope Joy Rose Douglas Lewis Marshall Nelson Wilson Jackson Franklin Curtis Stuart
Elliott Leon Owen Perry Kay Dawn Holly Ruby Penny Amber Crystal""".split())

# Capitalized words that can follow a role word or a first name without being a person's name.
_NOT_NAMES = {
    "Chef", "Owner", "Manager", "Principal", "Principle", "Assistant", "Asst", "Plumber", "Landlord",
    "Person", "Pic", "Field", "Health", "Heath", "Department", "Division", "Food", "Safety", "Allergen",
    "Awareness", "Pest", "Control", "Commissioner", "Licensed", "Discarded", "Product", "Boston", "City",
    "Fire", "Building", "Buiding", "Disposal", "Governor", "Mask", "Data", "Inspector", "Supervisor",
    "Staff", "Employee", "Employees", "Management", "The", "A", "An", "Operator", "Contact", "Is", "Was",
    "In", "Of", "For", "On", "At", "To", "And", "Not", "No", "Provide", "Clean", "Remove", "Repair",
    "Certified", "Protection", "Service", "Services", "State", "Code", "Kitchen", "Server",
    "St", "Street", "Ave", "Avenue", "Rd", "Road", "Sq", "Square", "Place", "Blvd", "Hall", "Park",
    "Center", "School", "Restaurant", "Cafe", "Pizza", "Market", "Bakery", "Company", "Inc", "Corp",
    "Has", "Had", "Will", "Must", "Shall", "Should", "Stated", "Said", "Did", "Does", "Needs", "Unable",
    "Certificate", "Certification", "Expired", "Present", "Posted", "Available", "Onsite", "During",
    "Exterminator", "Exterminating", "Environmental", "Sanitation", "Inspectional", "Regulations",
}
# Words that introduce a named person. After a TITLE, two capitalized words are treated as a
# name. The broader ROLE words also start ordinary phrases ("Employee Hand Sink"), so after
# them we only mask a known first name.
_TITLE_WORDS = (r"owner|manager|pic|p\.i\.c\.?|chef|supervisor|operator|person in charge|commissioner|"
                r"inspector|bfd|contractor|gm|cfpm|director|officer|president|captain|landlord|instructor")
_ROLE_WORDS = _TITLE_WORDS + r"|contact|isd|cfm|employee|cook|bartender|server|plumber|exterminator|technician|tech|principal"
_SEP = r"\s*[-:,(]*\s*"       # "PIC: Jane", "GM(Jane", "owner - Jane"
_CAP = r"[A-Z][a-z'\-]{1,}"    # one capitalized word

_HONORIFIC = re.compile(rf"\b(Mr|Mrs|Ms|Miss|Dr)\b\.?\s+{_CAP}(?:\s+{_CAP})?")
# "owner Jane Doe", "Commissioner John Roe", "GM(Jane Doe": a role word, then two capitalized words.
_ROLE_FULL = re.compile(rf"(?i:\b(?:{_TITLE_WORDS})\b(?:\s+is|\s+was|\s+named)?){_SEP}({_CAP})\s+({_CAP})")
# "PIC Jane": a role word, then one word that is a known first name.
_ROLE_FIRST = re.compile(rf"(?i:\b(?:{_ROLE_WORDS})\b(?:\s+is|\s+was|\s+named)?){_SEP}({_CAP})\b()")
# "spoke with Chris", "per John": one capitalized word is enough to be a name here.
_SPOKE = re.compile(
    rf"(?i:\b(?:spoke|speak|spoken|speaking|talked|met|discussed)\s+(?:with|to)(?:\s+the)?|\bper)\s+({_CAP})(\s+{_CAP})?")
# "certificate posted for Jane Doe", "certified ... in the name of Jane Doe"
_CERT = re.compile(
    rf"(?i:certificat\w*|certified|servsafe|serv safe|allergen\w*|cfpm|food protection manager)[^.|]{{0,60}}?"
    rf"(?i:\b(?:for|belongs to|issued to|in the name of|held by|name of)\b)\s*[:\-]?\s*({_CAP})\s+({_CAP})")
# "stated by Scott Doe", "and Wayne Doe": a linking word, then a first name and a second word.
_LINK_FULL = re.compile(rf"(?i:\b(?:stated(?:\s+by)?|by|and|with|told|from)\s+)({_CAP})\s+({_CAP})")
# "Jane Doe" anywhere, when the first word is a known first name.
# The pattern starts at the first name itself; a generic "two capitalized words" pattern would
# pair "Captain Jane" first and never test "Jane Doe".
_FULL_NAME = re.compile(r"\b(" + "|".join(sorted(FIRST_NAMES, key=len, reverse=True)) + rf")\s+({_CAP})\b")
_EMAILS = [
    re.compile(r"[\w.\-]+@[\w\-]+(?:\.[\w\-]+)+"),                                       # name@site.com
    re.compile(r"[A-Za-z]+\.[A-Za-z]+\s?@\s?[\w\-]+(?:\s?\.\s?[A-Za-z]{2,4}\b)?"),       # first.last @ site .gov
    re.compile(r"(?i)\b[\w.\-]+\s?@\s?(?:boston|cityofboston|gmail|yahoo|hotmail|aol|comcast|verizon)"
               r"(?:\s?\.\s?[a-z]{2,3}\b)?"),                                              # name @ boston.gov
]
_PHONE = re.compile(r"\(?\b\d{3}\)?[-.\s]{0,3}\d{3}[-.\s]{0,3}\d{4}\b")


def _mask_two(m):
    """Mask the two captured words unless one of them is a known non-name word."""
    first, second = m.group(1), (m.group(2) or "").strip()
    if first in _NOT_NAMES or second in _NOT_NAMES:
        return m.group(0)
    return m.group(0)[: m.start(1) - m.start(0)] + "[NAME]"


def _mask_if_first_name(m):
    """Mask only when the first captured word is a known first name."""
    if m.group(1) not in FIRST_NAMES or (m.group(2) or "").strip() in _NOT_NAMES:
        return m.group(0)
    return m.group(0)[: m.start(1) - m.start(0)] + "[NAME]"


def _mask_if_context_first_name(m):
    """Same, with the wider name list that is only safe inside a name context."""
    if m.group(1) not in _CONTEXT_FIRST_NAMES or (m.group(2) or "").strip() in _NOT_NAMES:
        return m.group(0)
    return m.group(0)[: m.start(1) - m.start(0)] + "[NAME]"


def mask_personal_info(text):
    """Mask personal names, emails and phone numbers in one comment."""
    if not isinstance(text, str) or not text:
        return text
    for rx in _EMAILS:
        text = rx.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _HONORIFIC.sub("[NAME]", text)
    text = _CERT.sub(_mask_two, text)
    text = _ROLE_FULL.sub(_mask_two, text)
    text = _SPOKE.sub(_mask_two, text)
    text = _LINK_FULL.sub(_mask_if_context_first_name, text)
    text = _ROLE_FIRST.sub(_mask_if_context_first_name, text)
    return _FULL_NAME.sub(_mask_if_first_name, text)


# --- Step 1: clean ---------------------------------------------------------
def _local_date(text_series):
    """Timestamps are stored in UTC ('2018-03-20 14:54:25+00'); return the Boston calendar date.
    Some rows add fractional seconds, so we parse the first 19 characters."""
    utc = pd.to_datetime(text_series.str.slice(0, 19), errors="coerce").dt.tz_localize("UTC")
    return utc.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()


def clean(raw):
    """Parse dates, standardize result codes, mask names, drop unusable and administrative rows."""
    assert not OWNER_COLUMNS & set(raw.columns), "owner-name columns must never be loaded"
    df = raw.copy()
    for col in ["licenseno", "businessname", "descript", "result", "comments", "zip", "city", "viol_level", "viol_status"]:
        df[col] = df[col].astype("string").str.strip()

    n0 = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "_id"])   # the source repeats some rows
    print(f"clean: dropped {n0 - len(df):,} exact duplicate rows")

    df["date"] = _local_date(df["resultdttm"])
    df["license_issue_date"] = _local_date(df["issdttm"])

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
    counts = ["label_fail", "n_viol", "n_lvl1", "n_lvl2", "n_lvl3", "n_rechecked_ok"]
    insp[counts] = insp[counts].astype("int64")
    insp["result"] = np.where(insp["label_fail"] == 1, "fail", "pass")
    insp["inspection_id"] = insp["licenseno"] + "_" + insp["date"].dt.strftime("%Y%m%d")
    assert insp["inspection_id"].is_unique
    return insp.drop(columns="any_pass")


# --- Step 3: history features (full sequence, past only) ---------------------
def _count_in_window(dates, window_days=WINDOW_DAYS):
    """For each (sorted) date, how many earlier dates fall within the previous `window_days`."""
    d = dates.to_numpy()
    start = np.searchsorted(d, d - np.timedelta64(window_days, "D"), side="left")
    return pd.Series(np.arange(len(d)) - start, index=dates.index)


def add_history_features(insp, cutoff_days=REINSPECTION_DAYS):
    df = insp.sort_values(["licenseno", "date"]).reset_index(drop=True)
    lic = df["licenseno"]
    g = df.groupby("licenseno")

    df["prior_n"] = g.cumcount()                                   # inspections before this one
    df["prior_n_3y"] = g["date"].transform(_count_in_window)       # ... in the previous 3 years
    df["is_first"] = (df["prior_n"] == 0).astype(int)
    # shift(1) first, so the mean only covers earlier inspections
    df["prior_fail_rate"] = g["label_fail"].transform(lambda s: s.shift(1).expanding().mean())
    df["last_fail"] = g["label_fail"].shift(1)
    df["last_n_viol"] = g["n_viol"].shift(1)
    df["last_n_lvl3"] = g["n_lvl3"].shift(1)
    df["days_since_last"] = (df["date"] - g["date"].shift(1)).dt.days
    df["prior_comments"] = g["comments"].shift(1).fillna("")

    # A visit soon after a failed one is a follow-up (re-inspection), not a routine inspection.
    # The flag itself only uses the past: the previous result and the gap since it.
    df["is_reinspection"] = ((df["last_fail"] == 1) & (df["days_since_last"] <= cutoff_days)).astype(int)

    # Routine-only history. The visit just before a routine inspection is usually a passed
    # re-inspection, so "last_fail" says little. These look back to the last ROUTINE visit:
    # blank out re-inspections, shift(1) to exclude the current row, then carry forward.
    routine = df["is_reinspection"] == 0
    for source, target in [("label_fail", "last_routine_fail"), ("n_viol", "last_routine_n_viol"),
                           ("n_lvl3", "last_routine_n_lvl3")]:
        df[target] = df[source].where(routine).groupby(lic).shift(1).groupby(lic).ffill()
    df["prior_routine_fail_rate"] = (df["label_fail"].where(routine).groupby(lic)
                                     .transform(lambda s: s.shift(1).expanding().mean()))

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


# --- Step 4: split and filter -------------------------------------------------
def add_split(df):
    """Fixed time split: train 2016-2022, validation 2023, test 2024 onward. Earlier rows are history."""
    df = df.copy()
    df["split"] = np.select([df["year"] < 2016, df["year"] <= 2022, df["year"] == 2023],
                            ["history", "train", "val"], default="test")
    return df


def build_model_table(raw):
    """Run every step. Returns (model_table, full_sequence_before_filtering)."""
    full = add_split(add_history_features(collapse_to_inspections(clean(raw))))
    in_period = full["date"] >= MODEL_START
    keep = in_period & (full["is_reinspection"] == 0)
    # licstatus describes today, not the inspection date: keep it out of the modeling table.
    table = full[keep].drop(columns="licstatus").reset_index(drop=True)
    n_period = int(in_period.sum())
    print(f"history only (before {MODEL_START}): {len(full) - n_period:,} inspections")
    print(f"re-inspections dropped: {n_period - len(table):,} of {n_period:,} "
          f"({1 - len(table) / n_period:.1%}); model table has {len(table):,} rows")
    assert not OWNER_COLUMNS & set(table.columns)
    return table, full


# --- Leakage checks -------------------------------------------------------------
def _history_by_hand(h, cutoff_days=REINSPECTION_DAYS):
    """Recompute the history features of one restaurant with a plain loop over its past rows.
    Deliberately written differently from add_history_features, to cross-check it."""
    rows, past = [], []     # past = list of (date, label, n_viol, n_lvl3, is_reinspection)
    for r in h.itertuples():
        prev = past[-1] if past else None
        gap = (r.date - prev[0]).days if prev else np.nan
        is_re = int(bool(prev) and prev[1] == 1 and gap <= cutoff_days)
        routine = [p for p in past if p[4] == 0]
        rows.append({
            "prior_n": len(past),
            "prior_n_3y": sum(1 for p in past if (r.date - p[0]).days <= WINDOW_DAYS),
            "is_first": int(not past),
            "prior_fail_rate": np.mean([p[1] for p in past]) if past else np.nan,
            "prior_routine_fail_rate": np.mean([p[1] for p in routine]) if routine else np.nan,
            "last_fail": prev[1] if prev else np.nan,
            "last_n_viol": prev[2] if prev else np.nan,
            "last_n_lvl3": prev[3] if prev else np.nan,
            "last_routine_fail": routine[-1][1] if routine else np.nan,
            "last_routine_n_viol": routine[-1][2] if routine else np.nan,
            "last_routine_n_lvl3": routine[-1][3] if routine else np.nan,
            "days_since_last": gap,
            "is_reinspection": is_re,
        })
        past.append((r.date, r.label_fail, r.n_viol, r.n_lvl3, is_re))
    return pd.DataFrame(rows, index=h.index)


def check_no_leakage(full, n_restaurants=3, n_cross_check=300, seed=42):
    """Two checks that history features only use earlier inspections.

    1. For 3 random restaurants: prior_n is the row position, last_fail is the previous
       label, prior_comments is the previous inspection's comments.
    2. For 300 random restaurants: every history column equals a by-hand recomputation.
    """
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

    licenses = pd.Series(full["licenseno"].unique()).sample(n_cross_check, random_state=seed)
    n_rows = 0
    for lic in licenses:
        h = full[full["licenseno"] == lic].sort_values("date")
        expected = _history_by_hand(h)
        actual = h[HISTORY_COLUMNS].astype(float)
        assert np.allclose(actual.to_numpy(), expected[HISTORY_COLUMNS].astype(float).to_numpy(), equal_nan=True), \
            f"history mismatch for license {lic}"
        assert (h["prior_comments"].iloc[1:].values == h["comments"].iloc[:-1].values).all()
        n_rows += len(h)
    print(f"  cross-check: {len(HISTORY_COLUMNS)} history columns match a by-hand recomputation "
          f"for {n_cross_check} restaurants ({n_rows:,} inspections)  OK")
