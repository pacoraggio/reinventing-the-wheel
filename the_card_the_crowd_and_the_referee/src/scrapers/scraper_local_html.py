import pandas as pd
import pickle
import re
from pathlib import Path

# ── Paths ──
HTML_DIR    = Path("../data/raw/fbref_html")
OUTPUT_PATH = Path("../data/raw/european_comps_data.pkl")

# ── Valid seasons ──
VALID_SEASONS = {
    "2016-2017", "2017-2018", "2018-2019", "2019-2020",
    "2020-2021", "2021-2022", "2022-2023", "2023-2024",
    "2024-2025", "2025-2026"
}

# ── Column mapping: {source_col: (squad_name, opponent_name)} ──
COL_MAP = {
    "Fls":   ("fls_committed",           "fls_won"),
    "Fld":   ("fld_drawn",               "fld_conceded"),
    "CrdY":  ("ycards_received",         "ycards_caused"),
    "CrdR":  ("rcards_received",         "rcards_caused"),
    "2CrdY": ("ycards_2nd_received",     "ycards_2nd_caused"),
    "Off":   ("offsides_committed",      "offsides_caused"),
    "Crs":   ("crosses_performed",       "crosses_faced"),
    "Int":   ("interceptions_performed", "interceptions_conceded"),
    "TklW":  ("tackles_won",             "tackles_conceded"),
    "PKwon": ("penalties_won",           "penalties_conceded"),
    "OG":    ("og_scored",               "og_forced"),
}

# ── Shared columns ──
SHARED_COLS = ["Squad", "# Pl", "90s"]

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[1] if col[1] and col[1] != col[0] else col[0]
            for col in df.columns
        ]
    return df

def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Remove header repeat rows and empty rows."""
    df = df[df["Squad"].notna()]
    df = df[~df["Squad"].str.contains("Squad|vs\\.", na=True)]
    df = df[df["Squad"] != "Squad"]
    return df.reset_index(drop=True)

def split_country_squad(df: pd.DataFrame) -> pd.DataFrame:
    """Extract 2-3 char country code from squad column."""
    pattern = r'^([a-z]{2,3})\s+(.+)$'
    extracted = df["squad"].str.extract(pattern)
    df.insert(1, "country_code", extracted[0])
    df["squad"] = extracted[1]
    return df

def extract_columns(df: pd.DataFrame, col_suffix: str) -> pd.DataFrame:
    """Extract shared + mapped cols from one table."""
    df = flatten_columns(df)

    result = {}

    # ── Shared columns ──
    for col in SHARED_COLS:
        match = next((c for c in df.columns if c == col), None)
        if match:
            key = col.lower().replace("# ", "num_").replace(" ", "_")
            result[key] = df[match]

    # ── Mapped columns ──
    for src_col, (squad_name, opp_name) in COL_MAP.items():
        target_name = squad_name if col_suffix == "squad" else opp_name
        match = next((c for c in df.columns if c == src_col), None)
        result[target_name] = df[match] if match else None

    return pd.DataFrame(result)

def parse_fbref_html(filepath: Path) -> pd.DataFrame | None:
    try:
        tables = pd.read_html(filepath)

        if len(tables) < 2:
            print(f"  ⚠️  {filepath.name} — less than 2 tables, skipping")
            return None

        # ── Flatten and clean both tables ──
        squad_df = clean_table(flatten_columns(tables[0].copy()))
        opp_df   = clean_table(flatten_columns(tables[1].copy()))

        # ── Extract columns ──
        squad_extracted = extract_columns(squad_df, "squad")
        opp_extracted   = extract_columns(opp_df,   "opponent")

        # ── Drop shared cols from opponent (already in squad) ──
        opp_extracted = opp_extracted.drop(
            columns=[c for c in ["squad", "num_pl", "90s"] if c in opp_extracted.columns],
            errors="ignore"
        )

        # ── Merge on index ──
        df = pd.concat([squad_extracted, opp_extracted], axis=1)

        # ── Extract country code from squad ──
        df = split_country_squad(df)

        # ── Add metadata ──
        stem  = filepath.stem
        parts = stem.split("_", 1)
        df["competition"] = parts[0]
        df["season"]      = parts[1] if len(parts) > 1 else "unknown"

        print(f"  ✅ {filepath.name} — {len(df)} teams")
        return df

    except Exception as e:
        print(f"  ⚠️  {filepath.name} — {e}")
        return None

def main():
    all_frames = {"UCL": [], "UEL": [], "UECL": []}

    for comp in all_frames.keys():
        print(f"\n🔍 Parsing {comp}...")
        files = sorted(HTML_DIR.glob(f"{comp}_*.html"))
        print(f"  📋 Found {len(files)} files")

        for html_file in files:
            season = html_file.stem.split("_", 1)[1]
            if season not in VALID_SEASONS:
                print(f"  ⏭️  Skipping {html_file.name} — pre 2016-2017")
                continue

            df = parse_fbref_html(html_file)
            if df is not None:
                all_frames[comp].append(df)

    # ── Combine ──
    all_euro = {
        comp: pd.concat(frames, ignore_index=True)
        for comp, frames in all_frames.items()
        if frames
    }

    # ── Save ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(all_euro, f)
    print(f"\n✅ Saved to {OUTPUT_PATH}")

    # ── Preview ──
    for comp, df in all_euro.items():
        print(f"\n--- {comp} ---")
        print(f"Shape:   {df.shape}")
        print(f"Seasons: {sorted(df['season'].unique())}")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head(3))

if __name__ == "__main__":
    main()
