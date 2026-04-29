import requests
import pandas as pd
import pickle
import time
import zipfile
import io
from pathlib import Path

import openpyxl  # pip install openpyxl



# ── Domestic leagues — individual CSVs ──
DOMESTIC_LEAGUES = {
    "Serie_A":        "I1",
    "Premier_League": "E0",
    "La_Liga":        "SP1",
}

# ── European — inside zip, file codes ──
EUROPEAN_LEAGUES = {
    "UCL": "CL",   # Champions League file inside zip
    "UEL": "EL",   # Europa League file inside zip
}

SEASONS = [
    "2324", "2223", "2122", "2021",
    "1920", "1819", "1718", "1617",
    "1516", "1415"
]

COLS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HF", "AF", "HY", "AY", "HR", "AR",
    "HS", "AS", "HST", "AST", "HC", "AC"
]

# ── Season format for Excel files ──
EXCEL_SEASONS = {
    "2324": "2023-2024", "2223": "2022-2023", "2122": "2021-2022",
    "2021": "2020-2021", "1920": "2019-2020", "1819": "2018-2019",
    "1718": "2017-2018", "1617": "2016-2017", "1516": "2015-2016",
    "1415": "2014-2015"
}

def fetch_european_from_excel(season: str) -> dict[str, pd.DataFrame]:
    """Download all-euro Excel and extract UCL and UEL sheets."""
    long_season = EXCEL_SEASONS[season]
    url = f"https://www.football-data.co.uk/mmz4281/{season}/all-euro-data-{long_season}.xlsx"
    r = requests.get(url, timeout=15)

    if r.status_code != 200:
        # Older seasons use .xls
        url = url.replace(".xlsx", ".xls")
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            print(f"  ❌ Excel {season} — status {r.status_code}")
            return {}

    print(f"  📥 Downloaded Excel {season} — checking sheets...")
    xl = pd.ExcelFile(io.BytesIO(r.content))
    print(f"  📋 Sheets: {xl.sheet_names}")

    results = {}
    for sheet in xl.sheet_names:
        if "Champions" in sheet or "UCL" in sheet or sheet == "CL":
            df = xl.parse(sheet)
            available = [c for c in COLS if c in df.columns]
            df = df[available].copy()
            df["season"] = season
            df["league"] = "UCL"
            results["UCL"] = df
            print(f"  ✅ UCL found in sheet '{sheet}' — {len(df)} matches")

        elif "Europa" in sheet or "UEL" in sheet or sheet == "EL":
            df = xl.parse(sheet)
            available = [c for c in COLS if c in df.columns]
            df = df[available].copy()
            df["season"] = season
            df["league"] = "UEL"
            results["UEL"] = df
            print(f"  ✅ UEL found in sheet '{sheet}' — {len(df)} matches")

    return results

def fetch_domestic(code: str, season: str) -> pd.DataFrame | None:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        df = pd.read_csv(io.StringIO(r.text), on_bad_lines="skip")
        available = [c for c in COLS if c in df.columns]
        df = df[available].copy()
        df["season"] = season
        df["league"] = code
        return df
    print(f"  ❌ {url} — status {r.status_code}")
    return None

def fetch_european_from_zip(file_code: str, season: str) -> pd.DataFrame | None:
    """Download seasonal zip and extract the relevant competition CSV."""
    zip_url = f"https://www.football-data.co.uk/mmz4281/{season}/data.zip"
    r = requests.get(zip_url, timeout=15)

    if r.status_code != 200:
        print(f"  ❌ zip {season} — status {r.status_code}")
        return None

    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))

        # ── Find matching file inside zip ──
        matches = [f for f in z.namelist() if file_code in f and f.endswith(".csv")]
        if not matches:
            print(f"  ⚠️  {file_code} not found in zip {season}. Files: {z.namelist()}")
            return None

        with z.open(matches[0]) as f:
            df = pd.read_csv(f, on_bad_lines="skip", encoding="latin1")

        available = [c for c in COLS if c in df.columns]
        df = df[available].copy()
        df["season"] = season
        df["league"] = file_code
        return df

    except Exception as e:
        print(f"  ❌ zip extract error {season}: {e}")
        return None

# def main():
#     all_data = {}

#     # ── Domestic leagues ──
#     for name, code in DOMESTIC_LEAGUES.items():
#         print(f"\n🔍 {name} ({code})")
#         frames = []
#         for season in SEASONS:
#             df = fetch_domestic(code, season)
#             if df is not None:
#                 frames.append(df)
#                 print(f"  ✅ {name} {season} — {len(df)} matches")
#             else:
#                 print(f"  ⚠️  {name} {season} — no data")
#             time.sleep(0.5)

#         if frames:
#             all_data[name] = pd.concat(frames, ignore_index=True)
#             print(f"  📦 {name} total: {len(all_data[name])} matches")

#     # ── European competitions ──
#     for name, file_code in EUROPEAN_LEAGUES.items():
#         print(f"\n🔍 {name} ({file_code})")
#         frames = []
#         for season in SEASONS:
#             df = fetch_european_from_zip(file_code, season)
#             if df is not None:
#                 frames.append(df)
#                 print(f"  ✅ {name} {season} — {len(df)} matches")
#             else:
#                 print(f"  ⚠️  {name} {season} — no data")
#             time.sleep(1)

#         if frames:
#             all_data[name] = pd.concat(frames, ignore_index=True)
#             print(f"  📦 {name} total: {len(all_data[name])} matches")

#     # ── Save ──
#     output_path = Path("../data/raw/european_leagues_data_test.pkl")  # ← updated folder name
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(output_path, "wb") as f:
#         pickle.dump(all_data, f)
#     print(f"\n✅ Saved to {output_path}")

#     # ── Preview ──
#     for name, df in all_data.items():
#         print(f"\n--- {name} ---")
#         print(f"Shape: {df.shape}")
#         print(f"Seasons: {sorted(df['season'].unique())}")
#         print(df.head(2))

if __name__ == "__main__":
    results = fetch_european_from_excel("2324")
    print(results)

# if __name__ == "__main__":
#     main()