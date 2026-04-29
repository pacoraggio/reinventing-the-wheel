import asyncio
import pandas as pd
import pickle
import json
import re
from pathlib import Path
from playwright.async_api import async_playwright

COMPETITIONS = {
    "UCL": "UEFA.CHAMPIONS",
    "UEL": "UEFA.EUROPA",
}

SEASONS = {
    2016: "2016-2017",
    2017: "2017-2018",
    2018: "2018-2019",
    2019: "2019-2020",
    2020: "2020-2021",
    2021: "2021-2022",
    2022: "2022-2023",
    2023: "2023-2024",
    2024: "2024-2025",
    2025: "2025-2026"
}

OUTPUT_PATH = Path("../data/raw/espn_cards_data.pkl")

def parse_espn_json(html: str, comp_key: str, season_label: str) -> pd.DataFrame | None:
    try:
        match = re.search(r"window\['__espnfitt__'\]\s*=\s*(\{.*\});", html, re.DOTALL)
        if not match:
            print(f"  ⚠️  __espnfitt__ block not found")
            return None

        data = json.loads(match.group(1))
        rows = data["page"]["content"]["statistics"]["tableRows"][0]

        records = []
        for row in rows:
            try:
                team_name = row[1]["name"] if isinstance(row[1], dict) else str(row[1])
                games     = row[2]["value"] if isinstance(row[2], dict) else None
                yellows   = row[3]["value"] if isinstance(row[3], dict) else None
                reds      = row[4]["value"] if isinstance(row[4], dict) else None
                records.append({
                    "team":         team_name,
                    "games_played": games,
                    "yellow_cards": yellows,
                    "red_cards":    reds,
                    "season":       season_label,
                    "competition":  comp_key,
                })
            except (IndexError, KeyError, TypeError):
                continue

        df = pd.DataFrame(records)
        print(f"  ✅ {comp_key} {season_label} — {len(df)} teams")
        return df

    except Exception as e:
        print(f"  ⚠️  Parse error: {e}")
        return None

async def fetch_espn_page(league_code: str, year: int, page) -> str | None:
    url = (
        f"https://www.espn.com/soccer/stats/_/league/"
        f"{league_code}/view/discipline/season/{year}"
    )
    try:
        await page.goto(url, timeout=60000)
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(3)
        return await page.content()
    except Exception as e:
        print(f"  ⚠️  {league_code} {year} — {e}")
        return None

async def main():
    all_data = {"UCL": [], "UEL": []}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page    = await browser.new_page()

        await page.goto(
            "https://www.espn.com/soccer/stats/_/league/UEFA.CHAMPIONS/view/discipline",
            timeout=60000
        )
        await page.wait_for_load_state("domcontentloaded")
        print("⏳ Accept cookies if prompted, then press ENTER...")
        input()

        for comp_key, league_code in COMPETITIONS.items():
            print(f"\n🔍 {comp_key}")
            for year, season_label in SEASONS.items():
                print(f"  ⏳ {season_label}...")
                html = await fetch_espn_page(league_code, year, page)
                if html:
                    df = parse_espn_json(html, comp_key, season_label)
                    if df is not None:
                        all_data[comp_key].append(df)
                        print(df.to_string())
                await asyncio.sleep(4)

        await browser.close()

    # ── Combine ──
    all_espn = {
        comp: pd.concat(frames, ignore_index=True)
        for comp, frames in all_data.items()
        if frames
    }

    # ── Save ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(all_espn, f)
    print(f"\n✅ Saved to {OUTPUT_PATH}")

    # ── Preview ──
    for comp, df in all_espn.items():
        print(f"\n--- {comp} ---")
        print(f"Shape:   {df.shape}")
        print(f"Seasons: {sorted(df['season'].unique())}")

if __name__ == "__main__":
    asyncio.run(main())