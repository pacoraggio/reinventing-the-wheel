"""
discipline_pipeline.py
----------------------
Parameterised hypothesis testing pipeline for football disciplinary analysis.

Unlike hypothesis_tests.py (single-season, full diagnostics) and
multi_season.py (pooled, hardcoded columns), functions here accept the
numerator and denominator column names as arguments so the same pipeline
works for both directions of the analysis:

  Fouls committed → yellow cards received  (the default)
      n_col = 'fouls_committed', k_col = 'yellow_cards'

  Fouls received → yellow cards forced
      n_col = 'fouls_received', k_col = 'yellow_cards_forced'

Functions
---------
run_boot_mw   — bootstrap + Mann-Whitney for one team on any slice of data
z_screen      — z-test screening for all teams in one season
classify      — label a result as Significant / Borderline / Null + direction
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_boot_mw(team_name, df,
                n_col='fouls_committed', k_col='yellow_cards',
                n_bootstrap=10_000, seed=42):
    """
    Bootstrap + Mann-Whitney pipeline for one team on an arbitrary slice of df.

    Works for a single season (pass one season's rows) or pooled across
    multiple seasons (pass all seasons' rows).  The reference distribution
    is always the rest of the league in the same slice.

    Parameters
    ----------
    team_name   : str          — team name as it appears in df['team']
    df          : pd.DataFrame — match-level data; must contain n_col and k_col
    n_col       : str          — denominator column (default 'fouls_committed')
    k_col       : str          — numerator column   (default 'yellow_cards')
    n_bootstrap : int          — bootstrap resamples (default 10,000)
    seed        : int          — random seed for reproducibility

    Returns
    -------
    dict with keys:
        team, n, k, p_hat, p0, p_boot, p_mw, r_rb, null_rates
    Returns None if the team has zero denominator rows.

    Notes
    -----
    The bootstrap loop draws numerator and denominator from two independent
    resamples of rest_df.  This introduces a small amount of extra noise in
    each null rate estimate but has negligible impact at n_bootstrap=10,000.
    """
    team_df = df[df['team'] == team_name].copy()
    rest_df = df[df['team'] != team_name].copy()

    n = int(team_df[n_col].sum())
    k = int(team_df[k_col].sum())
    if n == 0:
        return None

    p_hat = k / n
    p0    = float(rest_df[k_col].sum() / rest_df[n_col].sum())
    m     = len(team_df)

    rng = np.random.default_rng(seed)
    null_rates = np.array([
        rest_df.sample(n=m, replace=True,
                       random_state=int(rng.integers(1_000_000_000)))[k_col].sum() /
        rest_df.sample(n=m, replace=True,
                       random_state=int(rng.integers(1_000_000_000)))[n_col].sum()
        for _ in range(n_bootstrap)
    ])
    null_mean = null_rates.mean()
    p_boot    = float((np.abs(null_rates - null_mean) >= abs(p_hat - null_mean)).mean())

    team_rates = (team_df[k_col] / team_df[n_col]).replace([np.inf, -np.inf], np.nan).dropna()
    rest_rates = (rest_df[k_col] / rest_df[n_col]).replace([np.inf, -np.inf], np.nan).dropna()
    u, p_mw    = stats.mannwhitneyu(team_rates, rest_rates, alternative='two-sided')
    r_rb       = float((2 * u) / (len(team_rates) * len(rest_rates)) - 1)

    return dict(
        team=team_name, n=n, k=k,
        p_hat=round(p_hat, 4), p0=round(p0, 4),
        p_boot=round(p_boot, 4), p_mw=round(p_mw, 4),
        r_rb=round(r_rb, 4),
        null_rates=null_rates,
    )


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def z_screen(df_season, n_col='fouls_committed', k_col='yellow_cards'):
    """
    Z-test screening for every team in a single-season DataFrame.

    Parameters
    ----------
    df_season : pd.DataFrame — one season's match-level rows for one league
    n_col     : str          — denominator column (default 'fouls_committed')
    k_col     : str          — numerator column   (default 'yellow_cards')

    Returns
    -------
    pd.DataFrame with columns:
        team, n, k, rate, league_rate, z, p_z, significant, direction
    """
    rows = []
    for team in df_season['team'].unique():
        team_df = df_season[df_season['team'] == team]
        rest_df = df_season[df_season['team'] != team]

        n = int(team_df[n_col].sum())
        k = int(team_df[k_col].sum())
        if n == 0:
            continue

        p_hat = k / n
        p0    = float(rest_df[k_col].sum() / rest_df[n_col].sum())
        se    = np.sqrt(p0 * (1 - p0) / n)
        z     = (p_hat - p0) / se
        p_z   = float(2 * stats.norm.cdf(-abs(z)))

        rows.append(dict(
            team=team, n=n, k=k,
            rate=round(p_hat, 4),
            league_rate=round(p0, 4),
            z=round(z, 3), p_z=round(p_z, 4),
            significant=p_z < 0.05,
            direction='high' if p_hat > p0 else 'low',
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Result classification
# ---------------------------------------------------------------------------

def classify(p_boot, p_mw, direction):
    """
    Classify a pooled result as Significant, Borderline, or Null.

    Both tests significant (p < 0.05) → 'Significant (<direction>)'
    Exactly one test significant      → 'Borderline (<direction>)'
    Neither significant               → 'Null'

    Parameters
    ----------
    p_boot    : float — bootstrap p-value
    p_mw      : float — Mann-Whitney p-value
    direction : str   — 'low' or 'high'

    Returns
    -------
    str — one of 'Significant (low)', 'Significant (high)',
          'Borderline (low)', 'Borderline (high)', 'Null'
    """
    both = (p_boot < 0.05) and (p_mw < 0.05)
    one  = (p_boot < 0.05) or  (p_mw < 0.05)
    if both:
        return f'Significant ({direction})'
    elif one:
        return f'Borderline ({direction})'
    return 'Null'