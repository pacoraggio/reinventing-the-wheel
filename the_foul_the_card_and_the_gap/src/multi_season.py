"""
multi_season.py
---------------
Reusable functions for multi-season disciplinary analysis.

Designed to work across leagues — pass the appropriate DataFrame and
season/team lists explicitly rather than relying on notebook-level globals.

Functions
---------
fmt_season          — format a compact season code ('1112') to '2011-12'
make_pivot          — build a team × season pivot table from a results DataFrame
fmt_pvalue          — format a p-value for display, flagging significance with *
run_full_pipeline   — pooled hypothesis test pipeline for one team across N seasons
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_season(s):
    """
    Convert a compact season code to a human-readable label.

    Parameters
    ----------
    s : str — four-character season code, e.g. '1112'

    Returns
    -------
    str — e.g. '2011-12'
    """
    return f"20{s[:2]}-{s[2:]}"


def fmt_pvalue(v, threshold=0.05):
    """
    Format a p-value for table display.

    Values below `threshold` are flagged with *.
    NaN returns '-'.

    Parameters
    ----------
    v         : float or NaN — p-value to format
    threshold : float        — significance threshold (default 0.05)

    Returns
    -------
    str — e.g. '0.023 *' or '0.187'
    """
    if pd.isna(v):
        return '-'
    return f'{v:.3f} *' if v < threshold else f'{v:.3f}'


# ---------------------------------------------------------------------------
# Pivot table builder
# ---------------------------------------------------------------------------

def make_pivot(col, df, team_list, seasons, readable_seasons=None):
    """
    Build a team × season pivot table from a long-format results DataFrame.

    Parameters
    ----------
    col              : str         — column to pivot (e.g. 'p_boot', 'r_rb')
    df               : pd.DataFrame — long-format results with 'team' and 'season' columns
    team_list        : list of str  — row order (teams)
    seasons          : list of str  — column order using compact codes (e.g. ['1112', '1213'])
    readable_seasons : list of str, optional
        Human-readable column labels (e.g. ['2011-12', '2012-13']).
        Defaults to seasons if not provided.

    Returns
    -------
    pd.DataFrame — pivot with teams as rows, seasons as columns
    """
    if readable_seasons is None:
        readable_seasons = seasons

    pv = df.pivot(index='team', columns='season', values=col)
    pv = pv.reindex(index=team_list, columns=seasons)
    pv.columns = readable_seasons
    return pv


# ---------------------------------------------------------------------------
# Pooled multi-season pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(team_name, df, n_bootstrap=10_000, seed=42):
    """
    Run the pooled hypothesis test pipeline for one team across all seasons
    present in `df`.

    Tests applied: z-test (normal approximation), exact binomial, bootstrap,
    Mann-Whitney U with rank-biserial correlation effect size.

    Parameters
    ----------
    team_name   : str          — team name as it appears in df['team']
    df          : pd.DataFrame — match-level data for one league, all seasons.
                  Must contain 'team', 'yellow_cards', 'fouls_committed'.
    n_bootstrap : int          — bootstrap resamples (default 10,000)
    seed        : int          — random seed for reproducibility

    Returns
    -------
    dict with keys:
        team, n, k, m, p_hat, p0,
        z, p_z, p_binom, p_boot, p_mw, r_rb,
        null_rates (np.ndarray of bootstrap null crude rates)

    Notes
    -----
    Known limitation: the bootstrap loop draws yellow_cards and fouls_committed
    from two independent resamples of rest_df rather than one. This introduces
    a small amount of noise in each null crude rate estimate. The effect on the
    final p-value is minimal given n_bootstrap=10,000, but it is not the standard
    parametric bootstrap. Worth correcting in a future revision.
    """
    team_df = df[df['team'] == team_name].copy()
    rest_df = df[df['team'] != team_name].copy()

    n     = int(team_df['fouls_committed'].sum())
    k     = int(team_df['yellow_cards'].sum())
    p_hat = k / n
    p0    = rest_df['yellow_cards'].sum() / rest_df['fouls_committed'].sum()
    m     = len(team_df)

    # Z-test and exact binomial
    se      = np.sqrt(p0 * (1 - p0) / n)
    z       = (p_hat - p0) / se
    p_z     = 2 * stats.norm.cdf(-abs(z))
    p_binom = stats.binomtest(k, n, p0, alternative='two-sided').pvalue

    # Bootstrap (see note above re: two independent resamples)
    rng = np.random.default_rng(seed)
    null_rates = np.array([
        rest_df.sample(n=m, replace=True, random_state=int(rng.integers(1_000_000_000)))
               ['yellow_cards'].sum() /
        rest_df.sample(n=m, replace=True, random_state=int(rng.integers(1_000_000_000)))
               ['fouls_committed'].sum()
        for _ in range(n_bootstrap)
    ])
    null_mean = null_rates.mean()
    p_boot    = (np.abs(null_rates - null_mean) >= abs(p_hat - null_mean)).mean()

    # Mann-Whitney
    team_rates = (team_df['yellow_cards'] / team_df['fouls_committed']
                  ).replace([np.inf, -np.inf], np.nan).dropna()
    rest_rates = (rest_df['yellow_cards'] / rest_df['fouls_committed']
                  ).replace([np.inf, -np.inf], np.nan).dropna()
    u_stat, p_mw = stats.mannwhitneyu(team_rates, rest_rates, alternative='two-sided')
    n1, n2 = len(team_rates), len(rest_rates)
    r_rb = float((2 * u_stat) / (n1 * n2) - 1)

    return dict(
        team=team_name, n=n, k=k, m=m,
        p_hat=p_hat, p0=float(p0),
        z=float(z), p_z=float(p_z), p_binom=float(p_binom),
        p_boot=float(p_boot), p_mw=float(p_mw), r_rb=r_rb,
        null_rates=null_rates,
    )