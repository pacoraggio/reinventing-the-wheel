"""
hypothesis_tests.py
-------------------
Reusable functions for the one-team, one-season hypothesis test pipeline.

Pipeline steps
--------------
1. compute_team_stats     — observed rate p̂ and reference rate p₀
2. proportion_tests       — z-test (normal approx.) + exact binomial
3. dispersion_check       — Pearson χ² overdispersion check
4. autocorrelation_check  — between-match serial correlation
5. bootstrap_test         — simulation-based null distribution and p-value
6. mannwhitney_test       — non-parametric two-sample test + effect size
7. run_pipeline           — runs all steps and returns a single results dict
8. plot_bootstrap_grid    — visualise null distributions for multiple teams
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------------------
# Step 1 — observed and reference rates
# ---------------------------------------------------------------------------

def compute_team_stats(team_name, season_df):
    """
    Compute the observed crude rate for one team and the reference rate
    from the rest of the league (Juventus excluded from its own reference).

    Parameters
    ----------
    team_name : str
        Team name as it appears in season_df['team'].
    season_df : pd.DataFrame
        Match-level dataframe for one league and one season.
        Must contain 'team', 'yellow_cards', 'fouls_committed'.

    Returns
    -------
    dict with keys: team, n, k, p_hat, p0, m, team_df, rest_df
    """
    team_df = season_df[season_df['team'] == team_name].copy()
    rest_df = season_df[season_df['team'] != team_name].copy()

    n     = int(team_df['fouls_committed'].sum())
    k     = int(team_df['yellow_cards'].sum())
    p_hat = k / n
    p0    = rest_df['yellow_cards'].sum() / rest_df['fouls_committed'].sum()
    m     = len(team_df)

    return dict(team=team_name, n=n, k=k, p_hat=p_hat, p0=float(p0),
                m=m, team_df=team_df, rest_df=rest_df)


# ---------------------------------------------------------------------------
# Step 2 — parametric tests
# ---------------------------------------------------------------------------

def proportion_tests(n, k, p0):
    """
    One-sample proportion z-test (normal approximation) and exact binomial test.

    Parameters
    ----------
    n  : int   — number of trials (fouls committed)
    k  : int   — number of successes (yellow cards)
    p0 : float — reference rate (null hypothesis value)

    Returns
    -------
    dict with keys: se, z, p_value_z, p_value_binom,
                    condition_np0, condition_n1p0, conditions_met
    """
    se          = np.sqrt(p0 * (1 - p0) / n)
    z           = (k / n - p0) / se
    p_value_z   = float(2 * stats.norm.cdf(z))
    p_value_binom = float(stats.binomtest(k, n, p0, alternative='two-sided').pvalue)

    np0   = n * p0
    n1p0  = n * (1 - p0)

    return dict(
        se=float(se), z=float(z),
        p_value_z=p_value_z,
        p_value_binom=p_value_binom,
        condition_np0=float(np0),
        condition_n1p0=float(n1p0),
        conditions_met=(np0 > 10 and n1p0 > 10),
    )


# ---------------------------------------------------------------------------
# Step 3 — overdispersion check (Assumption 1: constant p)
# ---------------------------------------------------------------------------

def dispersion_check(team_df, p_hat):
    """
    Pearson χ² test for overdispersion relative to the binomial model.

    Parameters
    ----------
    team_df : pd.DataFrame — match-level rows for the team under test.
              Must contain 'fouls_committed' and 'yellow_cards'.
    p_hat   : float        — team's observed crude rate.

    Returns
    -------
    dict with keys: chi2, dof, phi (dispersion ratio), p_value
    """
    expected  = team_df['fouls_committed'] * p_hat
    observed  = team_df['yellow_cards']
    chi2      = float(((observed - expected) ** 2 /
                       (team_df['fouls_committed'] * p_hat * (1 - p_hat))).sum())
    dof       = len(team_df) - 1
    phi       = chi2 / dof
    p_value   = float(1 - stats.chi2.cdf(chi2, dof))

    return dict(chi2=chi2, dof=dof, phi=phi, p_value=p_value)


# ---------------------------------------------------------------------------
# Step 4 — autocorrelation check (Assumption 2: between-match independence)
# ---------------------------------------------------------------------------

def autocorrelation_check(team_df, max_lags=5):
    """
    Autocorrelation of per-match ycards_per_fouls at short lags.

    Parameters
    ----------
    team_df  : pd.DataFrame — must contain 'ycards_per_fouls' and 'date'.
    max_lags : int          — number of lags to compute (default 5).

    Returns
    -------
    dict with keys: autocorrs (list), ci_95 (±bound), max_abs_lag
    """
    rates     = team_df.sort_values('date')['ycards_per_fouls']
    autocorrs = [float(rates.autocorr(lag=l)) for l in range(1, max_lags + 1)]
    ci_95     = 1.96 / np.sqrt(len(rates))
    max_abs   = int(np.argmax(np.abs(autocorrs))) + 1

    return dict(autocorrs=autocorrs, ci_95=float(ci_95), max_abs_lag=max_abs)


# ---------------------------------------------------------------------------
# Step 5 — bootstrap test
# ---------------------------------------------------------------------------

def bootstrap_test(team_name, season_df, n_bootstrap=10_000, seed=42):
    """
    Bootstrap null distribution for one team's crude rate.

    Resamples `m` matches (with replacement) from the rest of the league
    `n_bootstrap` times to build a null distribution, then computes a
    two-tailed p-value.

    Parameters
    ----------
    team_name   : str
    season_df   : pd.DataFrame
    n_bootstrap : int   — number of bootstrap resamples (default 10,000)
    seed        : int   — random seed for reproducibility

    Returns
    -------
    dict with keys: null_rates, null_mean, p_hat, distance, p_value
    """
    rng     = np.random.default_rng(seed)
    stats_  = compute_team_stats(team_name, season_df)
    rest_df = stats_['rest_df']
    m       = stats_['m']
    p_hat   = stats_['p_hat']

    null_rates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rest_df.sample(n=m, replace=True,
                                random_state=int(rng.integers(1_000_000_000)))
        null_rates[i] = (sample['yellow_cards'].sum() /
                         sample['fouls_committed'].sum())

    null_mean = float(null_rates.mean())
    distance  = abs(p_hat - null_mean)
    p_value   = float((np.abs(null_rates - null_mean) >= distance).mean())

    return dict(null_rates=null_rates, null_mean=null_mean,
                p_hat=p_hat, distance=distance, p_value=p_value)


# ---------------------------------------------------------------------------
# Step 6 — Mann-Whitney U test
# ---------------------------------------------------------------------------

def mannwhitney_test(team_name, season_df):
    """
    Mann-Whitney U test comparing per-match rates of one team vs the rest.

    Parameters
    ----------
    team_name : str
    season_df : pd.DataFrame — must contain 'yellow_cards' and 'fouls_committed'.
                If 'ycards_per_fouls' is absent it is computed on the fly.

    Returns
    -------
    dict with keys: u_stat, p_value, n1, n2, r_rb (rank-biserial correlation)
    """
    if 'ycards_per_fouls' not in season_df.columns:
        season_df = season_df.copy()
        season_df['ycards_per_fouls'] = (season_df['yellow_cards'] /
                                          season_df['fouls_committed'])

    team_df    = season_df[season_df['team'] == team_name]
    rest_df    = season_df[season_df['team'] != team_name]
    team_rates = team_df['ycards_per_fouls'].dropna().values
    rest_rates = rest_df['ycards_per_fouls'].dropna().values

    u_stat, p_value = stats.mannwhitneyu(team_rates, rest_rates,
                                         alternative='two-sided')
    n1, n2 = len(team_rates), len(rest_rates)
    r_rb   = float((2 * u_stat) / (n1 * n2) - 1)

    return dict(u_stat=float(u_stat), p_value=float(p_value),
                n1=n1, n2=n2, r_rb=r_rb)


# ---------------------------------------------------------------------------
# Step 7 — full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(team_name, season_df, n_bootstrap=10_000, seed=42):
    """
    Run the complete hypothesis test pipeline for one team.

    Returns
    -------
    dict with keys: stats, proportion, dispersion, autocorr, bootstrap, mw

    Example
    -------
    results = run_pipeline('Juventus', test_df)
    print(results['bootstrap']['p_value'])
    print(results['mw']['r_rb'])
    """
    s = compute_team_stats(team_name, season_df)

    # Compute ycards_per_fouls if not present
    if 'ycards_per_fouls' not in season_df.columns:
        season_df = season_df.copy()
        season_df['ycards_per_fouls'] = (season_df['yellow_cards'] /
                                          season_df['fouls_committed'])

    return dict(
        stats      = s,
        proportion = proportion_tests(s['n'], s['k'], s['p0']),
        dispersion = dispersion_check(s['team_df'], s['p_hat']),
        autocorr   = autocorrelation_check(s['team_df']),
        bootstrap  = bootstrap_test(team_name, season_df, n_bootstrap, seed),
        mw         = mannwhitney_test(team_name, season_df),
    )


# ---------------------------------------------------------------------------
# Step 8 — visualisation
# ---------------------------------------------------------------------------

def plot_bootstrap_grid(teams, season_df, n_bootstrap=10_000, seed=42,
                        ncols=2, palette='steelblue'):
    """
    Plot bootstrap null distributions for multiple teams in a grid.

    Shaded red regions mark the two-tailed rejection area — their combined
    area equals the bootstrap p-value.

    Parameters
    ----------
    teams       : list of str — team names to plot
    season_df   : pd.DataFrame
    n_bootstrap : int
    seed        : int
    ncols       : int — number of columns in the grid (default 2)
    palette     : str — colour for the null distribution bars

    Returns
    -------
    matplotlib.figure.Figure
    """
    nrows = int(np.ceil(len(teams) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    rng = np.random.default_rng(seed)

    for ax, tname in zip(axes, teams):
        boot = bootstrap_test(tname, season_df, n_bootstrap,
                              seed=int(rng.integers(1_000_000_000)))
        null      = boot['null_rates']
        null_mean = boot['null_mean']
        p_hat     = boot['p_hat']
        distance  = boot['distance']
        p_value   = boot['p_value']

        lo = null_mean - distance
        hi = null_mean + distance

        counts, edges, patches = ax.hist(null, bins=60, color=palette,
                                         alpha=0.75, edgecolor='white',
                                         linewidth=0.3)
        for patch, left in zip(patches, edges[:-1]):
            if left <= lo or left >= hi:
                patch.set_facecolor('crimson')
                patch.set_alpha(0.55)

        ax.axvline(p_hat, color='crimson', linewidth=2.2,
                   label=f'Observed ({p_hat:.3f})')
        ax.axvline(null_mean, color='black', linewidth=1.2, linestyle=':',
                   label=f'Null mean ({null_mean:.3f})')

        ax.set_title(f'{tname}   —   p = {p_value:.4f}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Crude rate (yellow cards / fouls)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)

    # hide unused axes
    for ax in axes[len(teams):]:
        ax.set_visible(False)

    fig.suptitle('Bootstrap Null Distributions\nShaded regions = p-value (two-tailed)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
