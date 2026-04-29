"""
discipline_stats.py
-------------------
Reusable functions for analysing discipline-related statistics
(fouls, yellow cards, red cards) from team_matches data.

Designed to work with the team-perspective long-format dataframe
where each match generates two rows (one per team).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ===========================================================================
# TestResults class
# ===========================================================================

class TestResults:
    """Container for statistical test results with formatted printing."""

    def __init__(self, title, stats_dict, tests_dict, effect_sizes=None):
        """
        Parameters
        ----------
        title : str
            Description of the comparison (e.g. "Home vs Away - fouls_committed").
        stats_dict : dict
            Descriptive statistics (means, stds, sample sizes, difference).
        tests_dict : dict of dict
            Each key is a test name, value is a dict with at least
            'statistic' (optional) and 'p_value'.
        effect_sizes : dict, optional
            Effect size measures (e.g. {"cohens_d": 0.5, "rank_biserial": 0.1}).
        """
        self.title = title
        self.stats = stats_dict
        self.tests = tests_dict
        self.effect_sizes = effect_sizes or {}

    def __repr__(self):
        return self._format()

    def _format(self, test_names=None):
        lines = []
        w = 50
        lines.append(self.title)
        lines.append("-" * w)

        for k, v in self.stats.items():
            if isinstance(v, float):
                lines.append(f"  {k:25s} {v:+.4f}" if 'diff' in k.lower()
                             else f"  {k:25s} {v:.4f}")
            else:
                lines.append(f"  {k:25s} {v}")

        lines.append("-" * w)

        tests_to_show = test_names if test_names else list(self.tests.keys())
        for name in tests_to_show:
            if name not in self.tests:
                continue
            t = self.tests[name]
            stat_str = f"stat={t['statistic']:.4f}, " if t.get('statistic') is not None else ""
            p_str = f"p={t['p_value']:.4f}"
            sig = "Sig." if t['p_value'] < 0.05 else "Not sig."
            lines.append(f"  {name:25s} {stat_str}{p_str}  ({sig})")

        if self.effect_sizes:
            lines.append("-" * w)
            for k, v in self.effect_sizes.items():
                lines.append(f"  {k:25s} {v:.4f}")

        lines.append("=" * w)
        return "\n".join(lines)

    def show(self, *test_names):
        """Print results for specific tests only."""
        print(self._format(test_names=list(test_names)))

    def to_dict(self):
        """Return all results as a flat dictionary."""
        d = {}
        d.update(self.stats)
        for name, t in self.tests.items():
            if t.get('statistic') is not None:
                d[f"{name}_stat"] = t['statistic']
            d[f"{name}_p"] = t['p_value']
        d.update(self.effect_sizes)
        return d

    def to_frame(self):
        """Return results as a single-row DataFrame."""
        return pd.DataFrame([self.to_dict()])


# ===========================================================================
# Data preparation
# ===========================================================================

def filter_league_season(df, league='Serie_A', season='2526'):
    """
    Filter team_matches for a specific league and/or season.

    Parameters
    ----------
    df : pd.DataFrame
        The team_matches dataframe.
    league : str or None
        League name (e.g. 'Serie_A', 'Premier_League'). None to skip filtering.
    season : str or None
        Season identifier (e.g. '2526'). None to skip filtering.

    Returns
    -------
    pd.DataFrame
        Filtered copy of the dataframe.
    """
    mask = pd.Series(True, index=df.index)
    if league is not None:
        mask &= df['league'] == league
    if season is not None:
        mask &= df['season'] == season
    return df[mask].copy()



# ===========================================================================
# Descriptive statistics
# ===========================================================================

def team_summary(df, column):
    """
    Per-team summary statistics for a given column.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe (single league/season).
    column : str
        Column to summarise (e.g. 'fouls_committed', 'yellow_cards').

    Returns
    -------
    pd.DataFrame
        Summary with total, mean, std, count, diff_from_league.
    """
    result = (
        df.groupby('team')[column]
        .agg(['sum', 'mean', 'std', 'count'])
        .round(2)
        .sort_values('mean', ascending=False)
    )
    result = result.rename(columns={'sum': 'total'})
    league_mean = df[column].mean()
    result['diff_from_league'] = (result['mean'] - league_mean).round(2)
    result.attrs['league_mean'] = league_mean
    return result


def rate_summary(df, numerator='yellow_cards', denominator='fouls_committed'):
    """
    Per-team summary of the rate (numerator / denominator),
    comparing the mean-of-per-match-ratios with the ratio-of-totals.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe (single league/season) with team-level rows.
    numerator : str
        Column for the numerator (e.g. 'yellow_cards').
    denominator : str
        Column for the denominator (e.g. 'fouls_committed').

    Returns
    -------
    pd.DataFrame
        One row per team, sorted by ratio_of_totals.
    """
    col_ratio = f'{numerator}_per_{denominator}'
    df = df.copy()
    df[col_ratio] = df[numerator] / df[denominator]

    agg = (
        df.groupby('team')
        .agg(
            total_denominator=(denominator, 'sum'),
            total_numerator=(numerator, 'sum'),
            matches=('team', 'size'),
            mean_ratio=(col_ratio, 'mean'),
        )
        .reset_index()
    )

    agg['ratio_of_totals'] = agg['total_numerator'] / agg['total_denominator']
    agg['inv_mean_ratio'] = 1 / agg['mean_ratio']
    agg['inv_ratio_of_totals'] = 1 / agg['ratio_of_totals']

    return (
        agg.sort_values(by=['ratio_of_totals', 'team'])
        .reset_index(drop=True)
    )


def venue_summary(df, column):
    """
    Home vs. away descriptive statistics for a given column.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe (single league/season).
    column : str
        Column to summarise.

    Returns
    -------
    pd.DataFrame
        Output of groupby('venue')[column].describe().
    """
    return df.groupby('venue')[column].describe()


# ===========================================================================
# Plots
# ===========================================================================

def plot_distribution(df, column, title=None, color='steelblue', bins=15,
                      figsize=(10, 5), xlabel=None, ylabel='Count',
                      discrete=False, kde=True):
    """
    Plot the overall distribution of a column with mean and median lines.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    title : str, optional
    color : str
    bins : int or sequence
    figsize : tuple
    xlabel : str, optional
    ylabel : str
    discrete : bool
        If True, use discrete=True in histplot.
    kde : bool

    Returns
    -------
    fig, ax
    """
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#f0f0f0')

        sns.histplot(df[column], bins=bins, kde=kde, color=color,
                     alpha=0.7, discrete=discrete, ax=ax)

        mean_val = df[column].mean()
        median_val = df[column].median()
        ax.axvline(mean_val, color='tomato', linestyle='--', linewidth=1.5,
                   label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color='darkorange', linestyle='--', linewidth=1.5,
                   label=f"Median: {median_val:.1f}")

        ax.set_xlabel(xlabel or column.replace('_', ' ').title())
        ax.set_ylabel(ylabel)
        ax.set_title(title or f'Distribution of {column.replace("_", " ")}')
        ax.legend()
        plt.tight_layout()

    return fig, ax


def plot_home_away_distribution(df, column, title=None, bins=15,
                                figsize=(10, 5), xlabel=None, ylabel='Count',
                                home_color='steelblue', away_color='tomato',
                                discrete=False, kde=True):
    """
    Plot overlapping home/away distributions with separate mean lines.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    title : str, optional
    bins : int or sequence
    figsize : tuple
    xlabel : str, optional
    ylabel : str
    home_color : str
    away_color : str
    discrete : bool
    kde : bool

    Returns
    -------
    fig, ax
    """
    home = df[df['venue'] == 'home'][column]
    away = df[df['venue'] == 'away'][column]

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#f0f0f0')

        sns.histplot(home, bins=bins, kde=kde, color=home_color,
                     alpha=0.6, label='Home', discrete=discrete, ax=ax)
        sns.histplot(away, bins=bins, kde=kde, color=away_color,
                     alpha=0.6, label='Away', discrete=discrete, ax=ax)

        ax.axvline(home.mean(), color=home_color, linestyle='--', linewidth=1.5,
                   label=f'Home mean: {home.mean():.1f}')
        ax.axvline(away.mean(), color=away_color, linestyle='--', linewidth=1.5,
                   label=f'Away mean: {away.mean():.1f}')

        ax.set_xlabel(xlabel or column.replace('_', ' ').title())
        ax.set_ylabel(ylabel)
        ax.set_title(title or f'{column.replace("_", " ").title()} - Home vs Away')
        ax.legend()
        plt.tight_layout()

    return fig, ax


def plot_boxplot(df, column, title=None, color='steelblue',
                 figsize=(12, 4), xlabel=None, ylabel='',
                 flier_color='red', strip=True, annotate=True,
                 label_color='#2d6a2d', ax=None):
    """
    Horizontal boxplot of a single column with IQR summary annotations.

    Draws the box, whiskers and fliers, overlays a strip plot of raw points,
    and (optionally) marks lower whisker, Q1, median, Q3 and upper whisker
    with dashed reference lines and rotated labels above the plot.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    title : str, optional
    color : str
        Box fill colour. Default 'steelblue'.
    figsize : tuple
        Ignored when `ax` is provided.
    xlabel : str, optional
        Falls back to the column name title-cased.
    ylabel : str
    flier_color : str
        Colour of outlier markers. Default 'red'.
    strip : bool
        If True, overlay a stripplot of raw observations.
    annotate : bool
        If True, add Q1/median/Q3/whisker reference lines and labels.
    label_color : str
        Colour of the annotation labels.
    ax : matplotlib.axes.Axes, optional
        If provided, plot on this axes instead of creating a new figure.

    Returns
    -------
    fig, ax
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    median = df[column].median()
    lower_whisker = df[column][df[column] >= q1 - 1.5 * iqr].min()
    upper_whisker = df[column][df[column] <= q3 + 1.5 * iqr].max()

    with sns.axes_style("whitegrid"):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.set_facecolor('#f0f0f0')

        sns.boxplot(
            data=df,
            x=column,
            color=color,
            flierprops=dict(marker='o', color=flier_color,
                            markerfacecolor=flier_color, markersize=6),
            ax=ax,
        )
        for patch in ax.patches:
            patch.set_alpha(0.4)

        if strip:
            sns.stripplot(
                data=df,
                x=column,
                color='black',
                size=3,
                alpha=0.2,
                ax=ax,
            )

        if annotate:
            line_props = dict(color='grey', linestyle='--', linewidth=1)
            labels = [
                (lower_whisker, f'Lower whisker: {lower_whisker:.1f}'),
                (q1,            f'Q1: {q1:.1f}'),
                (median,        f'Median: {median:.1f}'),
                (q3,            f'Q3: {q3:.1f}'),
                (upper_whisker, f'Upper whisker: {upper_whisker:.1f}'),
            ]
            for val, label in labels:
                ax.axvline(val, **line_props)
                ax.text(val, 1.02, label, transform=ax.get_xaxis_transform(),
                        fontsize=9, color=label_color,
                        ha='left', va='bottom', rotation=35)

        ax.set_xlabel(xlabel or column.replace('_', ' ').title())
        ax.set_ylabel(ylabel)
        ax.set_title(title or f'Boxplot of {column.replace("_", " ")}',
                     pad=70 if annotate else 10)
        plt.tight_layout()

    return fig, ax


# def plot_league_bar(summary_df, title=None, figsize=(12, 7),
#                     xlabel='Difference from league average',
#                     above_color='tomato', below_color='steelblue'):
#     """
#     Diverging horizontal bar chart of team deviations from league average.

#     Parameters
#     ----------
#     summary_df : pd.DataFrame
#         Output of team_summary(), must have 'diff_from_league' column.
#     title : str, optional
#     figsize : tuple
#     xlabel : str
#     above_color : str
#     below_color : str

#     Returns
#     -------
#     fig, ax
#     """
#     with sns.axes_style("whitegrid"):
#         fig, ax = plt.subplots(figsize=figsize)
#         ax.set_facecolor('#f0f0f0')

#         colors = [above_color if x > 0 else below_color
#                   for x in summary_df['diff_from_league']]

#         ax.barh(summary_df.index, summary_df['diff_from_league'],
#                 color=colors, alpha=0.8)
#         ax.axvline(0, color='black', linewidth=0.8)
#         ax.set_xlabel(xlabel)
#         ax.set_title(title or 'Team deviation from league average')
#         ax.invert_yaxis()
#         plt.tight_layout()

#     return fig, ax

def plot_league_bar(summary_df, title=None, figsize=(12, 7),
                    xlabel='Difference from league average',
                    above_color='tomato', below_color='steelblue',
                    ax=None):
    """
    Diverging horizontal bar chart of team deviations from league average.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of team_summary(), must have 'diff_from_league' column.
    title : str, optional
    figsize : tuple
    xlabel : str
    above_color : str
    below_color : str
    ax : matplotlib.axes.Axes, optional
        If provided, plot on this axes instead of creating a new figure.

    Returns
    -------
    fig, ax
    """
    with sns.axes_style("whitegrid"):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.set_facecolor('#f0f0f0')

        colors = [above_color if x > 0 else below_color
                  for x in summary_df['diff_from_league']]

        ax.barh(summary_df.index, summary_df['diff_from_league'],
                color=colors, alpha=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel(xlabel)

        if title:
            ax.set_title(title)

    return fig, ax



def plot_forest(results_df, df, column, title=None, figsize=(12, 8),
                xlabel='Difference from rest of league',
                above_color='tomato', below_color='steelblue',
                ns_color='grey'):
    """
    Forest plot showing each team's difference from the rest with 95% CI.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of test_all_teams(), must have 'team', 'diff', 'welch_p'.
    df : pd.DataFrame
        The filtered data (needed to compute CIs).
    column : str
        The column being tested.
    title : str, optional
    figsize : tuple
    xlabel : str
    above_color : str
    below_color : str
    ns_color : str
        Color for non-significant teams.

    Returns
    -------
    fig, ax
    """
    from matplotlib.lines import Line2D

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#f0f0f0')

        plot_df = results_df.sort_values('diff', ascending=True).copy()

        ci_data = []
        for _, row in plot_df.iterrows():
            team_data = df[df['team'] == row['team']][column]
            se = team_data.std() / np.sqrt(len(team_data))
            ci_data.append(se * 1.96)
        plot_df['ci'] = ci_data

        colors = [above_color if p < 0.05 and d > 0
                  else below_color if p < 0.05 and d < 0
                  else ns_color
                  for p, d in zip(plot_df['welch_p'], plot_df['diff'])]

        y_pos = range(len(plot_df))

        ax.errorbar(plot_df['diff'], y_pos, xerr=plot_df['ci'],
                    fmt='none', ecolor='black', elinewidth=1, capsize=3, zorder=1)
        ax.scatter(plot_df['diff'], y_pos, c=colors, s=80, zorder=2,
                   edgecolors='black', linewidth=0.5)

        ax.axvline(0, color='black', linewidth=1, linestyle='-')
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(plot_df['team'])
        ax.set_xlabel(xlabel)
        ax.set_title(title or f'Forest plot - {column.replace("_", " ")}')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=above_color,
                   markersize=10, label='Significantly above average'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=below_color,
                   markersize=10, label='Significantly below average'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=ns_color,
                   markersize=10, label='Not significant'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()

    return fig, ax


# ===========================================================================
# Normality checks
# ===========================================================================

def check_normality(df, column, figsize=(16, 5)):
    """
    Run Q-Q plots and Shapiro-Wilk test for overall, home, and away.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
        The Q-Q plot figure.
    results : dict
        Shapiro-Wilk results for each group.
    """
    groups = [
        ('All', df[column]),
        ('Home', df[df['venue'] == 'home'][column]),
        ('Away', df[df['venue'] == 'away'][column]),
    ]

    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, (label, data) in zip(axes, groups):
            ax.set_facecolor('#f0f0f0')
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot - {label}')
        plt.tight_layout()

    results = {}
    for label, data in groups:
        w, p = stats.shapiro(data)
        results[label] = {'W': w, 'p_value': p}
        normal_str = "Normal (p>0.05)" if p > 0.05 else "Not normal (p<0.05)"
        print(f"{label:6s} -- W={w:.4f}, p={p:.6f} {normal_str}")

    return fig, results


# ===========================================================================
# Hypothesis testing — home vs. away
# ===========================================================================

def test_home_away(df, column, n_permutations=10_000, seed=42):
    """
    Compare home vs. away using Levene, Student, Welch, Mann-Whitney,
    and permutation tests.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    n_permutations : int
    seed : int

    Returns
    -------
    TestResults
    """
    np.random.seed(seed)

    home = df[df['venue'] == 'home'][column]
    away = df[df['venue'] == 'away'][column]
    observed_diff = away.mean() - home.mean()

    # Levene
    lev_stat, lev_p = stats.levene(home, away)

    # Student's t-test
    t_s, p_s = stats.ttest_ind(home, away, equal_var=True)

    # Welch's t-test
    t_w, p_w = stats.ttest_ind(home, away, equal_var=False)

    # Mann-Whitney U
    u_stat, u_p = stats.mannwhitneyu(home, away, alternative='two-sided')
    n1, n2 = len(home), len(away)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    # Permutation test
    pooled = np.concatenate([home.values, away.values])
    n_home = len(home)
    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(pooled)
        perm_diffs[i] = pooled[n_home:].mean() - pooled[:n_home].mean()
    p_perm = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    # Cohen's d
    pooled_std = np.sqrt((home.std()**2 + away.std()**2) / 2)
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0

    stats_dict = {
        'home_mean': home.mean(),
        'away_mean': away.mean(),
        'difference (away-home)': observed_diff,
        'home_std': home.std(),
        'away_std': away.std(),
        'home_n': len(home),
        'away_n': len(away),
    }

    tests_dict = {
        'levene': {'statistic': lev_stat, 'p_value': lev_p},
        'student_t': {'statistic': t_s, 'p_value': p_s},
        'welch_t': {'statistic': t_w, 'p_value': p_w},
        'mannwhitney': {'statistic': u_stat, 'p_value': u_p},
        'permutation': {'statistic': None, 'p_value': p_perm},
    }

    effect_sizes = {
        'cohens_d': cohens_d,
        'rank_biserial': rank_biserial,
    }

    return TestResults(
        title=f"Home vs Away -- {column}",
        stats_dict=stats_dict,
        tests_dict=tests_dict,
        effect_sizes=effect_sizes,
    )


# ===========================================================================
# Hypothesis testing — team vs. rest
# ===========================================================================

def test_team_vs_rest(df, column, team_name, n_permutations=10_000, seed=42):
    """
    Compare one team against the rest of the league.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    team_name : str
    n_permutations : int
    seed : int

    Returns
    -------
    TestResults
    """
    np.random.seed(seed)

    team = df[df['team'] == team_name][column]
    rest = df[df['team'] != team_name][column]
    observed_diff = team.mean() - rest.mean()

    # Student's t-test
    t_s, p_s = stats.ttest_ind(team, rest, equal_var=True)

    # Welch's t-test
    t_w, p_w = stats.ttest_ind(team, rest, equal_var=False)

    # Mann-Whitney U
    u_stat, u_p = stats.mannwhitneyu(team, rest, alternative='two-sided')
    n1, n2 = len(team), len(rest)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    # Permutation test
    pooled = np.concatenate([team.values, rest.values])
    n_team = len(team)
    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(pooled)
        perm_diffs[i] = pooled[:n_team].mean() - pooled[n_team:].mean()
    p_perm = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    # Cohen's d
    pooled_std = np.sqrt((team.std()**2 + rest.std()**2) / 2)
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0

    stats_dict = {
        'team': team_name,
        'team_mean': team.mean(),
        'rest_mean': rest.mean(),
        'difference (team-rest)': observed_diff,
        'team_std': team.std(),
        'rest_std': rest.std(),
        'team_n': len(team),
        'rest_n': len(rest),
    }

    tests_dict = {
        'student_t': {'statistic': t_s, 'p_value': p_s},
        'welch_t': {'statistic': t_w, 'p_value': p_w},
        'mannwhitney': {'statistic': u_stat, 'p_value': u_p},
        'permutation': {'statistic': None, 'p_value': p_perm},
    }

    effect_sizes = {
        'cohens_d': cohens_d,
        'rank_biserial': rank_biserial,
    }

    return TestResults(
        title=f"{team_name} vs Rest -- {column}",
        stats_dict=stats_dict,
        tests_dict=tests_dict,
        effect_sizes=effect_sizes,
    )


def test_all_teams(df, column, n_permutations=10_000, seed=42):
    """
    Run test_team_vs_rest for every team in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    n_permutations : int
    seed : int

    Returns
    -------
    pd.DataFrame
        One row per team with means, p-values, and effect sizes.
    """
    teams = df['team'].unique()
    rows = []
    for team_name in teams:
        r = test_team_vs_rest(df, column, team_name,
                              n_permutations=n_permutations, seed=seed)
        d = r.to_dict()
        rows.append(d)

    result = pd.DataFrame(rows).sort_values('team_mean', ascending=False).round(4)
    return result


# ===========================================================================
# Hypothesis testing — head-to-head
# ===========================================================================

def compare_two_teams(df, column, team_a, team_b, n_permutations=10_000, seed=42):
    """
    Compare two specific teams head to head.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    team_a : str
    team_b : str
    n_permutations : int
    seed : int

    Returns
    -------
    TestResults
    """
    np.random.seed(seed)

    a = df[df['team'] == team_a][column]
    b = df[df['team'] == team_b][column]
    diff = a.mean() - b.mean()

    # Welch's t-test
    t_w, p_w = stats.ttest_ind(a, b, equal_var=False)

    # Mann-Whitney U
    u_stat, u_p = stats.mannwhitneyu(a, b, alternative='two-sided')
    n1, n2 = len(a), len(b)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    # Permutation test
    pooled = np.concatenate([a.values, b.values])
    n_a = len(a)
    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(pooled)
        perm_diffs[i] = pooled[:n_a].mean() - pooled[n_a:].mean()
    perm_p = np.mean(np.abs(perm_diffs) >= np.abs(diff))

    # Cohen's d
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0

    stats_dict = {
        'team_a': team_a,
        'team_b': team_b,
        'mean_a': a.mean(),
        'mean_b': b.mean(),
        f'difference ({team_a}-{team_b})': diff,
        'std_a': a.std(),
        'std_b': b.std(),
        'n_a': len(a),
        'n_b': len(b),
    }

    tests_dict = {
        'welch_t': {'statistic': t_w, 'p_value': p_w},
        'mannwhitney': {'statistic': u_stat, 'p_value': u_p},
        'permutation': {'statistic': None, 'p_value': perm_p},
    }

    effect_sizes = {
        'cohens_d': cohens_d,
        'rank_biserial': rank_biserial,
    }

    return TestResults(
        title=f"{team_a} vs {team_b} -- {column}",
        stats_dict=stats_dict,
        tests_dict=tests_dict,
        effect_sizes=effect_sizes,
    )


# ===========================================================================
# Hypothesis testing — general comparisons (two groups, one group vs threshold)
# ===========================================================================


def compare_two_groups(a, b, label_a='Group A', label_b='Group B', n_permutations=10_000, seed=42):
    """
    Compare two independent groups on a numeric variable.

    Parameters
    ----------
    a : array-like
        Values for group A.
    b : array-like
        Values for group B.
    label_a : str
        Display name for group A.
    label_b : str
        Display name for group B.
    n_permutations : int
    seed : int

    Returns
    -------
    TestResults
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    np.random.seed(seed)

    n1, n2 = len(a), len(b)
    diff = a.mean() - b.mean()

    # Student's t-test
    t_s, p_s = stats.ttest_ind(a, b, equal_var=True)

    # Welch's t-test
    t_w, p_w = stats.ttest_ind(a, b, equal_var=False)

    # Mann-Whitney U
    u_stat, u_p = stats.mannwhitneyu(a, b, alternative='two-sided')
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    # Cohen's d
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else np.nan

    # Permutation test
    pooled = np.concatenate([a, b])
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(pooled)
        perm_diff = pooled[:n1].mean() - pooled[n1:].mean()
        if abs(perm_diff) >= abs(diff):
            count += 1
    p_perm = count / n_permutations

    return TestResults(
        f"{label_a} vs {label_b}",
        {
            f'mean_{label_a}': round(a.mean(), 4),
            f'mean_{label_b}': round(b.mean(), 4),
            f'std_{label_a}': round(a.std(ddof=1), 4),
            f'std_{label_b}': round(b.std(ddof=1), 4),
            f'n_{label_a}': n1,
            f'n_{label_b}': n2,
            'mean_diff': round(diff, 4),
        },
        {
            'student_t': {'statistic': round(t_s, 4), 'p_value': round(p_s, 4)},
            'welch_t': {'statistic': round(t_w, 4), 'p_value': round(p_w, 4)},
            'mann_whitney': {'statistic': round(u_stat, 1), 'p_value': round(u_p, 4)},
            'permutation': {'statistic': None, 'p_value': round(p_perm, 4)},
        },
        {
            'cohens_d': round(cohens_d, 4),
            'rank_biserial': round(rank_biserial, 4),
        }
    )


def compare_mean_to_threshold(a, threshold, label='Group', n_bootstrap=10_000, seed=42):
    """
    Test whether a group's mean differs from a known threshold (one-sample tests).

    Parameters
    ----------
    a : array-like
        Observed values.
    threshold : float
        The reference value to test against (no uncertainty).
    label : str
        Display name for the group.
    n_bootstrap : int
    seed : int

    Returns
    -------
    TestResults
    """
    a = np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)

    n = len(a)
    obs_mean = a.mean()
    diff = obs_mean - threshold

    # One-sample t-test
    t_stat, p_t = stats.ttest_1samp(a, threshold)

    # Wilcoxon signed-rank test
    try:
        w_stat, p_w = stats.wilcoxon(a - threshold, alternative='two-sided')
    except ValueError:
        w_stat, p_w = np.nan, 1.0

    # Cohen's d (one-sample)
    cohens_d = diff / a.std(ddof=1) if a.std(ddof=1) > 0 else np.nan

    # Bootstrap test
    centred = a - obs_mean + threshold
    count = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(centred, size=n, replace=True)
        if abs(sample.mean() - threshold) >= abs(diff):
            count += 1
    p_boot = count / n_bootstrap

    return TestResults(
        f"{label} vs threshold ({threshold})",
        {
            'observed_mean': round(obs_mean, 4),
            'threshold': threshold,
            'n': n,
            'mean_diff': round(diff, 4),
        },
        {
            'one_sample_t': {'statistic': round(t_stat, 4), 'p_value': round(p_t, 4)},
            'wilcoxon': {'statistic': round(float(w_stat), 1) if not np.isnan(w_stat) else None, 'p_value': round(float(p_w), 4)},
            'bootstrap': {'statistic': None, 'p_value': round(p_boot, 4)},
        },
        {
            'cohens_d': round(cohens_d, 4),
        }
    )
