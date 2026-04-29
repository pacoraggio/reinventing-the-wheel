import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


_DISCIPLINE_COL_LABELS = {
    'HF':  'Home Fouls Committed',
    'AF':  'Away Fouls Committed',
    'HY':  'Home Yellow Cards',
    'AY':  'Away Yellow Cards',
    'HR':  'Home Red Cards',
    'AR':  'Away Red Cards',
}

_VIOLIN_COLS = ['HF', 'AF', 'HY', 'AY']
_BAR_COLS    = ['HR', 'AR']
_CARD_ORDER  = ['0', '1', '2+']


def plot_discipline_distributions(plot_data, league_order=None, palette='Set2'):
    """
    Produces two figures for discipline data exploration:
      - Violin plots for fouls (HF, AF) and yellow cards (HY, AY)
      - Bar charts for red cards (HR, AR), bucketed into 0 / 1 / 2+

    Red cards are treated separately because their distribution is heavily
    concentrated at zero, making violin plots uninformative for them.

    Parameters
    ----------
    plot_data : pd.DataFrame
        Combined dataframe with a 'League' column and discipline columns
        (HF, AF, HY, AY, HR, AR). Build it with:
            pd.concat([df.assign(League=name) for name, df in league_dfs.items()])
    league_order : list, optional
        Display order of leagues. Defaults to the order found in the data.
    palette : str
        Seaborn palette name. Default 'Set2'.

    Returns
    -------
    fig_violin : matplotlib.figure.Figure
    fig_bar    : matplotlib.figure.Figure
    """
    if league_order is None:
        league_order = list(plot_data['League'].unique())

    league_palette = dict(zip(league_order, sns.color_palette(palette, len(league_order))))

    # --- Violin plots: fouls and yellow cards ---
    fig_violin, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(_VIOLIN_COLS):
        overall_median = plot_data[col].median()
        sns.violinplot(
            data=plot_data,
            x='League', y=col,
            hue='League',
            order=league_order,
            hue_order=league_order,
            ax=axes[i],
            palette=league_palette,
            legend=False,
            inner='box',
            cut=0,
        )
        axes[i].axhline(overall_median, color='black', linestyle='--',
                        linewidth=1.0, alpha=0.5)
        axes[i].set_title(_DISCIPLINE_COL_LABELS[col], fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', rotation=30)

    fig_violin.suptitle('Fouls and Yellow Cards Distribution by League',
                        fontsize=16, fontweight='bold')
    fig_violin.tight_layout()

    # --- Bar charts: red cards ---
    fig_bar, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, col in enumerate(_BAR_COLS):
        bucketed = (
            plot_data[['League', col]]
            .copy()
            .assign(**{col: plot_data[col].clip(upper=2).astype(int)})
        )
        counts = (
            bucketed.groupby(['League', col])
            .size()
            .reset_index(name='count')
        )
        counts['pct'] = counts.groupby('League')['count'].transform(
            lambda x: x / x.sum() * 100
        )
        counts[col] = counts[col].astype(str).replace({'2': '2+'})

        sns.barplot(
            data=counts,
            x=col, y='pct',
            hue='League',
            hue_order=league_order,
            order=_CARD_ORDER,
            palette=league_palette,
            ax=axes[i],
        )
        axes[i].set_title(_DISCIPLINE_COL_LABELS[col], fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Red Cards')
        axes[i].set_ylabel('% of Matches')
        axes[i].legend(title='League', fontsize=9)

    fig_bar.suptitle('Red Cards Distribution by League',
                     fontsize=16, fontweight='bold')
    fig_bar.tight_layout()

    return fig_violin, fig_bar


def grey_missing(ax, pivot, teams, seasons):
    """
    Overlay grey rectangles on heatmap cells where pivot data is missing.

    Call this after `sns.heatmap(...)` to fill NaN cells with a neutral grey,
    distinguishing genuinely missing data (team not in league that season)
    from low values that might otherwise blend into the colormap.

    Parameters
    ----------
    ax      : matplotlib.axes.Axes — the heatmap axes
    pivot   : pd.DataFrame         — the pivot table passed to sns.heatmap
    teams   : list of str          — row labels (must match pivot index order)
    seasons : list of str          — column labels (must match pivot column order,
                                     i.e. readable season labels after renaming)
    """
    for i, team in enumerate(teams):
        for j, season in enumerate(seasons):
            if np.isnan(pivot.loc[team, season]):
                ax.add_patch(
                    plt.Rectangle((j, i), 1, 1, fill=True, color='#cccccc', lw=0)
                )