"""Plotting utilities for variance decomposition style figures.

This module provides two functions:
- variance_decomposition_plot: creates a figure resembling a variance decomposition
  diagram with an overall distribution and conditional half-violins for Y|X_i.
- plot_horizontal_violins: creates rotated (vertical) violin plots for numeric
  columns in a DataFrame.

Both functions return (fig, ax) and do not call plt.show(), allowing callers to
further customize or save figures.
"""
from __future__ import annotations
from typing import Sequence, Iterable, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import stats
except ImportError as e:  # pragma: no cover
    raise ImportError("scipy is required for KDE estimation. Please install scipy.") from e


def variance_decomposition_plot(
    distributions: Sequence[Iterable[float]],
    overall: Optional[Iterable[float]] = None,
    x_labels: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (14, 6),
    colors: Optional[Sequence] = None,
    overall_color = (1.0, 0.8, 0.0),
    cond_cmap = None,
    bandwidth_adjust: float = 1.0,
    arrow: bool = True,
    annotate: bool = True,
    dotted: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a variance decomposition style figure.

    Parameters
    ----------
    distributions : sequence of array-like
        Each element is a sample representing Y|X_i.
    overall : array-like, optional
        Unconditional Y samples. If None, concatenation of conditional samples.
    x_labels : sequence of str, optional
        Labels for X positions. Defaults to X1, X2, ...
    figsize : (w, h)
        Figure size in inches.
    colors : sequence of colors, optional
        Colors for conditional half violins.
    overall_color : color
        Color for overall left distribution.
    cond_cmap : matplotlib colormap, optional
        Colormap used if explicit colors not provided.
    bandwidth_adjust : float
        Factor to adjust KDE bandwidth ( >1 wider, <1 narrower ).
    arrow : bool
        Draw double headed arrow for overall variance span.
    annotate : bool
        Include textual annotations (E(Y|X_i), V(Y|X_i), etc.).
    dotted : bool
        Include dotted horizontal lines at means.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axes.
    """
    if not distributions:
        raise ValueError("distributions cannot be empty")

    cond_arrays = [np.asarray(a, dtype=float)[~np.isnan(a)] for a in distributions]
    overall_array = (np.asarray(list(overall), dtype=float)[~np.isnan(overall)]
                     if overall is not None else np.concatenate(cond_arrays))

    cond_means = [a.mean() if a.size else np.nan for a in cond_arrays]
    cond_vars = [a.var(ddof=1) if a.size > 1 else 0.0 for a in cond_arrays]
    overall_mean = overall_array.mean() if overall_array.size else np.nan
    overall_var = overall_array.var(ddof=1) if overall_array.size > 1 else 0.0

    n = len(cond_arrays)
    if x_labels is None:
        x_labels = [f'$X_{i+1}$' for i in range(n)]

    if colors is None:
        if cond_cmap is None:
            base = plt.cm.Blues
            colors = [base(0.35 + 0.5 * i / max(n - 1, 1)) for i in range(n)]
        else:
            colors = [cond_cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)

    y_min = 0.0
    ymax_candidates = [a.max() for a in cond_arrays if a.size]
    if overall_array.size:
        ymax_candidates.append(overall_array.max())
    # y_max = max(ymax_candidates) * 1.05 if ymax_candidates else 1.0
    y_max = max(ymax_candidates) 

    if overall_array.size > 1:
        kde_overall = stats.gaussian_kde(overall_array)
        kde_overall.set_bandwidth(kde_overall.factor * bandwidth_adjust)
        y_grid = np.linspace(y_min, y_max, 600)
        dens = kde_overall(y_grid)
        dens_scaled = dens / dens.max() * 1.2
        ax.fill_betweenx(y_grid, 0, dens_scaled, color=overall_color, alpha=0.9, edgecolor='black', linewidth=1)
        ax.plot([0, dens_scaled.max() * 0.05], [overall_mean, overall_mean], 'k-', lw=2)
        std_o = np.sqrt(overall_var)
        ax.plot([dens_scaled.max() * 0.025, dens_scaled.max() * 0.025], [overall_mean - std_o, overall_mean + std_o], 'k-', lw=2)
        if arrow and std_o > 0:
            ax.annotate('', xy=(dens_scaled.max() * 0.025, overall_mean + std_o),
                        xytext=(dens_scaled.max() * 0.025, overall_mean - std_o),
                        arrowprops=dict(arrowstyle='<->', lw=1.5, color='k'))

    x_positions = np.arange(1, n + 1)
    for i, a in enumerate(cond_arrays):
        if a.size < 2:
            continue
        kde = stats.gaussian_kde(a)
        kde.set_bandwidth(kde.factor * bandwidth_adjust)
        y_grid = np.linspace(y_min, y_max, 400)
        dens = kde(y_grid)
        dens_scaled = dens / dens.max() * 0.6
        xpos = x_positions[i]
        ax.fill_betweenx(y_grid, xpos, xpos + dens_scaled, color=colors[i], alpha=0.9, edgecolor='black', linewidth=1)
        m = cond_means[i]
        v = cond_vars[i]
        std = np.sqrt(v)
        ax.scatter([xpos], [m], color='black', s=25, zorder=4)
        ax.plot([xpos, xpos], [m - std, m + std], 'k-', lw=2)
        cap = 0.08
        ax.plot([xpos - cap, xpos + cap], [m - std, m - std], 'k-', lw=2)
        ax.plot([xpos - cap, xpos + cap], [m + std, m + std], 'k-', lw=2)
        if annotate:
            ax.text(xpos + dens_scaled.max() + 0.05, m, f'$V(Y|X_{i+1})$', va='center', fontsize=9)

    if dotted:
        for m in cond_means:
            if not np.isnan(m):
                ax.axhline(m, color='black', linestyle=':', linewidth=0.8, alpha=0.6)
        if not np.isnan(overall_mean):
            ax.axhline(overall_mean, color='black', linestyle=':', linewidth=1, alpha=0.8)

    for xpos in x_positions:
        ax.axvline(xpos, color='black', linestyle='--', linewidth=1)

    if annotate:
        for i, m in enumerate(cond_means):
            if not np.isnan(m):
                ax.text(-0.15, m, f'$E(Y|X_{i+1})$', ha='right', va='center', fontsize=10)
        if not np.isnan(overall_mean):
            ax.text(-0.15, overall_mean, '$E(Y)=E(E(Y|X_i))$', ha='right', va='center', fontsize=10)

    ax.set_xlim(-0.1, x_positions[-1] + 1.2)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel('$X$', fontsize=14)
    ax.set_ylabel('$Y$', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_horizontal_violins(
    data_df,
    columns: Sequence[str],
    figsize: Tuple[float, float] = (12, 10),
    show_mean: bool = True,
    show_variance: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create vertical (rotated) violin plots for selected numeric columns.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Source DataFrame.
    columns : sequence of str
        Columns to plot (must be numeric).
    figsize : (w, h)
        Figure size.
    show_mean : bool
        Whether to draw vertical dotted line spanning full Y range.
    show_variance : bool
        Whether to draw variance bar at each mean.
    """
    import pandas as pd  # local import to avoid hard dependency at module import time

    fig, ax = plt.subplots(figsize=figsize)
    numeric_cols = [c for c in columns if c in data_df.select_dtypes(include='number').columns]
    n_distributions = len(numeric_cols)
    x_positions = np.arange(n_distributions)

    for idx, col in enumerate(numeric_cols):
        data = data_df[col].dropna().to_numpy(dtype=float)
        if data.size < 2:
            continue
        density = stats.gaussian_kde(data)
        y_range = np.linspace(0 if data.min() >= 0 else data.min() - 0.5, data.max() + 0.5, 300)
        x_density = density(y_range)
        x_density_scaled = x_density / x_density.max() * 0.4
        ax.fill_between(x_positions[idx] + x_density_scaled, y_range, x_positions[idx],
                        alpha=0.6, color='skyblue', edgecolor='navy', linewidth=1.2)
        mean_val = data.mean()
        var_val = data.var(ddof=1)
        std_val = np.sqrt(var_val)
        if show_mean:
            # after first iteration y-limits may not be final; postpone vertical line until end
            pass
        if show_variance:
            ax.plot([x_positions[idx] - 0.15, x_positions[idx] + 0.15], [mean_val, mean_val], 'k-', lw=2.0)
            ax.plot([x_positions[idx], x_positions[idx]], [mean_val - std_val, mean_val + std_val], 'k-', lw=1.5)
            ax.plot([x_positions[idx] - 0.08, x_positions[idx] + 0.08], [mean_val - std_val, mean_val - std_val], 'k-', lw=1.5)
            ax.plot([x_positions[idx] - 0.08, x_positions[idx] + 0.08], [mean_val + std_val, mean_val + std_val], 'k-', lw=1.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'$X_{{{i+1}}}$' for i in range(n_distributions)], fontsize=11)
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_mean:
        y_min, y_max = ax.get_ylim()
        for xpos in x_positions:
            ax.plot([xpos, xpos], [y_min, y_max], 'k--', lw=0.8, alpha=0.6)

    fig.tight_layout()
    return fig, ax
