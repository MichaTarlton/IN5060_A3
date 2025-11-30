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
from typing import Sequence, Iterable, Optional, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    from scipy import stats
except ImportError as e:  # pragma: no cover
    raise ImportError("scipy is required for KDE estimation. Please install scipy.") from e


def _is_discrete(data: np.ndarray, max_unique: int = 15) -> bool:
    """Check if data is discrete (few unique integer values).
    
    Parameters
    ----------
    data : ndarray
        Numeric array to check
    max_unique : int
        Maximum number of unique values to consider discrete
        
    Returns
    -------
    bool
        True if data appears discrete (categorical/ordinal)
    """
    if data.size == 0:
        return False
    unique_vals = np.unique(data)
    # Check if all values are integers and there are few unique values
    is_integer = np.allclose(data, np.round(data))
    has_few_values = len(unique_vals) <= max_unique
    return is_integer and has_few_values


def _plot_discrete_distribution(ax, data: np.ndarray, xpos: float, width: float = 0.6,
                                 color='skyblue', edgecolor='black', alpha=0.9,
                                 orientation='horizontal'):
    """Plot a histogram-style distribution for discrete data.
    
    Parameters
    ----------
    ax : matplotlib Axes
    data : ndarray
        Discrete numeric data
    xpos : float
        X position for the distribution center
    width : float
        Maximum width of the distribution
    color : color spec
        Fill color
    edgecolor : color spec
        Edge color
    alpha : float
        Transparency
    orientation : str
        'horizontal' for sideways bars, 'vertical' for upright
    """
    if data.size == 0:
        return
    
    # Count frequencies for each unique value
    unique_vals, counts = np.unique(data, return_counts=True)
    proportions = counts / counts.max()  # normalize to max count
    
    if orientation == 'horizontal':
        # Horizontal bars extending to the right
        for val, prop in zip(unique_vals, proportions):
            bar_width = prop * width
            rect = plt.Rectangle((xpos, val - 0.4), bar_width, 0.8,
                                 facecolor=color, edgecolor=edgecolor,
                                 alpha=alpha, linewidth=1)
            ax.add_patch(rect)
    else:
        # Vertical bars extending upward
        for val, prop in zip(unique_vals, proportions):
            bar_height = prop * width
            rect = plt.Rectangle((val - 0.4, xpos), 0.8, bar_height,
                                 facecolor=color, edgecolor=edgecolor,
                                 alpha=alpha, linewidth=1)
            ax.add_patch(rect)


def _calculate_dispersion(data: np.ndarray, mean: float, variance_type: str, 
                          grand_mean: Optional[float] = None, all_data: Optional[np.ndarray] = None) -> float:
    """Calculate dispersion metric based on variance_type.
    
    Parameters
    ----------
    data : ndarray
        Data array (group data)
    mean : float
        Mean of the group data
    variance_type : str
        Type of dispersion to calculate
    grand_mean : float, optional
        Overall mean across all groups (for ANOVA metrics)
    all_data : ndarray, optional
        All data combined (for ANOVA metrics)
        
    Returns
    -------
    float
        Dispersion value
    """
    if data.size < 2:
        return 0.0
    
    if variance_type == 'sample':
        return data.var(ddof=1)
    elif variance_type == 'population':
        return data.var(ddof=0)
    elif variance_type == 'std':
        return data.var(ddof=1)
    elif variance_type == 'mse':
        # Mean Squared Error (from mean)
        return np.mean((data - mean) ** 2)
    elif variance_type == 'ss':
        # Sum of Squares (within group)
        return np.sum((data - mean) ** 2)
    elif variance_type == 'tukey':
        # Tukey's HSD uses pooled standard error
        # For single group, return standard error
        return data.var(ddof=1) / data.size
    elif variance_type == 'sst':
        # Total Sum of Squares (from grand mean)
        if grand_mean is not None:
            return np.sum((data - grand_mean) ** 2)
        return np.sum((data - mean) ** 2)
    elif variance_type == 'sse':
        # Error/Within Sum of Squares (from group mean)
        return np.sum((data - mean) ** 2)
    elif variance_type == 'sse':
        # Error/Within Sum of Squares (from group mean)
        return np.sum((data - mean) ** 2)
    elif variance_type == 'alt_sse':
        # Error/Within Sum of Squares (from group mean)
        return np.sum((data - mean) ** 2)
    elif variance_type == 'ssa':
        # Among/Between Groups Sum of Squares
        if grand_mean is not None:
            return data.size * (mean - grand_mean) ** 2
        return 0.0
    elif variance_type == 'ftest':
        # F-statistic component (MSA/MSE ratio will be computed at plot level)
        # Return MSE (within-group variance) for now
        return data.var(ddof=1)
    else:
        return data.var(ddof=1)


def _to_numeric_array(data: Iterable[Any]) -> Tuple[np.ndarray, Optional[dict]]:
    """Convert data to numeric array, handling categorical types.
    
    Returns
    -------
    numeric_array : ndarray
        Numeric version of the data
    category_map : dict or None
        Mapping from numeric codes to category labels if categorical, else None
    """
    arr = np.asarray(data)
    
    # Check if categorical (pandas Categorical or object dtype with strings)
    if hasattr(arr, 'dtype'):
        if pd.api.types.is_categorical_dtype(arr):
            # Pandas categorical
            codes = arr.codes.astype(float)
            codes[codes == -1] = np.nan  # handle missing values
            categories = {i: str(cat) for i, cat in enumerate(arr.categories)}
            return codes, categories
        elif arr.dtype == object or pd.api.types.is_string_dtype(arr):
            # Object dtype - try to factorize
            series = pd.Series(arr)
            codes, uniques = pd.factorize(series, sort=True)
            codes_float = codes.astype(float)
            codes_float[codes == -1] = np.nan
            categories = {i: str(val) for i, val in enumerate(uniques)}
            return codes_float, categories
    
    # Numeric data
    return arr.astype(float), None


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
    y_max: Optional[float] = None,
    variance_type: str = 'sample',
    show_dotted_lines: bool = True,
    show_y_annotations: bool = True,
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
    y_max : float, optional
        Force a specific maximum Y-axis value. If None, auto-calculated from data.
    dotted : bool
        Include dotted horizontal lines at means.
    variance_type : str, default 'sample'
        Type of variance/dispersion calculation:
        - 'sample': sample variance (ddof=1, unbiased estimator)
        - 'population': population variance (ddof=0)
        - 'std': standard deviation (ddof=1)
        - 'mse': mean squared error from mean
        - 'ss': sum of squares (total variation within group)
        - 'tukey': Tukey's HSD variance estimate (variance/n)
        - 'sst': Total Sum of Squares (from grand mean)
        - 'sse': Error/Within Sum of Squares (from group means)
        - 'ssa': Among/Between Groups Sum of Squares
        - 'ftest': F-statistic component (within-group variance)
    show_dotted_lines : bool, default True
        Whether to show horizontal dotted lines at mean values.
    show_y_annotations : bool, default True
        Whether to show y-axis text annotations (E(Y), E(Y|X_i), V(Y|X_i)).

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axes.
    """
    if not distributions:
        raise ValueError("distributions cannot be empty")
    
    # Validate variance type
    valid_types = {'sample', 'population', 'std', 'mse', 'ss', 'tukey', 'sst', 'sse', 'ssa', 'ftest'}
    if variance_type not in valid_types:
        raise ValueError(f"variance_type must be one of {valid_types}, got '{variance_type}'")

    # Convert to numeric, handling categorical data
    cond_arrays = []
    category_maps = []
    for dist in distributions:
        numeric_arr, cat_map = _to_numeric_array(dist)
        cond_arrays.append(numeric_arr[~np.isnan(numeric_arr)])
        category_maps.append(cat_map)
    
    if overall is not None:
        overall_numeric, overall_cat_map = _to_numeric_array(overall)
        overall_array = overall_numeric[~np.isnan(overall_numeric)]
    else:
        overall_array = np.concatenate(cond_arrays)
        overall_cat_map = category_maps[0] if category_maps and category_maps[0] else None

    # Calculate means and dispersions using the specified metric
    cond_means = [a.mean() if a.size else np.nan for a in cond_arrays]
    overall_mean = overall_array.mean() if overall_array.size else np.nan
    
    # For ANOVA-type metrics, pass grand mean
    cond_vars = [_calculate_dispersion(a, cond_means[i], variance_type, 
                                       grand_mean=overall_mean, all_data=overall_array) 
                 for i, a in enumerate(cond_arrays)]
    overall_var = _calculate_dispersion(overall_array, overall_mean, variance_type,
                                       grand_mean=overall_mean, all_data=overall_array)

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

    # Check if data is discrete (for categorical/Likert scale data)
    is_discrete = _is_discrete(overall_array) if overall_array.size else False
    
    # Set axis limits based on data type
    if is_discrete:
        # For discrete data, use exact min/max with small padding
        ymin_candidates = [a.min() for a in cond_arrays if a.size]
        ymax_candidates = [a.max() for a in cond_arrays if a.size]
        if overall_array.size:
            ymin_candidates.append(overall_array.min())
            ymax_candidates.append(overall_array.max())
        y_min = min(ymin_candidates) - 0.5 if ymin_candidates else 0.0
        calculated_y_max = max(ymax_candidates) + 0.5 if ymax_candidates else 1.0
    else:
        # For continuous data, use 0 as min with extended max
        y_min = 0.0
        ymax_candidates = [a.max() for a in cond_arrays if a.size]
        if overall_array.size:
            ymax_candidates.append(overall_array.max())
        calculated_y_max = max(ymax_candidates) * 1.05 if ymax_candidates else 1.0
    
    # Use forced y_max if provided, otherwise use calculated value
    if y_max is not None:
        final_y_max = y_max
    else:
        final_y_max = calculated_y_max

    if overall_array.size > 1:
        # Always use histogram bars (discrete distribution style)
        _plot_discrete_distribution(ax, overall_array, xpos=0, width=1.2,
                                   color=overall_color, alpha=0.9, orientation='horizontal')
        
        # Mean and variance indicators (matching conditional distribution style)
        std_o = np.sqrt(overall_var)
        ax.scatter([0], [overall_mean], color='black', s=25, zorder=4)
        ax.plot([0, 0], [overall_mean - std_o, overall_mean + std_o], 'k-', lw=2)
        cap = 0.08
        ax.plot([-cap, cap], [overall_mean - std_o, overall_mean - std_o], 'k-', lw=2)
        ax.plot([-cap, cap], [overall_mean + std_o, overall_mean + std_o], 'k-', lw=2)

    # Add spacing between overall and conditional distributions
    x_positions = np.arange(1, n + 1) + 0.5  # Add 0.5 spacing offset
    for i, a in enumerate(cond_arrays):
        if a.size < 2:
            continue
        
        xpos = x_positions[i]
        
        # Always use histogram bars (discrete distribution style)
        _plot_discrete_distribution(ax, a, xpos=xpos, width=0.6,
                                   color=colors[i], alpha=0.9, orientation='horizontal')
        
        # Mean and variance indicators
        m = cond_means[i]
        v = cond_vars[i]
        std = np.sqrt(v)
        ax.scatter([xpos], [m], color='black', s=25, zorder=4)
        ax.plot([xpos, xpos], [m - std, m + std], 'k-', lw=2)
        cap = 0.08
        ax.plot([xpos - cap, xpos + cap], [m - std, m - std], 'k-', lw=2)
        ax.plot([xpos - cap, xpos + cap], [m + std, m + std], 'k-', lw=2)
        if annotate and show_y_annotations:
            max_width_cond = 0.6
            ax.text(xpos + max_width_cond + 0.05, m, f'$V(Y|X_{i+1})$', va='center', fontsize=9)

    if dotted and show_dotted_lines:
        for m in cond_means:
            if not np.isnan(m):
                ax.axhline(m, color='black', linestyle=':', linewidth=0.8, alpha=0.6)
        if not np.isnan(overall_mean):
            ax.axhline(overall_mean, color='black', linestyle=':', linewidth=1, alpha=0.8)

    for xpos in x_positions:
        ax.axvline(xpos, color='black', linestyle='--', linewidth=1)

    if annotate and show_y_annotations:
        for i, m in enumerate(cond_means):
            if not np.isnan(m):
                ax.text(-0.15, m, f'$E(Y|X_{i+1})$', ha='right', va='center', fontsize=10)
        if not np.isnan(overall_mean):
            ax.text(-0.15, overall_mean, '$E(Y)=E(E(Y|X_i))$', ha='right', va='center', fontsize=10)

    ax.set_xlim(-0.1, x_positions[-1] + 1.2)
    ax.set_ylim(y_min, final_y_max)
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
    y_max: Optional[float] = None,
    variance_type: str = 'sample',
) -> Tuple[plt.Figure, plt.Axes]:
    """Create vertical (rotated) violin plots for selected columns.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Source DataFrame.
    columns : sequence of str
        Columns to plot (numeric or categorical).
    figsize : (w, h)
        Figure size.
    show_mean : bool
        Whether to draw vertical dotted line spanning full Y range.
    show_variance : bool
        Whether to draw variance bar at each mean.
    y_max : float, optional
        Force a specific maximum Y-axis value. If None, auto-calculated from data.
    variance_type : str, default 'sample'
        Type of variance/dispersion calculation:
        - 'sample': sample variance (ddof=1, unbiased estimator)
        - 'population': population variance (ddof=0)
        - 'std': standard deviation (ddof=1)
        - 'mse': mean squared error from mean
        - 'ss': sum of squares (total variation within group)
        - 'tukey': Tukey's HSD variance estimate (variance/n)
        - 'sst': Total Sum of Squares (from grand mean)
        - 'sse': Error/Within Sum of Squares (from group means)
        - 'ssa': Among/Between Groups Sum of Squares
        - 'ftest': F-statistic component (within-group variance)
    """
    # Validate variance type
    valid_types = {'sample', 'population', 'std', 'mse', 'ss', 'tukey', 'sst', 'sse', 'ssa', 'ftest'}
    if variance_type not in valid_types:
        raise ValueError(f"variance_type must be one of {valid_types}, got '{variance_type}'")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Process all specified columns, converting categorical to numeric
    processed_cols = []
    for col in columns:
        if col in data_df.columns:
            processed_cols.append(col)
    
    n_distributions = len(processed_cols)
    x_positions = np.arange(n_distributions)

    for idx, col in enumerate(processed_cols):
        # Convert to numeric, handling categorical
        numeric_data, cat_map = _to_numeric_array(data_df[col])
        data = numeric_data[~np.isnan(numeric_data)]
        
        if data.size < 2:
            continue
        
        # Always use histogram bars (rotated vertical)
        _plot_discrete_distribution(ax, data, xpos=x_positions[idx], width=0.4,
                                   color='skyblue', edgecolor='navy', alpha=0.6, orientation='vertical')
        
        mean_val = data.mean()
        # For ANOVA metrics, we'd need grand mean - using local mean as fallback
        var_val = _calculate_dispersion(data, mean_val, variance_type, grand_mean=mean_val, all_data=data)
        std_val = np.sqrt(abs(var_val))  # abs() to handle SSA which could be small
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

    # Apply y_max if specified
    if y_max is not None:
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], y_max)

    if show_mean:
        y_min, y_max = ax.get_ylim()
        for xpos in x_positions:
            ax.plot([xpos, xpos], [y_min, y_max], 'k--', lw=0.8, alpha=0.6)

    fig.tight_layout()
    return fig, ax


def calculate_statistics(
    distributions: Sequence[Iterable[float]],
    overall: Optional[Iterable[float]] = None,
    group_labels: Optional[Sequence[str]] = None,
    variance_types: Optional[List[str]] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate variance decomposition statistics for distributions.
    
    Parameters
    ----------
    distributions : sequence of array-like
        Each element is a sample representing Y|X_i.
    overall : array-like, optional
        Unconditional Y samples. If None, concatenation of conditional samples.
    group_labels : sequence of str, optional
        Labels for each distribution. Defaults to Group_1, Group_2, ...
    variance_types : list of str, optional
        Types of variance to calculate. If None, calculates all available types.
        Available: 'sample', 'population', 'std', 'mse', 'ss', 'tukey',
                   'sst', 'sse', 'ssa', 'ftest'
    output_csv : str, optional
        Path to save CSV file. If None, doesn't save to file.
        
    Returns
    -------
    pd.DataFrame
        Statistics table with rows for each group and columns for each metric
    """
    # Convert inputs
    numeric_dists = []
    for dist in distributions:
        arr, _ = _to_numeric_array(dist)
        numeric_dists.append(arr[~np.isnan(arr)])
    
    # Overall distribution
    if overall is None:
        overall_data = np.concatenate(numeric_dists)
    else:
        overall_data, _ = _to_numeric_array(overall)
        overall_data = overall_data[~np.isnan(overall_data)]
    
    grand_mean = overall_data.mean()
    n_groups = len(numeric_dists)
    
    # Default group labels
    if group_labels is None:
        group_labels = [f"Group_{i+1}" for i in range(n_groups)]
    
    # Default variance types
    if variance_types is None:
        variance_types = ['sample', 'population', 'std', 'mse', 'ss', 'tukey',
                          'sst', 'sse', 'ssa', 'ftest']
    
    # Calculate statistics
    stats_data = []
    
    for i, (data, label) in enumerate(zip(numeric_dists, group_labels)):
        row = {
            'Group': label,
            'N': len(data),
            'Mean': data.mean(),
        }
        
        mean = data.mean()
        
        # Calculate each variance type
        for vtype in variance_types:
            value = _calculate_dispersion(data, mean, vtype, grand_mean, overall_data)
            row[vtype.upper()] = value
        
        stats_data.append(row)
    
    # Add overall/total row
    overall_row = {
        'Group': 'Overall',
        'N': len(overall_data),
        'Mean': grand_mean,
    }
    
    overall_mean = overall_data.mean()
    for vtype in variance_types:
        value = _calculate_dispersion(overall_data, overall_mean, vtype, grand_mean, overall_data)
        overall_row[vtype.upper()] = value
    
    stats_data.append(overall_row)
    
    # Create DataFrame
    df = pd.DataFrame(stats_data)
    
    # Save to CSV if requested
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        print(f"Statistics saved to: {output_csv}")
    
    return df
