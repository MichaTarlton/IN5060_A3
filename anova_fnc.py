from __future__ import annotations
from typing import Sequence, Iterable, Optional, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    from scipy import stats
except ImportError as e:  # pragma: no cover
    raise ImportError("scipy is required for KDE estimation. Please install scipy.") from e

def anova(distributions: Sequence[Iterable[float]], 
          group_labels: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform one-way ANOVA (Analysis of Variance).
    
    Parameters
    ----------
    distributions : sequence of array-like or DataFrames
        Each element is a sample representing a group. Can be arrays or DataFrames.
    group_labels : sequence of str, optional
        Labels for each distribution. Defaults to Group_1, Group_2, ...
        
    Returns
    -------
    anova_table : pd.DataFrame
        ANOVA table with columns:
        - Source: Source of variation (Between Groups, Within Groups, Total)
        - SS: Sum of Squares
        - df: Degrees of freedom
        - MS: Mean Square (SS/df)
        - F: F-statistic (only for Between Groups)
        - p_value: p-value (only for Between Groups)
        
    summary_stats : pd.DataFrame
        Summary statistics for each group with columns:
        - Group: Group label
        - N: Sample size
        - Mean: Group mean
        - Std: Standard deviation
        - SE: Standard error
        - 95%_CI_Lower: Lower bound of 95% confidence interval
        - 95%_CI_Upper: Upper bound of 95% confidence interval
    """
    # Convert inputs to numeric arrays
    numeric_dists = []
    final_labels = []
    
    for idx, dist in enumerate(distributions):
        # Determine group label
        if group_labels is not None and idx < len(group_labels):
            label = group_labels[idx]
        else:
            label = f"Group_{idx+1}"
        
        # Check if it's a DataFrame
        if isinstance(dist, pd.DataFrame):
            # Flatten all columns into one array
            arr = dist.values.flatten()
            arr = arr[~np.isnan(arr)]
            numeric_dists.append(arr)
            final_labels.append(label)
        else:
            # Handle as regular array
            arr, _ = _to_numeric_array(dist)
            arr = arr[~np.isnan(arr)]
            numeric_dists.append(arr)
            final_labels.append(label)
    
    k = len(numeric_dists)  # number of groups
    
    # Calculate overall statistics
    all_data = np.concatenate(numeric_dists)
    grand_mean = all_data.mean()
    total_n = len(all_data)
    
    # Calculate Sum of Squares Between Groups (SSA/SSB)
    ssa = 0
    for arr in numeric_dists:
        n_i = len(arr)
        mean_i = arr.mean()
        ssa += n_i * (mean_i - grand_mean) ** 2
    
    # Calculate Sum of Squares Within Groups (SSE/SSW)
    sse = 0
    for arr in numeric_dists:
        mean_i = arr.mean()
        sse += np.sum((arr - mean_i) ** 2)
    
    # Calculate Total Sum of Squares (SST)
    sst = np.sum((all_data - grand_mean) ** 2)
    
    # Degrees of freedom
    df_between = k - 1
    df_within = total_n - k
    df_total = total_n - 1
    
    # Mean Squares
    msa = ssa / df_between  # Mean Square Between
    mse = sse / df_within   # Mean Square Within (Error)
    
    # F-statistic
    f_stat = msa / mse
    
    # p-value
    p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)
    
    # Create ANOVA table
    anova_table = pd.DataFrame({
        'Source': ['Between Groups', 'Within Groups', 'Total'],
        'SS': [ssa, sse, sst],
        'df': [df_between, df_within, df_total],
        'MS': [msa, mse, np.nan],
        'F': [f_stat, np.nan, np.nan],
        'p_value': [p_value, np.nan, np.nan]
    })
    
    # Create summary statistics table
    summary_data = []
    for arr, label in zip(numeric_dists, final_labels):
        n = len(arr)
        mean = arr.mean()
        std = arr.std(ddof=1)
        se = std / np.sqrt(n)
        
        # 95% confidence interval
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
        
        summary_data.append({
            'Group': label,
            'N': n,
            'Mean': mean,
            'Std': std,
            'SE': se,
            '95%_CI_Lower': ci_lower,
            '95%_CI_Upper': ci_upper
        })
    
    summary_stats = pd.DataFrame(summary_data)
    
    return anova_table, summary_stats
