from __future__ import annotations
from typing import Sequence, Iterable, Optional, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    from scipy import stats
except ImportError as e:  # pragma: no cover
    raise ImportError("scipy is required for KDE estimation. Please install scipy.") from e


def calculate_statistics(
    distributions: Sequence[Iterable[float]],
    overall: Optional[Iterable[float]] = None,
    group_labels: Optional[Sequence[str]] = None,
    subgroup_labels: Optional[Sequence[str]] = None,
    variance_types: Optional[List[str]] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate variance decomposition statistics for distributions.
    
    Parameters
    ----------
    distributions : sequence of array-like or DataFrames
        Each element is a sample representing Y|X_i. Can be arrays or DataFrames with multiple columns.
    overall : array-like, optional
        Unconditional Y samples. If None, concatenation of conditional samples.
    group_labels : sequence of str, optional
        Labels for each distribution (e.g., ['Male', 'Female']). 
        If distributions are DataFrames, these label each DataFrame's group.
        Defaults to Group_1, Group_2, ...
    variance_types : list of str, optional
        Types of variance to calculate. If None, calculates all available types.
        Available: 'sample', 'population', 'std', 'mse', 'ss', 'tukey',
                   'sst', 'sse', 'ssa', 'ftest'
    output_csv : str, optional
        Path to save CSV file. If None, doesn't save to file.
        
    Returns
    -------
    pd.DataFrame
        Statistics table with rows for each subgroup and columns including 'Group' (main label),
        'Subgroup' (column name), and each requested variance metric
    """
    # Convert inputs - handle DataFrames with multiple columns
    numeric_dists = []
    main_groups = []
    subgroup_names = []
    
    for idx, dist in enumerate(distributions):
        # Determine main group label
        if group_labels is not None and idx < len(group_labels):
            main_label = group_labels[idx]
        else:
            main_label = f"Group_{idx+1}"
        if subgroup_labels is not None and idx < len(subgroup_labels):
            sub_label = subgroup_labels[idx]
        else:
            sub_label = f"Group_{idx+1}"
        
        # Check if it's a DataFrame with multiple columns
        if isinstance(dist, pd.DataFrame):
            # Process each column as a separate distribution
            for col in dist.columns:
                arr, _ = _to_numeric_array(dist[col])
                numeric_dists.append(arr[~np.isnan(arr)])
                main_groups.append(main_label)
                subgroup_names.append(sub_label)
        else:
            # Handle as regular array
            arr, _ = _to_numeric_array(dist)
            numeric_dists.append(arr[~np.isnan(arr)])
            main_groups.append(main_label)
            subgroup_names.append('all')
    
    # Overall distribution
    if overall is None:
        overall_data = np.concatenate(numeric_dists)
    else:
        overall_data, _ = _to_numeric_array(overall)
        overall_data = overall_data[~np.isnan(overall_data)]
    
    grand_mean = overall_data.mean()
    k_length = len(numeric_dists)
    
    # Default variance types
    if variance_types is None:
        variance_types = ['sample', 'population', 'std', 'mse', 'ss', 'tukey',
                          'sst', 'sse', 'ssa', 'ftest']
    
    # Calculate statistics
    stats_data = []
    
    for data, main_group, subgroup in zip(numeric_dists, main_groups, subgroup_names):
        row = {
            'Group': main_group,
            'Subgroup': subgroup,
            'N': len(data),
            'Mean': data.mean(),
        }
        
        mean = data.mean()
        # comparison_data = numeric_dists[]
        # Calculate each variance type
        for vtype in variance_types:
            value = _calculate_dispersion(data, mean, vtype, k_length, grand_mean, overall_data)
            row[vtype.upper()] = value
        
        stats_data.append(row)
    
    # Add overall/total row
    overall_row = {
        'Group': 'Overall',
        'Subgroup': 'all',
        'N': len(overall_data),
        'Mean': grand_mean,
    }
    
    overall_mean = overall_data.mean()
    for vtype in variance_types:
        value = _calculate_dispersion(overall_data, overall_mean, vtype, k_length, grand_mean, overall_data)
        overall_row[vtype.upper()] = value
    
    stats_data.append(overall_row)
    
    # Create DataFrame
    df = pd.DataFrame(stats_data)
    
    # Save to CSV if requested
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        print(f"Statistics saved to: {output_csv}")
    
    return df


def _calculate_dispersion(data: np.ndarray, mean: float,            
                          variance_type: str, 
                          k_value,
                          grand_mean: Optional[float] = None, all_data: Optional[np.ndarray] = None
                          ) -> float:
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
    n = data.size

    if data.size < 2:
        return 0.0
    
    if variance_type == 'sample':
        return data.var(ddof=1)
    elif variance_type == 'population':
        return data.var(ddof=0)
    elif variance_type == 'std':
        return data.std(ddof=1)
    elif variance_type == 'mse':
        # Mean Squared Error (from mean) Same as variance?
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
    elif variance_type == 'alt_sse':
        # Error/Within Sum of Squares (from group mean)
        return np.sum((data - mean) ** 2)
    elif variance_type == 'ssa':
        # Among/Between Groups Sum of Squares
        if grand_mean is not None:
            return data.size * (mean - grand_mean) ** 2
        return 0.0
    elif variance_type == 'alt_ssa':
        # Among/Between Groups Sum of Squares
        if grand_mean is not None:
            return data.size * (mean - grand_mean) ** 2
        return 0.0
    elif variance_type == 'alt_ftest':
        # F = MSA/MSE where MSA = SSA/df_between, MSE = SSE/df_within
        SSA = n * (mean - grand_mean) ** 2
        SSE = np.sum((data - mean) ** 2)
        # these are assuming the above is correct SSA and SSE
        sigma_a = SSA / (k_value-1)
        sigma_e = SSE / k_value*(n-1)

        return sigma_a / sigma_e
    elif variance_type == 'old_ftest':
        # F-statistic component (MSA/MSE ratio will be computed at plot level)
        # Return MSE (within-group variance) for now
        return data.var(ddof=1)
    else:
        return data.var(ddof=1)


# Support Funcs --------------------------------------------

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
    # Check if all values are integers and there are few unique v
    is_integer = np.allclose(data, np.round(data))
    has_few_values = len(unique_vals) <= max_unique
    return is_integer and has_few_values



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


def tukey_hsd(distributions: Sequence[Iterable[float]], 
              group_labels: Optional[Sequence[str]] = None,
              column_label: Optional[str] = None,
              alpha: float = 0.05) -> pd.DataFrame:
    """Perform Tukey's Honest Significant Difference (HSD) test for pairwise comparisons.
    
    Parameters
    ----------
    distributions : sequence of array-like or DataFrames
        Each element is a sample representing a group. Can be arrays or DataFrames.
    group_labels : sequence of str, optional
        Labels for each distribution. Defaults to Group_1, Group_2, ...
    column_label : str, optional
        Label for the column/metric being analyzed (e.g., 'Difficulty', 'Control').
        Will be added to output table.
    alpha : float, default 0.05
        Significance level for confidence intervals
        
    Returns
    -------
    pd.DataFrame
        Pairwise comparison results with columns:
        - Group1: First group in comparison
        - Group2: Second group in comparison
        - Mean_Diff: Difference in means (Group1 - Group2)
        - SE: Standard error of the difference
        - Q_stat: Studentized range statistic
        - Q_crit: Critical value at alpha level
        - p_value: Approximate p-value
        - Significant: Whether difference is significant at alpha level
        - Lower_CI: Lower bound of confidence interval
        - Upper_CI: Upper bound of confidence interval
    """
    # Convert inputs to numeric arrays
    numeric_dists = []
    final_labels = []
    # Extract column names for subgroup tracking
    
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
    
    # Calculate pooled MSE (Mean Squared Error within groups)
    total_n = sum(len(arr) for arr in numeric_dists)
    df_within = total_n - k
    
    # Calculate MSE (pooled within-group variance)
    sse_total = sum(np.sum((arr - arr.mean()) ** 2) for arr in numeric_dists)
    mse = sse_total / df_within
    
    # Calculate group means and sample sizes
    means = np.array([arr.mean() for arr in numeric_dists])
    ns = np.array([len(arr) for arr in numeric_dists])
    
    # Results list
    results = []
    
    # First: Perform all pairwise comparisons between groups
    for i in range(k):
        for j in range(i + 1, k):
            # Calculate difference in means
            mean_diff = means[i] - means[j]
            
            # Calculate standard error
            se = np.sqrt(mse * (1/ns[i] + 1/ns[j]))
            
            # Calculate studentized range statistic (q)
            q_stat = abs(mean_diff) / (np.sqrt(mse / ((ns[i] + ns[j]) / 2)))
            
            # Get critical value from studentized range distribution
            # Using scipy.stats for critical value
            q_crit = stats.studentized_range.ppf(1 - alpha, k, df_within)
            
            # Calculate confidence interval
            margin = q_crit * se / np.sqrt(2)  # HSD uses q/sqrt(2) for CI
            lower_ci = mean_diff - margin
            upper_ci = mean_diff + margin
            
            # Determine significance
            significant = abs(q_stat) > q_crit
            
            # Approximate p-value
            p_value = 1 - stats.studentized_range.cdf(abs(q_stat), k, df_within)
            
            # Individual group deviations from grand mean
            group1_deviation = means[i] - grand_mean
            group2_deviation = means[j] - grand_mean
            
            # Individual group contributions to SSA
            group1_ssa = ns[i] * (means[i] - grand_mean) ** 2
            group2_ssa = ns[j] * (means[j] - grand_mean) ** 2
            
            # Effect sizes
            pooled_std = np.sqrt(mse)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            row_data = {
                'Comparison_Type': 'Pairwise',
                'Group1': final_labels[i],
                'Group2': final_labels[j],
                'k': k,
                'n': ns,
                'Mean1': means[i],
                'Mean2': means[j],
                'Grand_Mean': grand_mean,
                'Group1_Deviation': group1_deviation,
                'Group2_Deviation': group2_deviation,
                'Mean_Diff': mean_diff,
                'Group1_SSA': group1_ssa,
                'Group2_SSA': group2_ssa,
                'MSE_pooled': mse,
                'SE': se,
                'Q_stat': q_stat,
                'Q_crit': q_crit,
                'p_value': p_value,
                'Cohens_d': cohens_d,
                'Significant': significant,
                'Lower_CI': lower_ci,
                'Upper_CI': upper_ci
            }
            
            # Add column label if provided
            if column_label is not None:
                row_data = {'Metric': column_label, **row_data}
            
            results.append(row_data)
    
    # Second: Each group vs pooled (grand mean)
    for i in range(k):
        # Comparison to grand mean
        mean_diff = means[i] - grand_mean
        
        # Standard error for comparing to grand mean
        se = np.sqrt(mse / ns[i])
        
        # t-statistic for comparison to grand mean
        t_stat = mean_diff / se if se > 0 else 0
        
        # Use t-distribution for single group vs grand mean
        t_crit = stats.t.ppf(1 - alpha/2, df_within)
        
        # Confidence interval
        lower_ci = mean_diff - t_crit * se
        upper_ci = mean_diff + t_crit * se
        
        # p-value (two-tailed)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_within))
        
        # Determine significance
        significant = abs(t_stat) > t_crit
        
        # SSA for this group
        group_ssa = ns[i] * (means[i] - grand_mean) ** 2
        
        # Effect size
        pooled_std = np.sqrt(mse)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        row_data = {
            'Comparison_Type': 'vs_Pooled',
            'Group1': final_labels[i],
            'Group2': 'Pooled',
            'k': k,
            'n': ns,
            'Mean1': means[i],
            'Mean2': grand_mean,
            'Grand_Mean': grand_mean,
            'Group1_Deviation': mean_diff,
            'Group2_Deviation': 0.0,
            'Mean_Diff': mean_diff,
            'Group1_SSA': group_ssa,
            'Group2_SSA': 0.0,
            'MSE_pooled': mse,
            'SE': se,
            'Q_stat': t_stat,  # Using t_stat for vs pooled comparison
            'Q_crit': t_crit,
            'p_value': p_value,
            'Cohens_d': cohens_d,
            'Significant': significant,
            'Lower_CI': lower_ci,
            'Upper_CI': upper_ci
        }
        
        # Add column label if provided
        if column_label is not None:
            row_data = {'Metric': column_label, **row_data}
        
        results.append(row_data)
    
    return pd.DataFrame(results)


def anova(distributions: Sequence[Iterable[float]], 
          group_labels: Optional[Sequence[str]] = None,
          column_label: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform one-way ANOVA (Analysis of Variance).
    
    Parameters
    ----------
    distributions : sequence of array-like or DataFrames
        Each element is a sample representing a group. Can be arrays or DataFrames.
    group_labels : sequence of str, optional
        Labels for each distribution. Defaults to Group_1, Group_2, ...
    column_label : str, optional
        Label for the column/metric being analyzed (e.g., 'Difficulty', 'Control').
        Will be added to all output tables.
        
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
    
    #-------
    # 95% confidence interval for the mean
    # grand_t_crit = stats.t.ppf(0.975, total_n - 1)
    grand_ci_lower = grand_mean - t_crit * sst
    grand_ci_upper = grand_mean + t_crit * sst
    
    # 95% confidence interval for the deviation from grand mean
    # Using pooled MSE for standard error of deviation
    se_deviation = np.sqrt(mse / n)
    t_crit_pooled = stats.t.ppf(0.975, df_within)
    deviation_ci_lower = deviation_from_grand - t_crit_pooled * se_deviation
    deviation_ci_upper = deviation_from_grand + t_crit_pooled * se_deviation

    # Create ANOVA table
    anova_table_data = {
        'Source': ['Between Groups', 'Within Groups', 'Total'],
        'k': k,
        'n_total': total_n,
        'SS': [ssa, sse, sst],
        'df': [df_between, df_within, df_total],
        'MS': [msa, mse, np.nan],
        'F': [f_stat, np.nan, np.nan],
        'p_value': [p_value, np.nan, np.nan]
    }
    
    # Add column label if provided
    if column_label is not None:
        anova_table_data = {'Metric': [column_label] * 3, **anova_table_data}
    
    anova_table = pd.DataFrame(anova_table_data)
    
    # Create summary statistics table
    summary_data = []
    for arr, label in zip(numeric_dists, final_labels):
        n = len(arr)
        mean = arr.mean()
        std = arr.std(ddof=1)
        se = std / np.sqrt(n)
        
        # Deviation from grand mean
        deviation_from_grand = mean - grand_mean
        
        # Effect size (Cohen's d relative to grand mean)
        pooled_std = np.sqrt(mse)
        effect_size = deviation_from_grand / pooled_std if pooled_std > 0 else 0
        
        # Individual group contribution to between-group variance
        group_ssa = n * (mean - grand_mean) ** 2  # This group's SS between
        group_msa = group_ssa / 1  # MS for this single group comparison to grand mean
        
        # Individual group's within variance
        group_sse = np.sum((arr - mean) ** 2)
        group_mse = group_sse / (n - 1) if n > 1 else 0
        
        # F-statistic for this group vs pooled error
        # Using group's MSA and pooled MSE
        group_f = group_msa / mse if mse > 0 else 0
        
        # p-value for individual group F-statistic
        # df_between = 1 (one group vs grand mean), df_within from pooled error
        group_p_value = 1 - stats.f.cdf(group_f, 1, df_within) if group_f > 0 else 1.0
        
        # 95% confidence interval for the mean
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
        
        # 95% confidence interval for the deviation from grand mean
        # Using pooled MSE for standard error of deviation
        se_deviation = np.sqrt(mse / n)
        t_crit_pooled = stats.t.ppf(0.975, df_within)
        deviation_ci_lower = deviation_from_grand - t_crit_pooled * se_deviation
        deviation_ci_upper = deviation_from_grand + t_crit_pooled * se_deviation
        
        row_data = {
            'Group': label,
            'k': k,
            'N': n,
            'Mean': mean,
            'Grand_Mean': grand_mean,
            'Deviation': deviation_from_grand,
            'SSA': group_ssa,
            'MSA': group_msa,
            'SSE': group_sse,
            'MSE_group': group_mse,
            'MSE_pooled': mse,
            'F_stat': group_f,
            'p_value': group_p_value,
            'Effect_Size': effect_size,
            'Std': std,
            'SE': se,
            '95%_CI_Lower': ci_lower,
            '95%_CI_Upper': ci_upper,
            'Deviation_CI_Lower': deviation_ci_lower,
            'Deviation_CI_Upper': deviation_ci_upper
        }
        
        # Add column label if provided
        if column_label is not None:
            row_data = {'Metric': column_label, **row_data}
        
        summary_data.append(row_data)
    
    summary_stats = pd.DataFrame(summary_data)
    
    return anova_table, summary_stats
