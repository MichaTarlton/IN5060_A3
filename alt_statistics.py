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
