## Statistics Export Feature

The `calculate_statistics()` function provides comprehensive variance decomposition statistics:

### Usage
```python
stats_df = vp.calculate_statistics(
    distributions=[data1, data2, data3],
    group_labels=['Group1', 'Group2', 'Group3'],
    variance_types=['sample', 'std', 'mse', 'sst', 'sse', 'ssa'],
    output_csv='statistics.csv'  # Optional: saves to CSV
)
```

### Parameters
- **distributions**: List of array-like data for each group
- **overall**: Optional overall distribution (defaults to concatenation)
- **group_labels**: Names for each group (defaults to Group_1, Group_2, etc.)
- **variance_types**: List of variance metrics to calculate (defaults to all)
- **output_csv**: Path to save CSV file (optional)

### Available Variance Types
1. **sample**: Sample variance (ddof=1, unbiased)
2. **population**: Population variance (ddof=0)
3. **std**: Standard deviation (ddof=1)
4. **mse**: Mean Squared Error from mean
5. **ss**: Sum of Squares (within group)
6. **tukey**: Tukey's HSD estimate (variance/n)
7. **sst**: Total Sum of Squares (from grand mean)
8. **sse**: Error/Within Sum of Squares
9. **ssa**: Among/Between Groups Sum of Squares
10. **ftest**: F-test component (within-group variance)

### Output
Returns a pandas DataFrame with:
- One row per group plus an "Overall" row
- Columns: Group, N (sample size), Mean, and each requested variance metric
- Automatically saves to CSV if path specified
