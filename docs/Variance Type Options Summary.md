## Variance Type Options Summary
See line

The `variance_type` parameter supports **10 different dispersion metrics**:

### Basic Variance Metrics

#### 1. **`'sample'`** (default)
- **Formula**: $s^2 = \frac{\sum(x - \bar{x})^2}{n-1}$ (Bessel's correction, ddof=1)
- **Use case**: Unbiased estimator for population variance from sample data
- **Best for**: Standard statistical analysis, small samples

#### 2. **`'population'`**
- **Formula**: $\sigma^2 = \frac{\sum(x - \bar{x})^2}{n}$ (ddof=0)
- **Use case**: Variance of the complete population
- **Best for**: When you have the entire population, not a sample

#### 3. **`'std'`**
- **Formula**: Same as `'sample'` (ddof=1)
- **Use case**: Emphasizes standard deviation interpretation
- **Best for**: When thinking in terms of standard deviation units

#### 4. **`'mse'`** (Mean Squared Error)
- **Formula**: $MSE = \frac{\sum(x - \bar{x})^2}{n}$ (same as population variance)
- **Use case**: Average squared deviation per observation
- **Best for**: Comparing dispersion across groups of different sizes

#### 5. **`'ss'`** (Sum of Squares)
- **Formula**: $SS = \sum(x - \bar{x})^2$ (within-group, unnormalized)
- **Use case**: Total variation within a group
- **Best for**: ANOVA decomposition, variance component analysis

#### 6. **`'tukey'`** (Tukey's HSD)
- **Formula**: $\frac{s^2}{n}$ (variance estimate for pairwise comparisons)
- **Use case**: Standard error estimate for Tukey's Honestly Significant Difference
- **Best for**: Post-hoc pairwise comparisons after ANOVA

### ANOVA Metrics

#### 7. **`'sst'`** (Total Sum of Squares)
- **Formula**: $SST = \sum(x - \bar{x}_{grand})^2$ 
- **Use case**: Total variation from grand mean across all groups
- **Best for**: ANOVA total variation, variance decomposition
- **Note**: Measures how much each observation deviates from the overall mean

#### 8. **`'sse'`** (Error/Within Sum of Squares)
- **Formula**: $SSE = \sum(x - \bar{x}_{group})^2$
- **Use case**: Within-group variation (same as 'ss' but emphasizes ANOVA context)
- **Best for**: ANOVA residual/error term, unexplained variation
- **Note**: Measures variation within each group around its own mean

#### 9. **`'ssa'`** (Among/Between Groups Sum of Squares)
- **Formula**: $SSA = n(\bar{x}_{group} - \bar{x}_{grand})^2$
- **Use case**: Between-group variation explained by group membership
- **Best for**: ANOVA treatment effect, explained variation
- **Note**: Measures how much group means differ from the grand mean

#### 10. **`'ftest'`** (F-test Component)
- **Formula**: Within-group variance (same as 'sample')
- **Use case**: Component for F-statistic calculation
- **Best for**: Preparing data for F-ratio computation
- **Note**: F = MSA/MSE where MSA = SSA/df_between, MSE = SSE/df_within

### ANOVA Decomposition Identity

**SST = SSE + SSA**

- **SST**: Total variation in the data
- **SSE**: Variation within groups (unexplained)
- **SSA**: Variation between groups (explained by grouping)

### Usage Examples

```python
# Standard sample variance (default)
fig, ax = vp.variance_decomposition_plot(data, variance_type='sample')

# Sum of squares for ANOVA-style analysis
fig, ax = vp.variance_decomposition_plot(data, variance_type='ss')

# MSE for comparing groups of different sizes
fig, ax = vp.variance_decomposition_plot(data, variance_type='mse')

# ANOVA Total Sum of Squares
fig, ax = vp.variance_decomposition_plot(data, variance_type='sst')

# ANOVA Error Sum of Squares (within-group)
fig, ax = vp.variance_decomposition_plot(data, variance_type='sse')

# ANOVA Among-Groups Sum of Squares (between-group)
fig, ax = vp.variance_decomposition_plot(data, variance_type='ssa')
```
