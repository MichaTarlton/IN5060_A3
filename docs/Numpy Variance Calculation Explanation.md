# Variance Calculation Explanation
See line 198 of variance_plot

## Code Overview
```python
a.var(ddof=1)
```

## Function Breakdown

### `a.var(ddof=1)`
- **Purpose**: Calculates the sample variance of array `a`
- **Method**: Uses Pandas/NumPy variance function
- **Parameter**: `ddof=1` applies Bessel's correction

## Key Concepts

### Degrees of Freedom Correction (`ddof`)
| Parameter | Formula | Use Case |
|-----------|---------|----------|
| `ddof=0` | $\frac{\sum(x_i - \bar{x})^2}{N}$ | Population variance |
| [ddof=1](http://_vscodecontentref_/8) | $\frac{\sum(x_i - \bar{x})^2}{N-1}$ | Sample variance |

### Why [ddof=1](http://_vscodecontentref_/9)?
- Provides **unbiased estimation** of population variance
- Compensates for using sample mean instead of true population mean
- Standard practice in inferential statistics

## Example
```python
import pandas as pd

data = [1, 2, 3, 4, 5]
a = pd.Series(data)

population_var = a.var(ddof=0)  # 2.0
sample_var = a.var(ddof=1)      # 2.5
```

The sample variance is larger because dividing by $N-1$ instead of $N$ accounts for the uncertainty in estimating the population mean from sample data.