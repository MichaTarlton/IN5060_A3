# Python 3 — safer CI inversion for noncentrality parameter -> partial eta^2
# Requires: scipy
# pip install scipy

import math
from scipy.stats import ncf
from scipy.optimize import brentq

def safe_cdf_lambda(lambda_val, F_obs, df1, df2):
    """Return ncf.cdf but guard against NaNs and infinities."""
    try:
        val = ncf.cdf(F_obs, df1, df2, lambda_val)
        if not math.isfinite(val):
            return None
        return val
    except Exception:
        return None

def find_bracket_for_target(F_obs, df1, df2, target, max_hi=1e8):
    """
    Find [lo, hi] for lambda where cdf(lo) <= target <= cdf(hi)
    (or special-case when cdf(0) >= target -> lower bound is 0).
    Returns (lo, hi) where cdf(lo) and cdf(hi) are finite and bracket the target.
    Raises RuntimeError if it cannot bracket.
    """
    cdf0 = safe_cdf_lambda(0.0, F_obs, df1, df2)
    if cdf0 is None:
        raise RuntimeError("ncf.cdf(., lambda=0) returned non-finite value")
    if cdf0 >= target:
        # Lower bound is 0; bracket [0, hi] with hi>0 where cdf(hi) >= target
        hi = 1.0
        for _ in range(80):
            val = safe_cdf_lambda(hi, F_obs, df1, df2)
            if val is None:
                # if NaN, stop and accept hi as large finite cap (if cdf0 >= target we already good)
                break
            if val >= target:
                return 0.0, hi
            hi *= 2.0
            if hi > max_hi:
                break
        # if we didn't find hi, just return special-case lower=0 and hi as cap
        return 0.0, min(hi, max_hi)
    else:
        # need hi such that cdf(hi) >= target
        hi = 1.0
        for _ in range(200):
            val = safe_cdf_lambda(hi, F_obs, df1, df2)
            if val is None:
                # ncf.cdf produced NaN at this hi, so stop growing and use hi as cap
                hi = min(hi, max_hi)
                break
            if val >= target:
                return 0.0, hi
            hi *= 2.0
            if hi > max_hi:
                break
        # If we exit loop without success, raise or return a conservative bracket
        if safe_cdf_lambda(hi, F_obs, df1, df2) is None:
            # fallback to a finite hi
            hi = max_hi
            val = safe_cdf_lambda(hi, F_obs, df1, df2)
            if val is None:
                raise RuntimeError("Could not obtain finite ncf.cdf even at max_hi")
            if val >= target:
                return 0.0, hi
        raise RuntimeError("Could not bracket target in reasonable lambda range")

def ncp_ci_from_f_safe(F_obs, df1, df2, alpha=0.05):
    """
    Two-sided (1-alpha) CI for noncentrality parameter lambda using robust bracketing.
    Returns (lambda_lower, lambda_upper).
    """
    target_upper = 1.0 - alpha/2.0
    target_lower = alpha/2.0

    # Lower lambda: solve cdf(lambda_L) = 1 - alpha/2  (may be 0)
    cdf0 = safe_cdf_lambda(0.0, F_obs, df1, df2)
    if cdf0 is None:
        raise RuntimeError("ncf.cdf(., lambda=0) returned non-finite value; cannot proceed")
    if cdf0 >= target_upper:
        lam_lower = 0.0
    else:
        lo, hi = find_bracket_for_target(F_obs, df1, df2, target_upper)
        # ensure hi is finite and cdf(hi) >= target_upper
        lam_lower = brentq(lambda lam: ncf.cdf(F_obs, df1, df2, lam) - target_upper, lo, hi)

    # Upper lambda: solve cdf(lambda_U) = alpha/2
    # Note: cdf decreases with lambda, so cdf(large) -> 0
    lo = 0.0
    # find hi such that cdf(hi) <= target_lower
    hi = 1.0
    max_hi = 1e8
    for _ in range(200):
        val = safe_cdf_lambda(hi, F_obs, df1, df2)
        if val is None:
            # NaN encountered; treat hi as cap and break
            break
        if val <= target_lower:
            break
        hi *= 2.0
        if hi > max_hi:
            break
    val_hi = safe_cdf_lambda(hi, F_obs, df1, df2)
    if val_hi is None:
        # try a smaller but still large hi
        hi = 1e6
        val_hi = safe_cdf_lambda(hi, F_obs, df1, df2)
        if val_hi is None:
            raise RuntimeError("ncf.cdf became non-finite for large lambda; cannot find upper bracket")
    if val_hi > target_lower:
        raise RuntimeError("Could not find hi where cdf(hi) <= target_lower")
    lam_upper = brentq(lambda lam: ncf.cdf(F_obs, df1, df2, lam) - target_lower, lo, hi)
    return lam_lower, lam_upper

def partial_eta2_ci_from_f_safe(F_obs, df1, df2, alpha=0.05):
    lamL, lamU = ncp_ci_from_f_safe(F_obs, df1, df2, alpha=alpha)
    eta2_L = lamL / (lamL + df2) if (lamL + df2) > 0 else 0.0
    eta2_U = lamU / (lamU + df2) if (lamU + df2) > 0 else 0.0
    return eta2_L, eta2_U, lamL, lamU


rows = [
    ("C(delay)",  1.828846, 4, 165),
    ("C(age)",    2.498297, 1, 165),
    ("C(delay):C(age)", 0.484956, 4, 165),
]

for name, Fobs, df1, df2 in rows:
    eta2_point = (Fobs * df1) / (Fobs * df1 + df2)
    try:
        eta2_L, eta2_U, lamL, lamU = partial_eta2_ci_from_f_safe(Fobs, df1, df2, alpha=0.05)
        print(f"{name}: point {eta2_point:.4f}, 95% CI [{eta2_L:.4f}, {eta2_U:.4f}]")
    except Exception as e:
        print(f"{name}: error computing CI: {e}")


