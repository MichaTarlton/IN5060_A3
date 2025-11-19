import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless

import fig.variance_plot as vp


def test_variance_decomposition_basic():
    # Create synthetic conditional distributions
    rng = np.random.default_rng(42)
    arrays = [rng.normal(loc=mu, scale=1.0, size=200) for mu in (0, 5, 10, 15)]
    fig, ax = vp.variance_decomposition_plot(arrays, dotted=True, annotate=True)
    # Basic structural assertions
    assert fig is not None and ax is not None
    # Should have at least one text annotation (overall mean or E(Y|X_i))
    texts = [t.get_text() for t in ax.texts]
    assert any('E(Y' in txt for txt in texts)
    # Close figure to free memory
    matplotlib.pyplot.close(fig)


def test_plot_horizontal_violins_basic():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'a': rng.normal(size=150),
        'b': rng.normal(loc=2, size=150),
        'c': rng.integers(0, 5, size=150)
    })
    fig, ax = vp.plot_horizontal_violins(df, ['a', 'b', 'c'])
    assert fig is not None and ax is not None
    # Expect as many xticklabels as numeric columns selected
    assert len(ax.get_xticklabels()) == 3
    matplotlib.pyplot.close(fig)


def test_variance_decomposition_no_distributions():
    try:
        vp.variance_decomposition_plot([])
    except ValueError as e:
        assert 'empty' in str(e)
    else:
        raise AssertionError('Expected ValueError for empty distributions')
