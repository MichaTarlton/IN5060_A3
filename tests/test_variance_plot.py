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


def test_variance_decomposition_categorical():
    # Test with categorical data
    rng = np.random.default_rng(123)
    cat_data = [pd.Categorical(rng.choice(['low', 'medium', 'high'], size=100)) for _ in range(3)]
    fig, ax = vp.variance_decomposition_plot(cat_data)
    assert fig is not None and ax is not None
    matplotlib.pyplot.close(fig)


def test_plot_horizontal_violins_categorical():
    # Test with mixed numeric and categorical columns
    df = pd.DataFrame({
        'numeric': np.random.randn(100),
        'categorical': pd.Categorical(np.random.choice(['A', 'B', 'C'], size=100)),
        'strings': np.random.choice(['X', 'Y', 'Z'], size=100)
    })
    fig, ax = vp.plot_horizontal_violins(df, ['numeric', 'categorical', 'strings'])
    assert fig is not None and ax is not None
    matplotlib.pyplot.close(fig)


def test_variance_decomposition_likert_scale():
    # Test with Likert scale data (1-5 integers) - should use bars not KDE
    rng = np.random.default_rng(456)
    likert_data = [rng.integers(1, 6, size=150) for _ in range(4)]
    fig, ax = vp.variance_decomposition_plot(likert_data)
    assert fig is not None and ax is not None
    # Check that y-ticks match the discrete values
    y_ticks = ax.get_yticks()
    assert len(y_ticks) <= 10  # Should have discrete ticks, not many continuous ones
    matplotlib.pyplot.close(fig)


def test_variance_types():
    # Test different variance calculation types
    rng = np.random.default_rng(789)
    test_data = [rng.normal(loc=5, scale=2, size=100) for _ in range(3)]
    
    variance_types = ['sample', 'population', 'std', 'mse', 'ss', 'tukey', 'sst', 'sse', 'ssa', 'ftest']
    
    for vtype in variance_types:
        fig, ax = vp.variance_decomposition_plot(test_data, variance_type=vtype)
        assert fig is not None and ax is not None
        matplotlib.pyplot.close(fig)
    
    # Test invalid variance type
    try:
        vp.variance_decomposition_plot(test_data, variance_type='invalid')
    except ValueError as e:
        assert 'variance_type must be one of' in str(e)
    else:
        raise AssertionError('Expected ValueError for invalid variance_type')


def test_variance_types_horizontal():
    # Test variance types in plot_horizontal_violins
    rng = np.random.default_rng(999)
    df = pd.DataFrame({
        'a': rng.normal(size=100),
        'b': rng.normal(loc=3, size=100),
    })
    
    fig1, ax1 = vp.plot_horizontal_violins(df, ['a', 'b'], variance_type='sample')
    assert fig1 is not None and ax1 is not None
    matplotlib.pyplot.close(fig1)
    
    fig2, ax2 = vp.plot_horizontal_violins(df, ['a', 'b'], variance_type='population')
    assert fig2 is not None and ax2 is not None
    matplotlib.pyplot.close(fig2)


def test_toggle_options():
    # Test show_dotted_lines and show_y_annotations toggle options
    rng = np.random.default_rng(111)
    test_data = [rng.normal(loc=5, scale=2, size=100) for _ in range(3)]
    
    # Test with dotted lines on, y-annotations on (default)
    fig1, ax1 = vp.variance_decomposition_plot(test_data)
    assert fig1 is not None and ax1 is not None
    matplotlib.pyplot.close(fig1)
    
    # Test with dotted lines off, y-annotations on
    fig2, ax2 = vp.variance_decomposition_plot(test_data, show_dotted_lines=False)
    assert fig2 is not None and ax2 is not None
    matplotlib.pyplot.close(fig2)
    
    # Test with dotted lines on, y-annotations off
    fig3, ax3 = vp.variance_decomposition_plot(test_data, show_y_annotations=False)
    assert fig3 is not None and ax3 is not None
    matplotlib.pyplot.close(fig3)
    
    # Test with both off
    fig4, ax4 = vp.variance_decomposition_plot(test_data, 
                                               show_dotted_lines=False,
                                               show_y_annotations=False)
    assert fig4 is not None and ax4 is not None
    matplotlib.pyplot.close(fig4)


def test_calculate_statistics():
    # Test statistics calculation function
    import tempfile
    import os
    
    rng = np.random.default_rng(555)
    group1 = rng.normal(loc=3, scale=1, size=100)
    group2 = rng.normal(loc=5, scale=1.5, size=80)
    group3 = rng.normal(loc=4, scale=1.2, size=90)
    
    # Test without CSV output
    stats_df = vp.calculate_statistics(
        distributions=[group1, group2, group3],
        group_labels=['GroupA', 'GroupB', 'GroupC'],
        variance_types=['sample', 'population', 'std']
    )
    
    assert stats_df is not None
    assert len(stats_df) == 4  # 3 groups + overall
    assert 'Group' in stats_df.columns
    assert 'N' in stats_df.columns
    assert 'Mean' in stats_df.columns
    assert 'SAMPLE' in stats_df.columns
    assert 'POPULATION' in stats_df.columns
    assert 'STD' in stats_df.columns
    
    # Check group labels
    assert stats_df.iloc[0]['Group'] == 'GroupA'
    assert stats_df.iloc[1]['Group'] == 'GroupB'
    assert stats_df.iloc[2]['Group'] == 'GroupC'
    assert stats_df.iloc[3]['Group'] == 'Overall'
    
    # Check sample sizes
    assert stats_df.iloc[0]['N'] == 100
    assert stats_df.iloc[1]['N'] == 80
    assert stats_df.iloc[2]['N'] == 90
    assert stats_df.iloc[3]['N'] == 270  # total
    
    # Test with CSV output
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'test_stats.csv')
        stats_df_csv = vp.calculate_statistics(
            distributions=[group1, group2],
            group_labels=['A', 'B'],
            variance_types=['sample', 'mse'],
            output_csv=csv_path
        )
        
        assert os.path.exists(csv_path)
        
        # Read back and verify
        import pandas as pd
        df_read = pd.read_csv(csv_path)
        assert len(df_read) == 3  # 2 groups + overall
        assert 'SAMPLE' in df_read.columns
        assert 'MSE' in df_read.columns


def test_calculate_statistics_anova():
    # Test ANOVA-style statistics
    rng = np.random.default_rng(666)
    group1 = rng.normal(loc=2, scale=1, size=50)
    group2 = rng.normal(loc=4, scale=1, size=50)
    group3 = rng.normal(loc=3, scale=1, size=50)
    
    stats_df = vp.calculate_statistics(
        distributions=[group1, group2, group3],
        variance_types=['sst', 'sse', 'ssa']
    )
    
    # Check ANOVA decomposition: SST ≈ SSE + SSA
    total_sse = stats_df[stats_df['Group'] != 'Overall']['SSE'].sum()
    total_ssa = stats_df[stats_df['Group'] != 'Overall']['SSA'].sum()
    total_sst = stats_df[stats_df['Group'] == 'Overall']['SST'].values[0]
    
    # Should be approximately equal (within floating point tolerance)
    assert abs(total_sst - (total_sse + total_ssa)) < 1e-6


