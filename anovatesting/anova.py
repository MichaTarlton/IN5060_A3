import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import os
import argparse
import re
from scipy.stats import tukey_hsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing import Optional, Sequence, Tuple, Union
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.stats.anova import anova_lm


saveGraphs = False
showGraphs = True 
saveExcel = False




# ---------- Example: assume you already ran Tukey and have `tukey` ----------
# e.g.
# tukey = pairwise_tukeyhsd(endog=df['response'], groups=df['a_g'], alpha=0.05)
# or using MultiComparison:
# mc = MultiComparison(df['response'], df['a_g'])
# tukey = mc.tukeyhsd()

# For the code below we assume `tukey` variable exists.

# ---------- Helper: convert tukey.summary() to a DataFrame -------------
# The summary() is a SimpleTable; we can grab its .data attribute and turn to DataFrame.
def tukey_to_df(tukey):
    # summary() returns a SimpleTable; .data contains header+rows
    s = tukey.summary()
    data = s.data  # list of lists: first row = header, remaining rows = rows
    header = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=header)
    # convert numeric columns to floats where appropriate
    for col in ['meandiff', 'p-adj', 'lower', 'upper']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ---------- Build square matrices (meandiff and p-adj) -------------
def pairwise_to_matrices(tukey):
    df = tukey_to_df(tukey)
    groups = np.unique(np.concatenate([df['group1'].values, df['group2'].values]))
    groups = list(groups)
    n = len(groups)
    meandiff_mat = pd.DataFrame(np.zeros((n,n)), index=groups, columns=groups)
    pval_mat = pd.DataFrame(np.ones((n,n)), index=groups, columns=groups)
    # Fill in both [i,j] and [j,i] (with sign symmetric for meandiff)
    for _, row in df.iterrows():
        g1 = row['group1']
        g2 = row['group2']
        md = row['meandiff']
        p = row['p-adj']
        meandiff_mat.loc[g1, g2] = md
        meandiff_mat.loc[g2, g1] = -md
        pval_mat.loc[g1, g2] = p
        pval_mat.loc[g2, g1] = p
    # diagonal: set to NaN or 0
    meandiff_mat.values[np.diag_indices(n)] = 0.0
    pval_mat.values[np.diag_indices(n)] = 0.0
    return meandiff_mat, pval_mat

# ---------- Optional: reorder groups by hierarchical clustering (so similar groups near each other) ----------
from scipy.cluster.hierarchy import linkage, leaves_list
def cluster_order(matrix, method='average'):
    # use distances derived from matrix values (e.g., meandiff absolute)
    dist = np.abs(matrix.values)
    # make a condensed distance vector for linkage
    # ensure symmetric and zero diagonal -> scipy pdist expects condensed vector; we use linkage on flattened
    # simpler: compute pairwise distances via 1 - corr, but here use absolute values as dissimilarity
    # create a 1D condensed distance using scipy.spatial.distance.pdist if desired
    from scipy.spatial.distance import pdist, squareform
    dcond = pdist(dist)  # condensing
    Z = linkage(dcond, method=method)
    idx = leaves_list(Z)
    order = matrix.index[idx].tolist()
    return order

# ---------- Plotting function: p-value heatmap and meandiff heatmap ----------
def plot_pairwise_heatmaps(tukey, cluster=False, cmap_p='viridis_r', cmap_md='RdBu_r',
                           show_significance=True, sig_threshold=0.05):
    meandiff_mat, pval_mat = pairwise_to_matrices(tukey)
    groups = meandiff_mat.index.tolist()

    if cluster:
        order = cluster_order(meandiff_mat.abs())
        meandiff_mat = meandiff_mat.reindex(index=order, columns=order)
        pval_mat = pval_mat.reindex(index=order, columns=order)

    # mask upper triangle (so we see each pair once)
    mask = np.triu(np.ones_like(pval_mat, dtype=bool), k=0)

    # 1) p-value heatmap (use -log10(p) or p itself)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    # transform p to -log10 for visual emphasis (handle p=0 -> set min)
    p_for_plot = -np.log10(pval_mat.replace(0, np.nextafter(0, 1)))
    sns.heatmap(p_for_plot, ax=ax1, mask=mask, cmap=cmap_p, annot=False,
                cbar_kws={'label': '-log10(p_adj)'})
    ax1.set_title('Pairwise p-values (adjusted) [-log10 scale]')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    # annotate significance stars on the lower triangle if desired
    if show_significance:
        for i, gi in enumerate(pval_mat.index):
            for j, gj in enumerate(pval_mat.columns):
                if i > j:  # lower only
                    p = pval_mat.iloc[i, j]
                    if p <= 0.001:
                        s = '***'
                    elif p <= 0.01:
                        s = '**'
                    elif p <= 0.05:
                        s = '*'
                    else:
                        s = ''
                    if s:
                        ax1.text(j + 0.5, i + 0.5, s, ha='center', va='center', color='white', fontsize=10)

    # 2) mean-difference heatmap
    sns.heatmap(meandiff_mat, ax=ax2, mask=mask, cmap=cmap_md, center=0,
                annot=False, cbar_kws={'label': 'mean difference (group1 - group2)'})
    ax2.set_title('Pairwise mean differences')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

    # annotate mean diffs and optionally significance stars
    for i, gi in enumerate(meandiff_mat.index):
        for j, gj in enumerate(meandiff_mat.columns):
            if i > j:  # lower triangle
                md = meandiff_mat.iloc[i, j]
                p = pval_mat.iloc[i, j]
                txt = f"{md:.2f}"
                if show_significance:
                    if p <= 0.001:
                        txt += "\n***"
                    elif p <= 0.01:
                        txt += "\n**"
                    elif p <= 0.05:
                        txt += "\n*"
                ax2.text(j + 0.5, i + 0.5, txt, ha='center', va='center', color='black', fontsize=8)

    plt.tight_layout()
    plt.show()


def violingraph(dfs):


    # optional: provide display labels in the same order
    display_labels = ['delay', 'Difficulty', 'control', 'part of body']
    delay = ['0ms', '50ms', '100ms', '150ms', '200ms']
    #fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(8, 6))
    #axes = axes.flatten()
    #fig.suptitle("test", fontsize=16, fontweight='bold')
    
    legend_handles = []
    legend_labels = []
    
    print(dfs)
        
    means = dfs.groupby('Category').mean()
    stds  = dfs.groupby('Category').std()
    print("mean")        
    print(means)
    print(stds['Value'].tolist())
    sns.set(style="whitegrid")
    # use data=df[selected] — seaborn will create one violin per column in that order
    sns.violinplot(x='Category', y='Value', data=dfs, inner=None)
    plt.title("Difficulty response from all different categories of groups")
    plt.ylabel("User response on difficulty across all delays on a scale from 1-5")

    x = np.arange(len(means))
    #legend_labels.append('mean ± variance')
    h = plt.errorbar(x, means['Value'], yerr=stds['Value'], fmt='o', color='k', capsize=5, label='mean ± std')
    plt.legend()
    # replace xtick labels with your display names
    #axes.set_xticklabels(display_labels)
    #plt.set_xlabel('Question')
    #plt.set_ylabel('User respone on a scale from 1-5')
    #plt.set_title(delay[0])
    plt.axis(ymin=0.1,ymax=5.9)
    
    
    """                
    legend_ax = axes[5]  # same one we turned off above
    # Make it accept the legend centered inside:
    legend_ax.set_axis_on()           # temporarily enable axis for placing legend
    legend_ax.axis('off')             # hide ticks & frame but keep area
    legend_ax.legend(legend_handles, legend_labels,
                     loc='upper left', frameon=True, fontsize=10)
    """
    if saveGraphs:
        plt.savefig(title)
    if showGraphs:
        plt.show()
    
    
    


def select_first5_then_every4th(df: pd.DataFrame) -> pd.DataFrame:
    n = df.shape[1]
    # first five columns (0-based indices 0..4)
    first_five = list(range(0, min(5, n)))
    # every 4th column starting from column 6 (0-based index 5)
    every_4_from_6 = list(range(5, n, 4))
    keep_idx = first_five + every_4_from_6
    return df.iloc[:, keep_idx]


def twowayanova(df):
    
    
    df = df.loc[df['gender'] != 'Other']
    categories = ["delay", "difficulty", "control", "feel"]
    

    column_map = []
        
    for x in range(5):
    
        new_list = [s + str(x) for s in categories]
         
        column_map += new_list
    
    
    
    
    df.loc[df['age'] <= 24, 'age'] = 24
    df.loc[df['age'] >= 25, 'age'] = 25
    
    #print(df)
    
    melted = df.melt(id_vars=["gender", "age"], value_vars=["delay_0", "delay_1", "delay_2", "delay_3", "delay_4"],
                 var_name="delay", value_name="response")           
    model = ols('response ~ C(delay) + C(gender) + C(delay):C(gender)', data=melted).fit()
    anova_result = sm.stats.anova_lm(model, type=2)
    print("Two way anova results between age, delay for delay response")
    print(anova_result)
    print()    
    
    
   
    
    melted = df.melt(id_vars=["gender", "age"], value_vars=["difficulty_0", "difficulty_1", "difficulty_2", "difficulty_3", "difficulty_4"],
                 var_name="delay", value_name="response")           
    #model = ols('response ~ C(delay) + C(age) + C(delay):C(age)', data=melted).fit()

    melted['gender'] = melted['gender'].astype('category')
    melted['age'] = melted['age'].astype('category')


    model = ols('response ~ C(delay) * C(age)', data=melted).fit()
    anova_result = sm.stats.anova_lm(model, type=2)
    print("Two way anova results between age, delay for difficulty response")
    print(anova_result)
    
    #anova_result.to_excel("Two way anova results between age, delay for difficulty response.xlsx")
    
    
    
    
    
    # Python: compute eta-squared and partial eta-squared for each effect
    import numpy as np
    ss = {'delay': 6.552941, 'age': 1.535666, 'interaction': 1.685794, 'residual': 146.513834}
    ss_total = ss['delay'] + ss['age'] + ss['interaction'] + ss['residual']
    eta2 = {k: v/ss_total for k,v in ss.items()}
    partial_eta2_delay = ss['delay'] / (ss['delay'] + ss['residual'])
    partial_eta2_age   = ss['age']   / (ss['age']   + ss['residual'])
    partial_eta2_inter = ss['interaction'] / (ss['interaction'] + ss['residual'])
    print("eta2:", eta2)
    print("partial eta2 (delay, age, interaction):", partial_eta2_delay, partial_eta2_age, partial_eta2_inter)
    
    
    
    
    
    # assume 'model' is the fitted OLS (smf.ols(...).fit())
    import scipy.stats as stats
    resid = model.resid
    print("Shapiro-Wilk p:", stats.shapiro(resid).pvalue)

    # Levene across delay x age cells
    groups = [g['response'].values for _, g in melted.groupby(['delay','age'])]
    print("Levene p:", stats.levene(*groups).pvalue)
    
    sns.pointplot(data=melted, x='delay', y='response', hue='age', dodge=True, capsize=.1)
    plt.title('Interaction plot: response by and age')
    plt.show()
    
    
    
    # ========================================================
    # Tukey comparisons for factor A (averaged over B)
    # ========================================================
    # pairwise_tukeyhsd expects endog (dependent) and groups (factor labels)
    tukey_A = pairwise_tukeyhsd(endog=melted['response'], groups=melted['delay'], alpha=0.05)
    print("\nTukey HSD for factor gender (averaged over age):")
    #print(tukey_A)
    print(tukey_A.summary())

    # ========================================================
    # Tukey comparisons for factor B (averaged over A)
    # ========================================================
    tukey_B = pairwise_tukeyhsd(endog=melted['response'], groups=melted['age'], alpha=0.05)
    print("\nTukey HSD for factor age (averaged over gender):")
    #print(tukey_B)
    print(tukey_B.summary())

    # ========================================================
    # Tukey comparisons for interaction: compare each A:B cell
    # ========================================================
    melted['a_g'] = melted['delay'].astype(str) + ":" + melted['age'].astype(str)
    tukey_inter = pairwise_tukeyhsd(endog=melted['response'], groups=melted['a_g'], alpha=0.05)
    print("\nTukey HSD for interaction (a:g cells):")
    #print(tukey_inter)
    print(tukey_inter.summary())
    
    
    plot_pairwise_heatmaps(tukey_inter, cluster=True, show_significance=True)
    
    
    
    
    print()
    
    
    female_old = melted['response'][(melted['age'] == 25) & (melted['gender'] == "Female")].tolist()
    female_young = melted['response'][(melted['age'] == 24) & (melted['gender'] == "Female")].tolist()
    male_old = melted['response'][(melted['age'] == 25) & (melted['gender'] == "Male")].tolist()
    male_young = melted['response'][(melted['age'] == 24) & (melted['gender'] == "Male")].tolist()
    

    data = {
        'Value': np.concatenate([female_old, female_young, male_old, male_young]),
        'Category': ['female_old'] * len(female_old) + ['female_young'] * len(female_young) + ['male_old'] * len(male_old) + ['male_young'] * len(male_young)
    }
    
    newdf = pd.DataFrame(data)
    
    violingraph(newdf)
    
    
    

    anova_result.to_excel('Two way anova results between age, delay for difficulty response.xlsx',  index=False)
    
    
   

    melted = df.melt(id_vars=["gender", "age"], value_vars=["control_0", "control_1", "control_2", "control_3", "control_4"],
                 var_name="delay", value_name="response")
                 
                 
    model = ols('response ~ C(delay) + C(gender) + C(delay):C(gender)', data=melted).fit()
    anova_result = sm.stats.anova_lm(model, type=2)
    print("Two way anova results between age, delay for control")
    print(anova_result)
    print()


    melted = df.melt(id_vars=["gender", "age"], value_vars=["body_0", "body_1", "body_2", "body_3", "body_4"],
                 var_name="delay", value_name="response")
                 
                 
    model = ols('response ~ C(delay) + C(gender) + C(delay):C(gender)', data=melted).fit()
    anova_result = sm.stats.anova_lm(model, type=2)
    print("Two way anova results between age, delay for body/feel response")
    print(anova_result)
    print()

      



   
parser = argparse.ArgumentParser()
parser.add_argument('--hide', action='store_true', default=False, help='Show graphs (True/False)')
parser.add_argument('--save', action='store_true', default=False, help='save graphs (True/False)') 
parser.add_argument('--saveExcel', action='store_true', default=False, help='save excel sheets of the filtered data (True/False)')        
args = parser.parse_args()

if args.hide:
    print("Graphs will be hidden")
    showGraphs = False

if args.save:
    print("Graphs will be saved as png")
    saveGraphs = args.save
    
if args.saveExcel:
    print("excell sheets will be saved")
    saveExcel = args.saveExcel


parent_dir = os.path.dirname(os.getcwd())  
filePath = f"{parent_dir}\data\questionnaire_data-561422-2025-11-17-1240.xlsx"
df = pd.read_excel(filePath)

_base_map = {
    'Participant number': 'participant_id',
    'What is your gender': 'gender',
    'How old are you?': 'age',
    'What is your dominant hand?': 'dominant_hand',
    'How experienced are you with robotic systems?': 'robotics_experience',
    'Did you experience delays between your actions and the robot\'s movements?': 'delay',
    'How difficult was it to perform the task?': 'difficulty',
    'I felt like I was controlling the movement of the robot': 'control',
    'It felt like the robot was part of my body': 'body'
}

repeating_keys = {'delay','difficulty','control','body'}
rename_map = {}
for col in df.columns:
    if col.startswith('$'):  # keep metadata columns unchanged
        continue
    # Remove HTML entity artifacts
    col_clean = col.replace('&#39;', "'")
    # Extract suffix like .1, .2, etc.
    m = re.search(r'\.(\d+)$', col_clean)
    base = re.sub(r'\.(\d+)$', '', col_clean)
    short = _base_map.get(base, None)
    if short is None:
        # If unknown and not metadata, create a generic snake_case key
        generic = re.sub(r'[^A-Za-z0-9]+', '_', base).strip('_').lower()
        short = generic or base.lower()
    if m:
        suffix = f"_{m.group(1)}"  # condition index from original numbering
    elif short in repeating_keys:
        suffix = "_0"  # first occurrence without explicit suffix
    else:
        suffix = ""
    rename_map[col] = short + suffix

df = df.rename(columns=rename_map)

df.drop(columns=df.columns[df.columns.str.contains(r'\$')], inplace=True)





twowayanova(df)


