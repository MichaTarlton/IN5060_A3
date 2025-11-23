import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import os
import argparse
from typing import Optional, Sequence, Tuple, Union



saveGraphs = False
showGraphs = True 
saveExcel = False




def violin_compare_two(
    data1,
    data2,
    *,
    labels: Tuple[str, str] = ("dataset1", "dataset2"),
    index_labels: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    palette: Optional[Sequence[str]] = None,
    inner: str = "quartile",   # inner representation: "box", "quartile", "point", "stick", or None
    split: bool = False,       # if True and hue has two levels, violins are split (overlapped)
    orient: str = "v",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot side-by-side violins comparing data1 and data2 across matching indices (columns).

    Parameters
    - data1, data2: pandas DataFrame or 2D array-like (same number of columns).
        If 1D (single series) is passed, it will be converted to a single-column DataFrame.
    - labels: tuple with labels for the two datasets used in the legend.
    - index_labels: optional sequence to use as x-axis labels (length must equal number of columns).
    - palette: sequence of two colors (or a seaborn palette name); defaults to seaborn color cycle.
    - inner: what to show inside the violin.
    - split: if True and hue has exactly two levels, violins are split (overlapped); else shown side-by-side.
    - orient: 'v' for vertical violins (default) or 'h' for horizontal.
    - xlabel, ylabel, title: plot text.
    - show_legend: whether to show the legend.
    - save_path, dpi: to save the figure.

    Returns (fig, ax).
    """
    # Convert inputs to DataFrames
    df1 = pd.DataFrame(data1).copy()
    df2 = pd.DataFrame(data2).copy()
    
    
    legend_handles = []
    legend_labels = []
    

    if df1.shape[1] != df2.shape[1]:
        raise ValueError("data1 and data2 must have the same number of columns/indices")

    n_cols = df1.shape[1]
    if index_labels is not None and len(index_labels) != n_cols:
        raise ValueError("index_labels length must match number of columns")

    # Name the columns consistently so melt produces a variable column
    col_names = list(range(n_cols)) if index_labels is None else list(index_labels)
    df1.columns = col_names
    df2.columns = col_names

    # Melt to long format and add dataset label
    long1 = df1.melt(var_name="index", value_name="value")
    long1["dataset"] = labels[0]
    long2 = df2.melt(var_name="index", value_name="value")
    long2["dataset"] = labels[1]
    long = pd.concat([long1, long2], ignore_index=True)

    # Ensure index is categorical and ordered
    long["index"] = pd.Categorical(long["index"], categories=col_names, ordered=True)

    # Palette handling
    if palette is None:
        palette = sns.color_palette(n_colors=2)
    elif isinstance(palette, str):
        palette = sns.color_palette(palette, n_colors=2)

    fig, ax = plt.subplots(figsize=figsize)
    
    # long: DataFrame with columns "index" and "value"
    means_df = long.groupby(["index","dataset"], as_index=False)["value"].mean()
    means_df = long.groupby(["index","dataset"], as_index=False)["value"].mean()
    
    means = means_df["value"].tolist()   # simple Python list of means in the grouped order

    print(long)
    
    std_df = long.groupby(["index","dataset"], as_index=False)["value"].std()
    stds = std_df["value"].tolist()   # simple Python list of means in the grouped order

    # result: DataFrame with columns ["index", "value"] where "value" is the mean
    # rename for clarity:
    
    print(means_df)
    meansReversed = []
    stdsReversed = []
    if(labels[0] == "Male"):

        print(len(means))
        for x in range(len(means)):
            if x % 2 == 0:
                meansReversed.append(means[x + 1])
                stdsReversed.append(stds[x + 1])
                
            else:
                meansReversed.append(means[x - 1])
                stdsReversed.append(stds[x - 1])
                
        
    else:
        meansReversed = means
        stdsReversed = stds
        
    print(meansReversed) 
    
   
    
    plt.grid(True, axis='y', color='black') 
    ax.set_axisbelow(True)

    # If split=True, seaborn will draw split violins for the two dataset levels when hue has 2 levels.
    sns.violinplot(
        data=long,
        x="index" if orient == "v" else "value",
        y="value" if orient == "v" else "index",
        hue="dataset",
        split=split,
        inner=None,
        palette=palette,
        orient=orient,
        ax=ax,
        cut=1,
    )
    
    
    
    ax.axis(ymin=0.1,ymax=5.9)
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        if orient == "v":
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Value")

    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        if orient == "v":
            ax.set_ylabel("Value")
        else:
            ax.set_ylabel("")

    if title:
        ax.set_title(title)

    if not show_legend:
        ax.get_legend().remove()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    
    
    
    
    violin_centers = []
    for coll in ax.collections:
    # Only consider PolyCollection objects which contain the violin bodies
        if isinstance(coll, matplotlib.collections.PolyCollection):
            # each path corresponds to one polygon (violin body)
            for path in coll.get_paths():
                verts = path.vertices
                x_mean = float(np.mean(verts[:, 0]))
                violin_centers.append(x_mean)

    
    h = ax.errorbar(violin_centers, meansReversed, yerr=stdsReversed, fmt='o', color='k', capsize=5, label='mean ± std')
 
    handles, labels = ax.get_legend_handles_labels()
    handles.append('mean ± std')
    labels.append(h[0])
    
    
    ax.legend(handles=handles, labels=labels, loc="upper right")
                     
    

    return fig, ax







def genderGroups(df):
    
    #makes violin graphs for males
    dfMale = df[df["What is your gender"] == "Male"]
    dfFemale = df[df["What is your gender"] == "Female"]
    
    delayMale  = dfMale.iloc[:, 5:25:4]
    delayFemale = dfFemale.iloc[:, 5:25:4]
        
    
    
    
    fig, ax = violin_compare_two(
        delayMale,
        delayFemale,
        labels=("Male", "Female"),
        index_labels=['0ms', '50ms', '100ms', '150ms', '200ms'],
        title="Experienced delay moving cubes",
        xlabel=("Delay"),
        ylabel=("Exerienced delay")
    )

    
    if saveGraphs:
        plt.savefig("Experienced delay moving cubes genders")
    plt.show()
    
    
    
    difficultyMale  = dfMale.iloc[:, 6:25:4]
    difficultyFemale = dfFemale.iloc[:, 6:25:4]
    
    #"createSingleLineGraph(difficultyMale, difficultyFemale, "Male", "Female", "Difficulty moving cubes", "Reported difficulty")
    
    fig, ax = violin_compare_two(
        difficultyMale,
        difficultyFemale,
        labels=("Male", "Female"),
        index_labels=['0ms', '50ms', '100ms', '150ms', '200ms'],
        title="Difficulty moving cubes",
        xlabel=("Delay"),
        ylabel=("Reported difficulty")
    )

    
    if saveGraphs:
        plt.savefig("Difficulty moving cubes genders")
    plt.show()


    controlMale  = dfMale.iloc[:, 7:25:4]
    controlFemale = dfFemale.iloc[:, 7:25:4]

   # createSingleLineGraph(controlMale, controlFemale, "Male", "Female", "Felt control moving cubes", "Reported sense of control")
    
    fig, ax = violin_compare_two(
        controlMale,
        controlFemale,
        labels=("Male", "Female"),
        index_labels=['0ms', '50ms', '100ms', '150ms', '200ms'],
        title="Felt control moving cubes",
        xlabel=("Delay"),
        ylabel=("Reported sense of control")
    )

    if saveGraphs:
        plt.savefig("Felt control moving cubes genders")
    plt.show()
    

    feelMale  = dfMale.iloc[:, 8:25:4]
    feelFemale = dfFemale.iloc[:, 8:25:4]

  #  createSingleLineGraph(feelMale, feelFemale, "Male", "Female", "Feeling like own arm moving cubes", "Reported sense of feel")
   
    fig, ax = violin_compare_two(
        feelMale,
        feelFemale,
        labels=("Male", "Female"),
        index_labels=['0ms', '50ms', '100ms', '150ms', '200ms'],
        title="How much arm feels like your own moving cubes",
        xlabel=("Delay"),
        ylabel=("Reported sense of arm feeling like your own")
    )
    if saveGraphs:
        plt.savefig("How much arm feels like your own moving cubes")
    plt.show()
    
    


            
            
            
            
            
            
"""
Makes violin graphs for both age groups both tasks

"""  
def ageGroups(df):

    dfYoung = df[df["How old are you?"] <= 24]
    dfOld = df[df["How old are you?"] >= 25]


    delayYoung  = dfYoung.iloc[:, 5:25:4]
    delayOld = dfOld.iloc[:, 5:25:4]
        
    
    
    
    fig, ax = violin_compare_two(
        delayYoung,
        delayOld,
        labels=("age 24 or lower", "age 25 or higher"),
        index_labels=['0ms', '50ms', '100ms', '150ms', '200ms'],
        title="Experienced delay moving cubes",
        xlabel=("Delay"),
        ylabel=("Exerienced delay")
    )

    if saveGraphs:
        plt.savefig("Experienced delay moving cubes ages")
    plt.show()
    
    
    
    difficultyMale  = dfYoung.iloc[:, 6:25:4]
    difficultyFemale = dfOld.iloc[:, 6:25:4]
    
    #"createSingleLineGraph(difficultyMale, difficultyFemale, "Male", "Female", "Difficulty moving cubes", "Reported difficulty")
    
    fig, ax = violin_compare_two(
        difficultyMale,
        difficultyFemale,
        labels=("age 24 or lower", "age 25 or higher"),
        index_labels=['0ms', '50ms', '100ms', '150ms', '200ms'],
        title="Difficulty moving cubes",
        xlabel=("Delay"),
        ylabel=("Reported difficulty")
    )

    if saveGraphs:
        plt.savefig("Difficulty moving cubes ages")
    plt.show()


    controlMale  = dfYoung.iloc[:, 7:25:4]
    controlFemale = dfOld.iloc[:, 7:25:4]

   # createSingleLineGraph(controlMale, controlFemale, "Male", "Female", "Felt control moving cubes", "Reported sense of control")
    
    fig, ax = violin_compare_two(
        controlMale,
        controlFemale,
        labels=("age 24 or lower", "age 25 or higher"),
        index_labels=['0ms', '50ms', '100ms', '150ms', '200ms'],
        title="Felt control moving cubes",
        xlabel=("Delay"),
        ylabel=("Reported sense of control")
    )

    if saveGraphs:
        plt.savefig("Felt control moving cubes ages")
    plt.show()
    

    feelMale  = dfYoung.iloc[:, 8:25:4]
    feelFemale = dfOld.iloc[:, 8:25:4]

  #  createSingleLineGraph(feelMale, feelFemale, "Male", "Female", "Feeling like own arm moving cubes", "Reported sense of feel")
   
    fig, ax = violin_compare_two(
        feelMale,
        feelFemale,
        labels=("age 24 or lower", "age 25 or higher"),
        index_labels=['0ms', '50ms', '100ms', '150ms', '200ms'],
        title="How much arm feels like your own moving cubes",
        xlabel=("Delay"),
        ylabel=("Reported sense of arm feeling like your own")
    )

    if saveGraphs:
        plt.savefig("How much arm feels like your own moving cubes ages")    
    plt.show()
    
   
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
df.drop(columns=df.columns[df.columns.str.contains(r'\$')], inplace=True)


genderGroups(df)
ageGroups(df)
