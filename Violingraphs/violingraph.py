import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def make_3_top_2_bottom(figsize=(12, 6), top_height=1, bottom_height=1, dpi=100):
    """
    Create a figure where all visible subplots are equal-sized using GridSpec.
    Layout: 2 rows x 3 columns grid. Top: 3 plots. Bottom: 2 plots (left and center).
    The bottom-right cell is left empty and hidden so all visible cells are identical.
    Returns (fig, axes) where axes is a list [ax0,...,ax5] corresponding to grid cells left->right, top->bottom.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    # Create 2x3 grid with equal cell sizes (equal height/width ratios)
    gs = gridspec.GridSpec(nrows=2, ncols=3, figure=fig,
                           height_ratios=[1, 1], width_ratios=[1, 1, 1])
    # Add one Axes per grid cell
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    # Example plots for the visible cells:
    # Hide the unused bottom-right cell so all visible axes are equal-sized
    axes[5].axis("off")  # or axes[5].set_visible(False)
    return fig, axes

def makeViolinGraphs(dfs, title='violingraph'):
    # optional: provide display labels in the same order
    display_labels = ['delay', 'Difficulty', 'control', 'part of body']
    delay = ['0ms', '50ms', '100ms', '150ms', '200ms']
    #fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(8, 6))
    fig, axes = make_3_top_2_bottom(figsize=(12, 6), top_height=1, bottom_height=1.2)
    #axes = axes.flatten()

    for i, ax in enumerate(axes):
        
        sns.set(style="whitegrid")
        # use data=df[selected] — seaborn will create one violin per column in that order
        sns.violinplot(data=dfs[i], ax=ax, cut=1)
        # replace xtick labels with your display names
        ax.set_xticklabels(display_labels)
        ax.set_xlabel('Question')
        ax.set_ylabel('Difficulty')
        ax.set_title(delay[i])
        if i == 4:
            break
        
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.show()
    
"""
Makes violin graphs for all groups both tasks

"""
def allGroups(df):
    dfs = [df.iloc[:, i : i + 4] for i in range(5, 25, 4)]
    makeViolinGraphs(dfs, 'Moving Cubes')
    dfs = [df.iloc[:, i : i + 4] for i in range(25, 45, 4)]
    makeViolinGraphs(dfs, 'Moving Cans')
"""
Makes violin graphs for male/female genders both tasks

""" 
def genderGroups(df):
    
    #makes violin graphs for males
    dfMale = df[df["What is your gender"] == "Male"]
    dfs = [dfMale.iloc[:, i : i + 4] for i in range(5, 25, 4)]
    makeViolinGraphs(dfs, 'Moving Cubes Males')
    dfs = [dfMale.iloc[:, i : i + 4] for i in range(25, 45, 4)]
    makeViolinGraphs(dfs, 'Moving Cans Males')
    
    #makes violin graphs for females
    dfFemale = df[df["What is your gender"] == "Female"]
    dfs = [dfFemale.iloc[:, i : i + 4] for i in range(5, 25, 4)]
    makeViolinGraphs(dfs, 'Moving Cubes Females')
    dfs = [dfFemale.iloc[:, i : i + 4] for i in range(25, 45, 4)]
    makeViolinGraphs(dfs, 'Moving Cans Females')
    
    #makes excel sheets to inspect what data is used
    '''
    try :
        dfMale.to_excel('Male.xlsx', index=False)
    except PermissionError:
        print("Could not make Male.xlsx")
        
    try :
        dfFemale.to_excel('Female.xlsx', index=False)
    except PermissionError:
        print("Could not make Female.xlsx")    
    '''
"""
Makes violin graphs for both age groups both tasks

"""  
def ageGroups(df):
    dfYoung = df[df["How old are you?"] <= 24]
    dfs = [dfYoung.iloc[:, i : i + 4] for i in range(5, 25, 4)]
    makeViolinGraphs(dfs, 'Moving Cubes age 24 or less')
    dfs = [dfYoung.iloc[:, i : i + 4] for i in range(25, 45, 4)]
    makeViolinGraphs(dfs, 'Moving Cans age 24 or less')

    dfOld = df[df["How old are you?"] >= 25]
    dfs = [dfOld.iloc[:, i : i + 4] for i in range(5, 25, 4)]
    makeViolinGraphs(dfs, 'Moving Cans age 25 or more')
    dfs = [dfOld.iloc[:, i : i + 4] for i in range(25, 45, 4)]
    makeViolinGraphs(dfs, 'Moving Cans age 25 or more')
    
    
    '''
        #makes excel sheets to inspect what data is used
    try :
        dfYoung.to_excel('young.xlsx', index=False)
    except PermissionError:
        print("Could not make young.xlsx")
        
    try :
        dfOld.to_excel('old.xlsx', index=False)
    except PermissionError:
        print("Could not make old.xlsx")       
    '''
    
filePath = 'questionnaire_data-561422-2025-11-17-1240.xlsx'
df = pd.read_excel(filePath)
df.drop(columns=df.columns[df.columns.str.contains(r'\$')], inplace=True)
allGroups(df)
genderGroups(df)
ageGroups(df)
