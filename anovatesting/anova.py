import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import os
import argparse
import re
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing import Optional, Sequence, Tuple, Union



saveGraphs = False
showGraphs = True 
saveExcel = False


def select_first5_then_every4th(df: pd.DataFrame) -> pd.DataFrame:
    n = df.shape[1]
    # first five columns (0-based indices 0..4)
    first_five = list(range(0, min(5, n)))
    # every 4th column starting from column 6 (0-based index 5)
    every_4_from_6 = list(range(5, n, 4))
    keep_idx = first_five + every_4_from_6
    return df.iloc[:, keep_idx]


def twowayanova(df):
    
    
    
    categories = ["delay", "difficulty", "control", "feel"]
    
    column_map = []
        
    for x in range(5):
    
        new_list = [s + str(x) for s in categories]
         
        column_map += new_list
    
    
    
    
    df.loc[df['age'] <= 24, 'age'] = 24
    df.loc[df['age'] >= 25, 'age'] = 25
    
    print(df)
    
    melted = df.melt(id_vars=["gender", "age"], value_vars=["delay_0", "delay_1", "delay_2", "delay_3", "delay_4"],
                 var_name="delay", value_name="response")           
    model = ols('response ~ C(delay) + C(gender) + C(delay):C(gender)', data=melted).fit()
    anova_result = sm.stats.anova_lm(model, type=2)
    print("Two way anova results between age, delay for delay response")
    print(anova_result)
    print()    
    
    
    
    melted = df.melt(id_vars=["gender", "age"], value_vars=["difficulty_0", "difficulty_1", "difficulty_2", "difficulty_3", "difficulty_4"],
                 var_name="delay", value_name="response")           
    model = ols('response ~ C(delay) + C(gender) + C(delay):C(gender)', data=melted).fit()
    anova_result = sm.stats.anova_lm(model, type=2)
    print("Two way anova results between age, delay for difficulty response")
    print(anova_result)
    print()
    
    anova_result.to_excel('Two way anova results between age, delay for difficulty response.xlsx',  index=False)


    melted = df.melt(id_vars=["gender", "gender"], value_vars=["control_0", "control_1", "control_2", "control_3", "control_4"],
                 var_name="delay", value_name="response")
                 
                 
    model = ols('response ~ C(delay) + C(gender) + C(delay):C(gender)', data=melted).fit()
    anova_result = sm.stats.anova_lm(model, type=2)
    print("Two way anova results between age, delay for control")
    print(anova_result)
    print()


    melted = df.melt(id_vars=["gender", "gender"], value_vars=["body_0", "body_1", "body_2", "body_3", "body_4"],
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


