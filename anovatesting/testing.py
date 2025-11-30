import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Prepare your multiple lists
list_a = np.random.normal(loc=0, scale=1, size=100)
list_b = np.random.normal(loc=1, scale=0.5, size=100)
list_c = np.random.normal(loc=-0.5, scale=1.5, size=100)

# 2. Create a dictionary and convert to a Pandas DataFrame
data_dict = {
    'List A': list_a,
    'List B': list_b,
    'List C': list_c
}
df = pd.DataFrame(data_dict)

# 3. Melt the DataFrame to long format for Seaborn
df_melted = df.melt(var_name='List Name', value_name='Value')

# 4. Create the violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(x='List Name', y='Value', data=df_melted)
plt.title('Violin Plot of Multiple Lists')
plt.ylabel('Value')
plt.xlabel('List')
plt.show()