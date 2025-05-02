# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Emissions price

import pickle
import pandas as pd
import matplotlib.pyplot as plt

# +
with open("dfs.pkl", "rb") as f:
    dfs = pickle.load(f)

dfd = dfs['data']
# -

# Pre-treatment to create a subcategory of the dataframe including only the parameter 'data' 
dfd = dfs['data']
dfd.columns = ['Attribute', 'Year', 'Region', 'Value', 'Scenario']
dfd['Year'] = pd.to_numeric(dfd['Year'], errors='coerce')
dfd = dfd[dfd['Year'] <= 2100]

# Select attributes and convert MtCO2eq into GtCO2eq
df_emis = dfd[dfd['Attribute'].isin(['46_CO2 price (US$ per ton CO2)',
                                     '46a_CO2eq price (US$ per ton CO2eq)',
                                    ])]
df_pghg = dfd[dfd['Attribute'].isin(['46a_CO2eq price (US$ per ton CO2eq)',
                                    ])]

# +
pv_pghg = df_pghg.pivot_table(index=['Year'], columns='Scenario', values='Value', aggfunc='mean')
pv_pghg = pv_pghg.reset_index()

# Plot
ax = pv_pghg.drop(columns=['Year']).plot(
    kind='line', stacked=False, figsize=(14, 6), colormap='tab20')

x_labels = pv_pghg['Year'].to_list()
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=90)

# Labels, legend, etc.
plt.title("Global average price of GHG", weight='bold')
ax.set_ylabel("Price [$/tCO₂]")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
# -


