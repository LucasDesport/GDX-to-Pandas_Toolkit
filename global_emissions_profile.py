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

import pickle
import pandas as pd
import matplotlib.pyplot as plt

# # Global emissions profile

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
df_ghg = dfd[dfd['Attribute'].isin(['14a_GHGinCO2eq (million ton)',
                                    '06a_Pos_CO2_fossil (million ton)',
                                    '07_CO2_industrial (million ton)',
                                    '08_CO2_land use change (million ton)',
                                    '08b_NE_bioccs',
                                    '08c_NE_daccs'])]
df_ghg.loc[:, 'Value'] = df_ghg['Value'] / 1000

# +
# Pivot and rename
pv_ghg = df_ghg.pivot_table(index=['Scenario', 'Year'], columns='Attribute', values='Value', aggfunc='sum', sort=False)
pv_ghg.columns = ['Fossil CO₂', 'Process CO₂', 'AFOLU CO₂', 'CDR - BECCS', 'CDR - DACCS', 'Non-CO₂ emissions']
pv_ghg = pv_ghg.reset_index()

# Colors
colors = [
    '#d73027',  # Fossil CO₂
    '#f46d43',  # Process CO₂
    '#fdae61',  # AFOLU CO₂
    '#4575b4',  # CDR - BECCS
    '#74add1',  # CDR - DACCS
    '#66c2a5'   # Non-CO₂ emissions
]

# Keep scenarios and years
scenarios = pv_ghg['Scenario'].unique()
years = sorted(pv_ghg['Year'].unique())

# Create a single string index for plotting
pv_ghg['Index'] = pv_ghg['Scenario'] + '-' + pv_ghg['Year'].astype(str)
pv_ghg = pv_ghg.set_index('Index')

# Plot
ax = pv_ghg.drop(columns=['Scenario', 'Year']).plot(
    kind='bar', stacked=True, figsize=(14, 6), color=colors
)

# Set xticks to year only
x_labels = pv_ghg.index.to_list()
year_labels = [label.split('-')[1] for label in x_labels]
ax.set_xticks(range(len(year_labels)))
ax.set_xticklabels(year_labels, rotation=90)

# Second x-axis for scenarios
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())

# Calculate midpoint index of each scenario based on exact match in the Scenario column
scenario_ranges = []
for scenario in scenarios:
    indices = pv_ghg[pv_ghg['Scenario'] == scenario].index
    if len(indices) > 0:
        midpoint = indices.tolist()
        midpoint_pos = sum([pv_ghg.index.get_loc(i) for i in midpoint]) / len(midpoint)
        scenario_ranges.append((midpoint_pos, scenario))

scenario_positions, scenario_labels = zip(*scenario_ranges)
ax2.set_xticks(scenario_positions)
ax2.set_xticklabels(scenario_labels)
ax2.tick_params(axis='x', which='both', length=0)
ax2.spines['top'].set_visible(False)
ax2.set_frame_on(False)
ax2.xaxis.set_label_position('bottom')
ax2.xaxis.set_ticks_position('bottom')
ax2.spines['bottom'].set_position(('outward', 40))
ax.set_xlabel("")  # This removes the label "Index"
ax.yaxis.grid(True, linestyle='--', alpha=0.5)

# Labels, legend, etc.
plt.title("Global emissions profile", weight='bold')
ax.set_ylabel("Emissions [GtCO₂eq]")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
plt.tight_layout()

plt.savefig("global_emission_profile.png", dpi=300, bbox_inches='tight')

plt.show()


# -


