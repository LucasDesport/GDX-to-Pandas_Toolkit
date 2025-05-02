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

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from gdxpds import to_dataframes
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import importlib
import config


def load_gdx_dfs(
    scenario_paths: Dict[str, str],
    time_range: Tuple[int, int] = (2014, 2100),
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Load parameters from multiple GDX files into pandas DataFrames,
    tagging each row with its scenario and optionally pivoting time dimensions.

    Parameters
    ----------
    scenario_paths :
        Mapping of scenario name to .gdx file path, e.g.
        {'v-ref': 'all_bca_p0_r0_gdpg-m_aeeg-m_sekl-m.gdx'.gdx', 'bca': 'all_bca_p0_r0_gdpg-m_aeeg-m_sekl-m.gdx'.gdx'}
    time_range :
        Inclusive (min, max) for plausible integer time/dimension values.
    verbose :
        Print warnings, overrides, and pivot decisions.

    Returns
    -------
    dfs :
        A dict where for each parameter symbol P:
          - dfs['P'] is the concatenated "long" DataFrame with a 'Scenario' column
          - if pivoted, dfs['P_wide'] is the "wide" form DataFrame
    """
    # Temporary storage of long-form pieces per symbol
    aggregated: Dict[str, List[pd.DataFrame]] = {}

    # Load each scenario file
    for scenario, path in scenario_paths.items():
        raw = to_dataframes(path)
        for name, df in raw.items():
            # copy and tag with scenario
            part = df.copy()
            part['Scenario'] = scenario
            aggregated.setdefault(name, []).append(part)

    # Final assembled dict
    dfs: Dict[str, pd.DataFrame] = {}

    # Process each parameter across scenarios
    for name, parts in aggregated.items():
        # Concatenate all long-form pieces for this parameter
        long_df = pd.concat(parts, ignore_index=True)
        cols = long_df.columns.tolist()
        if len(cols) < 2:
            if verbose:
                print(f"[skipping] '{name}' has <2 columns: {cols!r}")
            continue

        # Identify dimension columns (all except the last) and the value column
        dims, val_col = cols[:-1], cols[-1]
        dfs[name] = long_df.copy()

    return dfs


importlib.reload(config)
print(config.scenario_map)

dfs = load_gdx_dfs(config.scenario_map)

# +
# Pre-treatment to create a subcategory of the dataframe including only the parameter 'data' 
dfd = dfs['data']
dfd.columns = ['Attribute', 'Year', 'Region', 'Value', 'Scenario']
dfd['Year'] = pd.to_numeric(dfd['Year'], errors='coerce')
dfd = dfd[dfd['Year'] <= 2050]

dfd.to_csv('output.csv', index=False)
# -

# # Emissions profile

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
pv_ghg = df_ghg.pivot_table(index=['Scenario', 'Year'], columns='Attribute', values='Value', aggfunc='sum')
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

# Find ranges for each scenario group
scenario_ranges = []
start = 0
for scenario in scenarios:
    count = sum(pv_ghg.index.str.startswith(scenario))
    scenario_ranges.append((start + count / 2 - 0.5, scenario))
    start += count

# Set ticks at midpoints for scenarios
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
ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
# -

# # Global average price of GHG

dfp = dfd[dfd['Attribute'].isin(["46_CO2 price (US$ per ton CO2)"])]

# +
pv_dfp = dfp.pivot_table(index=['Year'], columns='Scenario', values='Value', aggfunc='mean')
pv_dfp = pv_dfp.reset_index()

# Plot
ax = pv_dfp.drop(columns=['Year']).plot(
    kind='line', stacked=False, figsize=(14, 6), colormap='tab20')

x_labels = pv_dfp['Year'].to_list()
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=90)

# Labels, legend, etc.
plt.title("Global average price of GHG", weight='bold')
ax.set_ylabel("Price [$/tCO₂]")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# +
car = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}

x = car.items()

car["year"] = 2018

print(x)
# -


