'''
Collection of functions to be called in notebooks using importlib
'''

from typing import Dict, List, Optional, Union, Tuple
from gdxpds import to_dataframes

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

import importlib
import scenmap
from library import lib


from pathlib import Path


def gdx2dfs(
    scenario_paths: Dict[str, str],
    time_range: Tuple[int, int] = (2014, 2100),
    verbose: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Load parameters from multiple GDX files into pandas DataFrames
    """
    aggregated: Dict[str, List[pd.DataFrame]] = {}

    # Load each scenario file
    for scenario, path in scenario_paths.items():
        raw = to_dataframes(path)
        for name, df in raw.items():
            part = df.copy()
            part['Scenario'] = scenario
            aggregated.setdefault(name, []).append(part)

    dfs: Dict[str, pd.DataFrame] = {}

    for name, parts in aggregated.items():
        # Remove empty or all-NA DataFrames
        valid_parts = [
            df for df in parts
            if isinstance(df, pd.DataFrame) and not df.empty and not df.isna().all().all()
        ]

        if not valid_parts:
            if verbose:
                print(f"[skipping] '{name}' has no valid dataframes.")
            continue

        # Ensure all parts have the same columns
        col_sets = [tuple(df.columns) for df in valid_parts]
        if len(set(col_sets)) > 1:
            if verbose:
                print(f"[warning] '{name}' has mismatched columns across scenarios. Skipping.")
                for i, df in enumerate(valid_parts):
                    print(f"  Scenario {i} columns: {df.columns.tolist()}")
            continue

        try:
            long_df = pd.concat(valid_parts, ignore_index=True)
        except Exception as e:
            if verbose:
                print(f"[error] Failed to concat '{name}': {e}")
            continue

        cols = long_df.columns.tolist()
        if len(cols) < 2:
            if verbose:
                print(f"[skipping] '{name}' has <2 columns: {cols!r}")
            continue

        dims, val_col = cols[:-1], cols[-1]
        dfs[name] = long_df.copy()

    dfd = dfs['data']
    dfd.columns = ['Attribute', 'Year', 'Region', 'Value', 'Scenario']
    dfd['Year'] = pd.to_numeric(dfd['Year'], errors='coerce')
    dfd = dfd[dfd['Year'] <= 2100]

    dfd.to_csv('data.csv', index=False)
    
    return dfs, dfd

# # Global emissions profile

def gemis(dfd):

    '''
    Plot global emssions profiles from different scenarios.
    '''

    # Select emissions variables
    df = dfd[dfd['Attribute'].isin(['14a_GHGinCO2eq (million ton)',
                                    '06a_Pos_CO2_fossil (million ton)',
                                    '07_CO2_industrial (million ton)',
                                    '08_CO2_land use change (million ton)',
                                    '08b_NE_bioccs',
                                    '08c_NE_daccs'])]
    df.loc[:, 'Value'] = df['Value'] / 1000 # to convert in GtCO2eq

    # Pivot and rename
    pv = df.pivot_table(index=['Scenario', 'Year'], columns='Attribute', values='Value', aggfunc='sum', sort=False) # keeps scenarios order as listed in scenario_map
    pv.columns = ['Fossil CO₂', 'Process CO₂', 'AFOLU CO₂', 'CDR - BECCS', 'CDR - DACCS', 'Non-CO₂ emissions'] # rename emissions variables
    pv = pv.reset_index()

    # set colors
    colors = [
    '#d73027',  # Fossil CO₂
    '#f46d43',  # Process CO₂
    '#fdae61',  # AFOLU CO₂
    '#4575b4',  # CDR - BECCS
    '#74add1',  # CDR - DACCS
    '#66c2a5'   # Non-CO₂ emissions
    ]

    # Extract scenarios' names and years
    scenarios = pv['Scenario'].unique()
    years = sorted(pv['Year'].unique())

    # Create a single string index for plotting
    pv['Index'] = pv['Scenario'] + '-' + pv['Year'].astype(str)
    pv = pv.set_index('Index')

    # Plot
    ax = pv.drop(columns=['Scenario', 'Year']).plot(kind='bar', stacked=True, figsize=(14, 6), color=colors)

    # Set xticks to year only (first axis)
    x_labels = pv.index.to_list()
    year_labels = [label.split('-')[1] for label in x_labels]
    ax.set_xticks(range(len(year_labels)))
    ax.set_xticklabels(year_labels, rotation=90)

    # Second x-axis for scenarios (sercond axis)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    # Calculate midpoint index of each scenario based on exact match in the Scenario column
    scenario_ranges = []
    for scenario in scenarios:
        indices = pv[pv['Scenario'] == scenario].index
        if len(indices) > 0:
            midpoint = indices.tolist()
            midpoint_pos = sum([pv.index.get_loc(i) for i in midpoint]) / len(midpoint)
            scenario_ranges.append((midpoint_pos, scenario))

    # Plot characteristics
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
    plt.gca().yaxis.set_major_locator(MultipleLocator(5)) # increment each 5 GtCO2eq

    # Labels, legend, etc.
    plt.title("Global emissions profile", weight='bold')
    ax.set_ylabel("Emissions [GtCO₂eq]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    plt.tight_layout()
    
    plt.savefig(Path("global_emission_profile.png"), dpi=300, bbox_inches='tight')
    
    plt.show()

# # Global GHG price

def pemis(dfd, emis_type: str): #emis_type can only be 'co2' or 'ghg'
    '''
    Plot the evolution of the carbon price, either CO2 or GHG
    '''

    if emis_type == 'co2':
        df = dfd[dfd['Attribute'].isin(['46_CO2 price (US$ per ton CO2)'])]
    elif emis_type == 'ghg':
        df = dfd[dfd['Attribute'].isin(['46a_CO2eq price (US$ per ton CO2eq)'])]
    else:
        print("Error: choose emis_type = 'co2' or 'ghg'")
        return None

    pv = df.pivot_table(index=['Year'], columns='Scenario', values='Value', aggfunc='mean')
    pv = pv.reset_index()
    
    # Plot
    ax = pv.drop(columns=['Year']).plot(
        kind='line', stacked=False, figsize=(14, 6), colormap='tab20')
    
    x_labels = pv['Year'].to_list()
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90)
    
    # Labels, legend, etc.
    plt.title("Global emissions price", weight='bold')
    ax.set_ylabel("Price [$/tCO₂]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    plt.gca().yaxis.set_major_locator(MultipleLocator(25))
    plt.tight_layout()
    
    plt.savefig("global_emission_price.png", dpi=300, bbox_inches='tight')
    
    plt.show()

# # Exploring GRT variables

def plot_grt(attr, sector, region, draw, dfs):
    '''
    Compare across scenarios parameters definef by their sector G, region R, and time
    '''
    
    df = dfs[attr]
    dfs['sco2'].columns = ['t', 'G', 'R', 'Value', 'Scenario'] #to make this specific parameter fit with others
    df = df.loc[df['G'] == sector].copy() # .copy() is to prevent a pandas error when modifying the values in the next line
    df['Value'] *= lib['Converter'][attr]
    df = df[pd.to_numeric(df['t'], errors='coerce').notnull()] #you can filter the years here by replace '.notnull()' with < YYYY

    if region == 'global':
        df = df.groupby(['t', 'Scenario'], as_index=False, sort=False)['Value'].sum()
        df = df.pivot_table(index='t', columns='Scenario', values='Value', sort=False).reset_index()
        x_labels = df['t'].astype(str)
        scenario_columns = [col for col in df.columns if col != 't']
    else:
        df = df.pivot_table(index=['t', 'R'], columns='Scenario', values='Value', sort=False).reset_index()
        df = df[df['R'] == region]
        x_labels = df['t'].astype(str)
        scenario_columns = [col for col in df.columns if col not in ['t', 'R']]

    x = np.arange(len(x_labels))
    fig, ax = plt.subplots()

    if draw not in ['line', 'bar']:
        raise ValueError("Parameter 'draw' must be 'line' or 'bar'")

    if draw == 'line':
        for scenario in scenario_columns:
            ax.plot(x, df[scenario], marker='o', label=scenario)

    elif draw == 'bar':
        n_scenarios = len(scenario_columns)
        width = 0.8 / n_scenarios
        for i, scenario in enumerate(scenario_columns):
            offset = (i - n_scenarios / 2) * width + width / 2
            ax.bar(x + offset, df[scenario], width, label=scenario)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1))
    ax.set_title(f"{lib['Yaxis'][attr]} from {sector} in {region}")
    ax.set_xlabel('year')
    ax.set_ylabel(f'{lib['Yaxis'][attr]} in {lib['Unit'][attr]}')
    plt.tight_layout()
    plt.show()

    return df


