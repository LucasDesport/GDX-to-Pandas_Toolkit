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
from library import sectors
from library import regions

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

def plot_settings(pv, ax):

    # Create a single string index for plotting
    pv['Index'] = pv['Scenario'] + '-' + pv['Year'].astype(str)
    pv = pv.set_index('Index') 
    
    # Extract scenarios' names and years
    scenarios = pv['Scenario'].unique().tolist()
    years = sorted(pv['Year'].unique())
    
    # Set xticks to year only (first axis)
    year_labels = pv['Year'].astype(str).to_list()
    ax.set_xticks(range(len(year_labels)))
    ax.set_xticklabels(year_labels, rotation=90)

    # Create second axis for scenarios
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    # Calculate scenario label positions
    scenario_ranges = []
    for scenario in scenarios:
        indices = pv[pv['Scenario'] == scenario].index
        if len(indices) > 0:
            locs = [pv.index.get_loc(i) for i in indices]
            midpoint_pos = sum(locs) / len(locs)
            scenario_ranges.append((midpoint_pos, scenario))

    if scenario_ranges:
        scenario_positions, scenario_labels = zip(*scenario_ranges)
        ax2.set_xticks(scenario_positions)
        ax2.set_xticklabels(scenario_labels)
        ax2.tick_params(axis='x', which='both', length=0)
        ax2.spines['top'].set_visible(False)
        ax2.set_frame_on(False)
        ax2.xaxis.set_label_position('bottom')
        ax2.xaxis.set_ticks_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 40))

    ax.set_xlabel("")
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    return ax, ax2

# # Global emissions profile

def gemis(dfd, horizon):

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

    df = df[pd.to_numeric(df['Year'], errors='coerce') <= horizon]

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

    # Plot
    ax = pv.drop(columns=['Scenario', 'Year']).plot(kind='bar', stacked=True, figsize=(14, 6), color=colors)

    ax, ax2 = plot_settings(pv, ax)


    # Labels, legend, etc.
    plt.title("Global emissions profile", weight='bold')
    ax.set_ylabel("Emissions [GtCO₂eq]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    plt.tight_layout()
    plt.gca().yaxis.set_major_locator(MultipleLocator(5)) # increment each 5 GtCO2eq
    
    #plt.savefig(Path("global_emission_profile.png"), dpi=300, bbox_inches='tight')
    
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
    
    # plt.savefig("global_emission_price.png", dpi=300, bbox_inches='tight')
    
    plt.show()

# # Exploring GRT variables

def grt(attr, sector, region, dfs):

    dfs['sco2'].columns = ['t', 'G', 'R', 'Value', 'Scenario'] #to make this specific parameter fit with others

    if region == 'global':
        df = dfs[attr][(dfs[attr]['G'] == sector)].copy() 
    else:
        df = dfs[attr][(dfs[attr]['G'] == sector) & (dfs[attr]['R'] == region)].copy()

    df['Value'] *= lib['Converter'][attr]
    df = df[pd.to_numeric(df['t'], errors='coerce').notnull()] #you can filter the years here by replace '.notnull()' with < YYYY

    if region == 'global':
        if lib['type'][attr] == 'price':
            df = df.groupby(['t', 'Scenario'], as_index=False, sort=False)['Value'].mean()
        else:
            df = df.groupby(['t', 'Scenario'], as_index=False, sort=False)['Value'].sum()
        df = df.pivot_table(index='t', columns='Scenario', values='Value', sort=False).reset_index()
    else:
        df = df.pivot_table(index=['t', 'R'], columns='Scenario', values='Value', sort=False).reset_index()
        df = df.drop(columns=['R'])
        
    return df

def plot_grt(attr, sector, region, draw, dfs):
    '''
    Compare across scenarios parameters definef by their sector G, region R, and time such as agy(g,r,t)
    '''
    
    df = grt(attr, sector, region, dfs)

    x_labels = df['t'].astype(str)
    scenario_columns = [col for col in df.columns if col != 't']

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
    ax.set_title(f"{lib['Yaxis'][attr]} of {sectors.get(sector, f"{sector}")} in {region}")
    ax.set_xlabel('year')
    ax.set_ylabel(f'{lib['Yaxis'][attr]} in {lib['Unit'][attr]}')
    plt.tight_layout()
    plt.show()

def plot_sci(sector, region, dfs):
    '''
    Plot sectoral carbon intensity pathways across scenarios
    '''

    emis = grt('sco2', sector, region, dfs)
    prod = grt('agy', sector, region, dfs)

    ci = emis.copy()
    
    scenarios = [col for col in ci.columns if col not in ['t']]

    for scen in scenarios:
        ci[scen] = ci[scen]/prod[scen]

    x_labels = ci['t'].astype(str)
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots()

    n_scenarios = len(scenarios)
    width = 0.8 / n_scenarios
    for i, scenario in enumerate(scenarios):
        offset = (i - n_scenarios / 2) * width + width / 2
        ax.bar(x + offset, ci[scenario], width, label=scenario)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1))
    ax.set_title(f"Carbon intensity of {sectors.get(sector, f"{sector}")} in {region}")
    ax.set_xlabel('year')
    ax.set_ylabel(f"carbon intensity in kgCO2/USD")
    plt.tight_layout()
    plt.show()

def plot_leak(sector, region, dfs):
    '''
    Plot trade leakages across scenarios
    '''

    imp = grt('imflow',sector,region,dfs)
    exp = grt('exflow',sector,region,dfs)

    leakage = imp.copy()

    scenarios = [col for col in leakage.columns if col not in ['t']]

    for scen in scenarios:
        leakage[scen] = (imp[scen] - exp[scen])*10

    x_labels = leakage['t'].astype(str)
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots()

    n_scenarios = len(scenarios)
    width = 0.8 / n_scenarios
    for i, scenario in enumerate(scenarios):
        offset = (i - n_scenarios / 2) * width + width / 2
        ax.bar(x + offset, leakage[scenario], width, label=scenario)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1))
    ax.set_title(f"Trade leakage of {sectors.get(sector, f"{sector}")} in {region}")
    ax.set_xlabel('year')
    ax.set_ylabel(f"Leakage in billion USD")
    plt.tight_layout()
    plt.show()

def nrj(dfd, horizon):

    '''
    Plot global ormary energy use.
    '''

    # Select emissions variables
    df = dfd[dfd['Attribute'].isin(['15_coal (EJ)',
                                    '16_oil (EJ)',
                                    '17_gas (EJ)',
                                    '18b_bioenergy (EJ)',
                                    '19_nuclear (EJ)',
                                    '19b_hydro (EJ)',
                                    '20_renewables (wind&solar) (EJ)'
                                   ])]

    df = df[pd.to_numeric(df['Year'], errors='coerce') <= horizon]
    
    # Pivot and rename
    pv = df.pivot_table(index=['Scenario', 'Year'], columns='Attribute', values='Value', aggfunc='sum', sort=False) # keeps scenarios order as listed in scenario_map
    pv.columns = ['Coal', 'Oil', 'Gas', 'Bioenergy', 'Nuclear', 'Hydro', 'Renewables'] # rename emissions variables
    pv = pv.reset_index()

    # set colors
    colors = [
    '#000000',  # Coal
    '#5A5A5A',  # Oil
    '#981C42',  # Gas
    '#298B45',  # Bioenergy
    '#FCF604',  # Nuclear
    '#1212E8',  # Hydro
    '#2AB8D0',  # renewables
    ]
    
    # Plot
    ax = pv.drop(columns=['Scenario', 'Year']).plot(kind='bar', stacked=True, figsize=(14, 6), color=colors)

    ax, ax2 = plot_settings(pv, ax)

    # Labels, legend, etc.
    plt.title("Global primary energy", weight='bold')
    ax.set_ylabel("Energy [EJ]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    plt.tight_layout()
    plt.gca().yaxis.set_major_locator(MultipleLocator(50)) # increment each 5 GtCO2eq
    
    # plt.savefig(Path("global_primary_energy.png"), dpi=300, bbox_inches='tight')
    
    plt.show()

def gelec(dfd, horizon):

    '''
    Plot global electricity generation.
    '''

    # Select energy variables
    df = dfd[dfd['Attribute'].isin(['22a_coal_no CCS (TWh)',
                                    '22b_coal_CCS (TWh)',
                                    '23_oil (TWh)',
                                    '24b_gas_CCS (TWh)',
                                    '24a_gas_no CCS (TWh)',
                                    '27a_bioelectricity and other (TWh)',
                                    '27a_bioelectricity_CCS (TWh)',
                                    '25_nuclear (TWh)',
                                    '26_hydro (TWh)',
                                    '27_renewables (wind&solar) (TWh)',
                                   ])]

    df = df[pd.to_numeric(df['Year'], errors='coerce') <= horizon]
    
    # Pivot and rename
    pv = df.pivot_table(index=['Scenario', 'Year'], columns='Attribute', values='Value', aggfunc='sum', sort=False) # keeps scenarios order as listed in scenario_map
    pv.columns = ['Coal', 'Coal with CCS', 'Oil', 'Gas', 'Gas with CCS', 'Bioenergy', 'BECCS', 'Nuclear', 'Hydro', 'Renewables'] # rename emissions variables
    pv = pv.reset_index()

    # set colors
    style_dict = {
        'Coal':                {'color': '#000000', 'hatch': None},
        'Coal with CCS':       {'color': '#FFFFFF', 'hatch': '//'},
        'Oil':                 {'color': '#5A5A5A', 'hatch': None},
        'Gas':                 {'color': '#981C42', 'hatch': None},
        'Gas with CCS':        {'color': '#981C42', 'hatch': '//'},
        'Bioenergy':           {'color': '#298B45', 'hatch': None},
        'BECCS':               {'color': '#298B45', 'hatch': '//'},
        'Nuclear':             {'color': '#FCF604', 'hatch': None},
        'Hydro':               {'color': '#1212E8', 'hatch': None},
        'Renewables':          {'color': '#2AB8D0', 'hatch': None},
    }
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(pv))
    x = np.arange(len(pv))

    ax, ax2 = plot_settings(pv, ax)

    pv['Index'] = pv['Scenario'] + '-' + pv['Year'].astype(str)
    scenarios = pv['Scenario'].unique().tolist()

    for col in pv.drop(columns=['Scenario', 'Year', 'Index']).columns:
        values = pv[col].values
        style = style_dict[col]
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=col,
            color=style['color'],
            hatch=style['hatch'] if style['hatch'] else '',
            edgecolor='black' if style['hatch'] else style['color'],
            linewidth=1
        )
        bottom += values

    # Labels, legend, etc.
    plt.title("Global electricity generation", weight='bold')
    ax.set_ylabel("Electricity [TWh]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    #plt.gca().yaxis.set_major_locator(MultipleLocator(25))
    plt.tight_layout()
    
    # plt.savefig(Path("global_electricity.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
def ggdp(dfd, horizon):

    '''
    Plot global GDP by scenario and year.
    '''

    df = dfd[dfd['Attribute'].isin(['01_GDP (billion US$)'])]
    df = df[pd.to_numeric(df['Year'], errors='coerce') <= horizon]
    df['Value'] = df['Value']/1000 #trillion dolars
    
    # Pivot and rename
    pv = df.pivot_table(index=['Scenario', 'Year'], columns='Region', values='Value', aggfunc='sum', sort=False) # keeps scenarios order as listed in scenario_map
    pv = pv.reset_index()
    pv.rename(columns=regions, inplace=True)
    
    # Plot
    ax = pv.drop(columns=['Scenario', 'Year']).plot(kind='bar', stacked=True, figsize=(14, 6))
    ax, ax2 = plot_settings(pv, ax)

    # Labels, legend, etc.
    plt.title("Gross domestic product", weight='bold')
    ax.set_ylabel("Trillion US$")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    plt.tight_layout()
    
    #plt.savefig(Path("global_gdp.png"), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_egrt(sector, region, dfs):

    '''
    Plot energy consumption by sector across scenarios over time.
    '''

    if region == 'global':
        df = dfs['ee_sector'][dfs['ee_sector']['G'] == sector].copy()
        df = df.groupby(['t', 'Scenario', 'e'], as_index=False, sort=False)['Value'].sum()
        df = df.pivot_table(index=['Scenario', 't'], columns='e', values='Value', sort=False).reset_index()
    else:
        df =  dfs['ee_sector'][(dfs['ee_sector']['R'] == region) & (dfs['ee_sector']['G'] == sector)].copy()
        df = df.pivot_table(index=['Scenario', 't', 'R'], columns='e', values='Value', sort=False).reset_index()
        df = df.drop(columns=['R'])
   
    df.rename(columns=library.sectors, inplace=True)
    df.rename(columns={'t': 'Year'}, inplace=True)

    # Plot
    ax = df.drop(columns=['Scenario', 'Year']).plot(kind='bar', stacked=True, figsize=(14, 6))

    ax, ax2 = plot_settings(df, ax)

    # Labels, legend, etc.
    plt.title(f"Energy consumption of {sectors.get(sector, f"{sector}")} in {regions.get(region, f"{region}")}", weight='bold')
    ax.set_ylabel("Energy [EJ]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    plt.tight_layout()

    plt.show()