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
import plotly.express as px

import importlib
import scenmap
from library import lib
from library import sectors
from library import regions
from library import conv_R
from library import gwp_100y

import matplotlib as mpl
from matplotlib.lines import Line2D
import scienceplots

# Choose a style. You can use 'science', 'nature', 'ieee', etc.
mpl.style.use(['science', 'nature'])
mpl.rcParams['text.usetex'] = False

def gdx2dfs(
    scenario_paths: Dict[str, str],
    time_range: Tuple[int, int] = (2020, 2100),
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

    dfs['sco2'].columns = ['t', 'G', 'R', 'Value', 'Scenario'] #to make this specific parameter fit with others
    dfs['ACCA'].columns = ['R', 'G', 't', 'Value', 'Scenario'] #to make this specific parameter fit with others
    dfs['etotco2'].columns = ['G', 'R', 't', 'Value', 'Scenario'] #to make this specific parameter fit with others

    start_year, end_year = time_range

    for key, df in dfs.items():
        if 't' in df.columns:
            df['t'] = pd.to_numeric(df['t'], errors='coerce')
            dfs[key] = df[(df['t'] >= start_year) & (df['t'] <= end_year)]

    dfd = dfs['data']
    dfd.columns = ['Attribute', 'Year', 'Region', 'Value', 'Scenario']
    dfd['Year'] = pd.to_numeric(dfd['Year'], errors='coerce')

    dfd.to_csv('data.csv', index=False)
    
    return dfs, dfd

def plot_settings(pv, ax):
    # Create a single string index for plotting
    pv['Index'] = pv['Scenario'] + '-' + pv['Year'].astype(str)
    pv = pv.set_index('Index') 
    
    # Extract scenarios and years
    scenarios = pv['Scenario'].unique().tolist()
    years = sorted(pv['Year'].unique())

    # === DYNAMIC FIGURE SIZE BASED ON DATA ===
    num_bars = pv.shape[0]
    num_stacks = pv.shape[1] - 2 

    width = max(8, min(0.25 * num_bars, 30))
    height = max(4, min(0.5 * num_stacks, 10))
    ax.figure.set_size_inches(width, height)

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
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax2.tick_params(axis='x', which='both', length=0)
        ax2.spines['top'].set_visible(False)
        ax2.set_frame_on(False)
        ax2.xaxis.set_label_position('bottom')
        ax2.xaxis.set_ticks_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 25))

    ax.set_xlabel("")
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    return ax, ax2

# # Global emissions profile

def gemis(dfd, horizon=2100):

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
    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    pv.drop(columns=['Scenario', 'Year']).plot(kind='bar', stacked=True, figsize=(14, 6), color=colors, ax=ax)

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

def pemis(dfd, emis_type: str, horizon=2100): #emis_type can only be 'co2' or 'ghg'
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

    df = df[pd.to_numeric(df['Year'], errors='coerce') <= horizon]
    pv = df.pivot_table(index=['Year'], columns='Scenario', values='Value', aggfunc='mean')
    pv = pv.reset_index()
    
    # Plot
    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    pv.drop(columns=['Year']).plot(kind='line', stacked=False, figsize=(5, 3), colormap='tab20', ax=ax)
    
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

def grt(attr, sector, region, dfs, horizon=2100):

    if region == 'global':
        df = dfs[attr][(dfs[attr]['G'] == sector)].copy() 
    else:
        df = dfs[attr][(dfs[attr]['G'] == sector) & (dfs[attr]['R'] == region)].copy()

    df = df[pd.to_numeric(df['t'], errors='coerce') <= horizon]
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

def plot_grt(attr, sector, region, dfs, horizon=2100, draw='bar'):
    '''
    Compare across scenarios parameters definef by their sector G, region R, and time such as agy(g,r,t)
    '''
    
    df = grt(attr, sector, region, dfs, horizon)

    x_labels = df['t'].astype(str)
    scenario_columns = [col for col in df.columns if col != 't']
    x = np.arange(len(x_labels))
    
    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)

    if draw not in ['line', 'bar']:
        raise ValueError("Parameter 'draw' must be 'line' or 'bar'")

    if draw == 'line':
        for scenario in scenario_columns:
            ax.plot(x, df[scenario], marker='o', label=scenario)
    elif draw == 'bar':
        n_scenarios = len(scenario_columns)
        bar_width = 0.8 / n_scenarios
        for i, scenario in enumerate(scenario_columns):
            offset = (i - n_scenarios / 2) * bar_width + bar_width / 2
            ax.bar(x + offset, df[scenario], bar_width, label=scenario)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1))
    ax.set_title(f"{lib['Yaxis'][attr]} of {sectors['name'][sector]} in {region}")
    ax.set_xlabel('year')
    ax.set_ylabel(f'{lib['Yaxis'][attr]} in {lib['Unit'][attr]}')
    plt.tight_layout()
    plt.show()

def s2(sector, region, dfs, horizon=2100):
    '''
    Returns a dataframe with scope 2 emissions for a given sector across scenarios
    '''
    elec_cons = dfs['ei_t'].drop(columns=['e']) # this is the consumption of energy intermediates in production blocks d(g,r) # e is dropped because EPPA already ilters over ELEC
    elec_cons = elec_cons[elec_cons['R'].isin([region])] #filters over the desired region
    elec_cons = elec_cons.pivot_table(index=['G','t'], columns=['Scenario'], values='Value', sort=False).reset_index()
    elec_cons = elec_cons[(elec_cons['G'].isin([sector])) & (pd.to_numeric(elec_cons['t'], errors='coerce') <= horizon)].drop(columns='G') # filters over the desired sector
    elec_cons = elec_cons.reset_index().drop(columns='index')

    emis_elec = grt('sco2', 'ELEC', region, dfs)
    prod_elec = grt('agy', 'ELEC', region, dfs)

    selec_co2 = emis_elec.copy()
    selec_co2 = selec_co2[pd.to_numeric(selec_co2['t'], errors='coerce') <= horizon]
 
    for scen in selec_co2.drop(columns=['t']).columns:
        selec_co2[scen] = selec_co2[scen]/prod_elec[scen]
        
    # multiply by elec_cons to get the indirect emissions of electricity in the desired sector in MtCO2/$10B
    for i in selec_co2.drop(columns=['t']).columns:
        selec_co2[i] *= elec_cons[i] 

    return selec_co2
    
def sci(sector, region, dfs, dfd, horizon=2100, scope2: bool=False, ghg: bool=False, draw='bar', saveplt: bool=False):
    '''
    Plot sectoral carbon intensity pathways across scenarios
    '''
        
    bco2 = dfs['BCO2'].copy()
    ghgky = dfs['ghgky'].copy()
    ghgky['Value_CO2eq'] = ghgky['Value'] * ghgky['GHG'].map(gwp_100y) / 1000
    emis = (grt('sco2', sector, region, dfs, horizon=horizon) 
            + (grt('etotco2', sector, region, dfs, horizon=horizon) if sector in ['EINT', 'NMM', 'OIL', 'GAS'] else 0) 
            + (s2(sector, region, dfs, horizon=horizon) if scope2 is True else 0)
            + (bco2[bco2['R']==region].pivot_table(index='t', columns='Scenario', values='Value', aggfunc='sum').reset_index() if sector =='ELEC' else 0)
            + (ghgky[(ghgky['R'] == region) & (ghgky['*'] == sector)].pivot_table(index='t', columns='Scenario', values='Value_CO2eq', aggfunc='sum').reset_index() if ghg == True else 0))
    
    prod = grt('agy', sector, region, dfs, horizon=horizon)
    emis['t'] = prod['t'] #to reset the correct initial timeline (it messes up something two lines above)
        
    ejoe = dfs['ejoe']

    ci = emis.copy()
    ci = ci[pd.to_numeric(ci['t'], errors='coerce') <= horizon]
    
    scenarios = [col for col in ci.columns if col not in ['t']]

    for scen in scenarios:
        if sector == 'ELEC':
            ci[scen] = ci[scen]/dfd[(dfd['Attribute'] == '28a_TOTAL ELEC (TWh)') & (dfd['Region'] == 'USA') & (dfd['Scenario'] == 'vref')].drop(columns=['Attribute','Region','Scenario']).reset_index(drop=True)['Value']*1000
        elif sector in ['NMM','I_S']:
            ci[scen] = ci[scen]/(prod[scen]/conv_R.loc[sector, 'USA'])*1000
        else:
            ci[scen] = ci[scen]/prod[scen]*(ejoe[(ejoe['*'] == sector) & (ejoe['R'] == region)]['Value'].iloc[0] if sector in ['ROIL','OIL','GAS','COAL'] else 1)

    x_labels = ci['t'].astype(str)
    x = np.arange(len(x_labels))

    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)

    if draw == 'bar':
        n_scenarios = len(scenarios)
        width = 0.8 / n_scenarios
        for i, scenario in enumerate(scenarios):
            offset = (i - n_scenarios / 2) * width + width / 2
            ax.bar(x + offset, ci[scenario], width, label=scenario)
    elif draw =='line':
        for scenario in scenarios:
            ax.plot(x, ci[scenario], label=scenario)
    else:
        print('Error: choose either bar or line for attribute draw')
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1))
    ax.set_title(f"Carbon intensity (scope 1{f"+2" if scope2 == True else f""}) of {sectors.loc[sector, 'name']} in {regions.loc[region, 'name']}")
    ax.set_xlabel('year')
    if sector == 'ELEC':
        ax.set_ylabel(f"carbon intensity in kgCO₂{f"eq" if ghg == True else f""}/MWh")
    elif sector in ['I_S', 'NMM']:
        ax.set_ylabel(f"carbon intensity in kgCO₂{f"eq" if ghg == True else f""}/t")
    else:
        ax.set_ylabel(f"carbon intensity in kgCO₂{f"eq" if ghg == True else f""}/GJ")
    ax.set_ylim(0)
    plt.tight_layout()
    plt.savefig(f"sci_{sector}_{region}_CO2{f"eq" if ghg is True else f""}_scope1{f"&2" if scope2 is True else f""}.png") if saveplt is True else None
    plt.show()
    
def leak(sector, region, dfs, horizon=2100):

    imp = grt('imflow',sector,region,dfs,horizon)
    exp = grt('exflow',sector,region,dfs,horizon)

    leakage = imp.copy()

    scenarios = [col for col in leakage.columns if col not in ['t']]

    for scen in scenarios:
        leakage[scen] = (imp[scen] - exp[scen])

    x_labels = leakage['t'].astype(str)
    x = np.arange(len(x_labels))

    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)

    n_scenarios = len(scenarios)
    width = 0.8 / n_scenarios
    for i, scenario in enumerate(scenarios):
        offset = (i - n_scenarios / 2) * width + width / 2
        ax.bar(x + offset, leakage[scenario], width, label=scenario)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1))
    ax.set_title(f"Trade leakage of {sectors['name'][sector]} in {regions.loc[region, 'name']}")
    ax.set_xlabel('year')
    ax.set_ylabel(f"Leakage in billion USD")
    plt.tight_layout()
    plt.show()

def nrj(dfd, horizon=2100):

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

    # set colors
    colors = [
    '#3B3B3B',  # Coal
    '#7F4F24',  # Oil
    '#C44536',  # Gas
    '#287D57',  # Bioenergy
    '#F2C849',  # Nuclear
    '#2E86AB',  # Hydro
    '#91C499',  # renewables
    ]

    df = df[pd.to_numeric(df['Year'], errors='coerce') <= horizon]
    
    # Pivot and rename
    pv = df.pivot_table(index=['Scenario', 'Year'], columns='Attribute', values='Value', aggfunc='sum', sort=False) # keeps scenarios order as listed in scenario_map
    pv.columns = ['Coal', 'Oil', 'Gas', 'Bioenergy', 'Nuclear', 'Hydro', 'Renewables'] # rename emissions variables
    pv = pv.reset_index()

    fig, ax = plt.subplots(dpi=300, constrained_layout=True)
    df_plot = pv.drop(columns=['Scenario', 'Year']).plot(kind='bar', stacked=True, ax=ax, color=colors)
    ax, ax2 = plot_settings(pv, ax)

    # Labels, legend, etc.
    plt.title("Global primary energy")
    ax.set_ylabel("Energy [EJ]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(50))
    
    # plt.savefig(Path("global_primary_energy.png"))
    
    plt.show()

def gelec(dfd, horizon=2100):

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
        'Coal':                {'color': '#3B3B3B', 'hatch': None},
        'Coal with CCS':       {'color': '#3B3B3B', 'hatch': '...'},
        'Oil':                 {'color': '#7F4F24', 'hatch': None},
        'Gas':                 {'color': '#C44536', 'hatch': None},
        'Gas with CCS':        {'color': '#C44536', 'hatch': '...'},
        'Bioenergy':           {'color': '#287D57', 'hatch': None},
        'BECCS':               {'color': '#287D57', 'hatch': '...'},
        'Nuclear':             {'color': '#F2C849', 'hatch': None},
        'Hydro':               {'color': '#2E86AB', 'hatch': None},
        'Renewables':          {'color': '#91C499', 'hatch': None},
    }
    
    # Plot
    fig, ax = plt.subplots(dpi=300, constrained_layout=True)
    ax, ax2 = plot_settings(pv, ax)

    bottom = np.zeros(len(pv))
    x = np.arange(len(pv))

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
            edgecolor='white' if style['hatch'] else style['color'],
            alpha=0.7 if style['hatch'] else 1.0,
            linewidth=1
        )
        bottom += values

    # Labels, legend, etc.
    plt.title("Global electricity generation", weight='bold')
    ax.set_ylabel("Electricity [TWh]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    
    # plt.savefig(Path("global_electricity.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
def ggdp(dfd, agg="scenario", region='global', horizon=2100):
    '''
    Plot global GDP by scenario and year.
    '''
    df = dfd[dfd['Attribute'].isin(['01_GDP (billion US$)'])]
    df = df[pd.to_numeric(df['Year'], errors='coerce') <= horizon]
    df['Value'] = df['Value'] / 1000  # trillion dollars

    if agg=='scenario':
        if region!='global':
            df = df[df['Region'] == region]
        
        pv = df.pivot_table(index=['Year'], columns='Scenario', values='Value', aggfunc='sum', sort=False)
        pv = pv.reset_index()

        fig, ax = plt.subplots(dpi=300, constrained_layout=True)
        
        x_labels = pv['Year'].astype(str)
        scenario_columns = [col for col in pv.columns if col != 'Year']
        x = np.arange(len(x_labels))

        n_scenarios = len(scenario_columns)
        bar_width = 0.8 / n_scenarios
        for i, scenario in enumerate(scenario_columns):
            offset = (i - n_scenarios / 2) * bar_width + bar_width / 2
            ax.bar(x + offset, pv[scenario], bar_width, label=scenario)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1))

        if region=='global':
            ax.set_title(f"Global GDP")
        else:
            ax.set_title(f"GDP in {regions.loc[region, 'name']}")

    elif agg=='region':
        pv = df.pivot_table(index=['Scenario', 'Year'], columns='Region', values='Value', aggfunc='sum', sort=False)
        pv = pv.reset_index()
        pv.rename(columns={k: v['name'] for k, v in regions.items()}, inplace=True)

        region_columns = pv.drop(columns=['Scenario', 'Year']).columns
        colors = [regions.get(k, {}).get('color', '#CCCCCC') for k in regions if regions[k]['name'] in region_columns]

        fig, ax = plt.subplots(dpi=300, constrained_layout=True)
        pv.drop(columns=['Scenario', 'Year']).plot(kind='bar', stacked=True, color=colors, ax=ax)
        ax, ax2 = plot_settings(pv, ax)

        # Labels, legend, etc.
        plt.title("Gross domestic product", weight='bold')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))

    ax.set_ylabel("Trillion US$")
    
    #plt.savefig(Path("gdp.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_egrt(sector, region, dfs):

    '''
    Plot energy consumption by sector across scenarios over time.
    '''

    colors = [
        '#3B3B3B', #coal
        '#5A5A5A', #oil
        '#7F4F24', #refined oil
        '#C44536', #gas
        '#91C499', #electricity       
    ]
    
    if region == 'global':
        df = dfs['ee_sector'][dfs['ee_sector']['G'] == sector].copy()
        df = df.groupby(['t', 'Scenario', 'e'], as_index=False, sort=False)['Value'].sum()
        df = df.pivot_table(index=['Scenario', 't'], columns='e', values='Value', sort=False).reset_index()
    else:
        df =  dfs['ee_sector'][(dfs['ee_sector']['R'] == region) & (dfs['ee_sector']['G'] == sector)].copy()
        df = df.pivot_table(index=['Scenario', 't', 'R'], columns='e', values='Value', sort=False).reset_index()
        df = df.drop(columns=['R'])
   
    df = df[pd.to_numeric(df['t'], errors='coerce').notnull()]
    df.rename(columns=sectors['name'], inplace=True)
    df.rename(columns={'t': 'Year'}, inplace=True)

   # Plot
    fig, ax = plt.subplots(dpi=300, constrained_layout=True)
    df_plot = df.drop(columns=['Scenario', 'Year']).plot(kind='bar', stacked=True, ax=ax, color=colors)
    ax, ax2 = plot_settings(df, ax)

    # Labels, legend, etc.
    plt.title(f"Energy consumption of {sectors.loc[sector,'name']} in {regions.loc[region, 'name']}")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel("Energy [EJ]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    
    plt.show()

def data(attr, dfd, region='global', horizon=2100):
    '''
    Simple function designed to look at only one parameter of 'data' across scenarios
    '''

    df = dfd.copy()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df[(df['Attribute'] == attr) & (df['Year'] <= horizon)]

    if region != 'global':
        df = df[(df['Region'] == region)]
        df = df.pivot_table(index=['Year'], columns=['Scenario'], values='Value', sort=False)
    else:
        df = df.pivot_table(index=['Year'], columns=['Scenario'], values='Value', aggfunc='sum', sort=False)

    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    
    df.plot(kind='bar', stacked=False, width=0.8, ax=ax)

    plt.title(f"{attr} in {region}")
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))

    plt.show()

def ne_inputs(g,ne,R,dfs,horizon=2100):
    '''
    function created initially to look at intermediates inputs of OTHR and EINT
    '''

    df = dfs['AAI'].copy()
    df = df[pd.to_numeric(df['t'], errors='coerce') <= horizon]
    df['Value'] = df['Value'] * 10

    if R != 'global':
        pv = df[df['R'].isin([R])].pivot_table(index=['t'], columns=['Scenario'], values='Value', aggfunc='sum', sort=False).reset_index()
    else:
        pv = df.pivot_table(index=['t'], columns=['Scenario'], values='Value', aggfunc='sum', sort=False).reset_index()
    
    scenarios = df['Scenario'].unique().tolist()
    years = sorted(df['t'].unique())

    num_bars = df.shape[0]
    num_stacks = df.shape[1] - 2 

    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(dpi=300, figsize=(width, height), constrained_layout=True)
    pv_plot = pv.drop(columns=['t']).plot(kind='bar', stacked=False, ax=ax, width=0.8)

    # Calculate scenario label positions
    scenario_ranges = []
    for scenario in scenarios:
        indices = df[df['Scenario'] == scenario].index
        if len(indices) > 0:
            locs = [df.index.get_loc(i) for i in indices]
            midpoint_pos = sum(locs) / len(locs)
            scenario_ranges.append((midpoint_pos, scenario))

    year_labels = pv['t'].astype(str).to_list()
    ax.set_xticks(range(len(year_labels)))
    ax.set_xticklabels(year_labels, rotation=90)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1))
    fig.suptitle(f"Intermediate {sectors.loc[ne, 'name']} inputs in {regions.loc[R, 'name']} {sectors.loc[g, 'name']}", y=1.05)
    ax.set_ylabel('Input flow [B US$]')

    plt.show()

def ne_inputs_bd(g,R,dfs,horizon=2100):
    '''
    function created initially to look at the breakdown intermediates inputs of OTHR and EINT across scenarios
    '''

    df = dfs['AAI'].copy()
    df = df[df['G'].isin([g])]
    df = df[pd.to_numeric(df['t'], errors='coerce') <= horizon]
    df['Value'] = df['Value'] * 10

    if R != 'global':
        pv = df[df['R'].isin([R])].pivot_table(index=['Scenario', 't'], columns=['ne'], values='Value', aggfunc='sum', sort=False).reset_index()
    else:
        pv = df.pivot_table(index=['Scenario', 't'], columns=['ne'], values='Value', aggfunc='sum', sort=False).reset_index()

    pv.rename(columns={'t': 'Year'}, inplace=True)

   # Plot
    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(dpi=300, figsize=(width,height), constrained_layout=True)
    df_plot = pv.drop(columns=['Scenario', 'Year']).plot(kind='bar', stacked=True, ax=ax)
    ax, ax2 = plot_settings(pv, ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))

    year_labels = pv['Year'].astype(str).to_list()
    ax.set_xticks(range(len(year_labels)))
    ax.set_xticklabels(year_labels, rotation=90)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1))
    fig.suptitle(f'Intermediate inputs in {regions.loc[R, 'name']} {sectors['name'][g]}', y=1.02)
    ax.set_ylabel('Input flow [B US$]')

    plt.show()

def ne_inputs_bd_plotly(g, R, dfs, horizon=2100):
    '''
    Plotly version of ne_inputs_bd: percentage stacked bar chart
    '''
    df = dfs['AAI'].copy()
    df = df[df['G'].isin([g])]
    df = df[pd.to_numeric(df['t'], errors='coerce') <= horizon]
    df['Value'] = df['Value'] * 10  # Convert to B US$

    if R != 'global':
        pv = df[df['R'].isin([R])].pivot_table(index=['Scenario', 't'], columns=['ne'], values='Value', aggfunc='sum', sort=False).reset_index()
    else:
        pv = df.pivot_table(index=['Scenario', 't'], columns=['ne'], values='Value', aggfunc='sum', sort=False).reset_index()

    pv.rename(columns={'t': 'Year'}, inplace=True)

    df_long = pv.melt(id_vars=['Scenario', 'Year'], var_name='ne', value_name='Value')
    df_long['Value'] = df_long['Value'].fillna(0)

    total = df_long.groupby(['Scenario', 'Year'])['Value'].transform('sum')
    df_long['Percent'] = (df_long['Value'] / total) * 100

    # Plot
    fig = px.bar(
        df_long,
        x='Year',
        y='Percent',
        color='ne',
        text=df_long['Percent'].apply(lambda x: f"{x:.0f}%" if x > 5 else ""),
        facet_col='Scenario',
        title=f'Intermediate inputs in {R} for {sectors['name'][g]}',
        labels={'Percent': 'Percentage (%)', 'ne': 'Input'},
    )

    fig.update_traces(textposition='inside')
    fig.update_layout(
        barmode='stack',
        yaxis_range=[0, 100],
        yaxis_title='Input flow [%]',
        xaxis_title='Year',
        legend_title='Input',
        margin=dict(t=60, b=50),
        height=600
    )

    fig.show()

def sci_2scen(glist: list, dfs, horizon=2050):
    '''
    Compare two carbon intensity pathways across user-defined sectors (glist),
    normalized to an index of 100 at base year (first scenario).
    '''

    scenario_linestyles = ['-', '--']  # reference vs alternative
    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width*1.5, height), dpi=300)
    
    custom_legend = []

    for j, i in enumerate(glist):  # i = sector code
        color = sectors.loc[i,'color']
        name = sectors.loc[i,'name']
    
        emis = grt('sco2', i, 'USA', dfs)
        emis = emis[pd.to_numeric(emis['t'], errors='coerce') >= 2020]
        prod = grt('agy', i, 'USA', dfs)
        
        ci = emis.copy()
        ci = ci[pd.to_numeric(ci['t'], errors='coerce') <= horizon].reset_index(drop=True)

        scenarios = [col for col in ci.columns if col != 't']
        
        for scen in scenarios:
            ci[scen] = ci[scen] / prod[scen]
            ci[scen] = ci[scen] / ci[scen].iloc[0] * 100
            ci.rename(columns={scen: f"{scen}-{i}"}, inplace=True)

        scenarios = [col for col in ci.columns if col != 't']
        x_labels = ci['t'].astype(str)
        x = np.arange(len(x_labels))

        for k, scenario in enumerate(scenarios):
            linestyle = scenario_linestyles[k % len(scenario_linestyles)]
            ax.plot(
                x,
                ci[scenario],
                label=scenario,
                color=color,
                linestyle=linestyle,
            )

        # Add to legend
        custom_legend.append(Line2D([0], [0], color=color, label=name))
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_ylabel("Relative CI tCO2/USD (Index = 100)")
    ax.set_title("Carbon Intensity across sectors")

    # Legend outside the plot
    ax.legend(handles=custom_legend, loc='upper left', bbox_to_anchor=(1.005, 1))
    
    plt.tight_layout()
    plt.show()
