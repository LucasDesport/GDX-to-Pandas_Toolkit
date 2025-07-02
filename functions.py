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
from library import data_elec_map
from library import data_nrj_map
from library import data_emis_map
from library import regions_dict

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
    dfd = dfd[(dfd['Year'] >= start_year) & (dfd['Year'] <= end_year)]

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

def emis(dfd, region='global', horizon=2100, saveplt=False):
    '''
    Plot global or regional emissions profiles from different scenarios.
    '''

    # Select and convert relevant emission variables
    df = dfd[dfd['Attribute'].isin(data_emis_map)].copy()
    df['Value'] = df['Value'] / 1000  # Convert to GtCO₂eq
    df = df[pd.to_numeric(df['Year'], errors='coerce') <= horizon]

    if region != 'global':
        region_names = regions.loc[region, 'name']
        if isinstance(region_names, pd.Series):
            df = df[df['Region'].isin([region])]
        else:
            df = df[df['Region'] == region] 

    # Pivot and rename
    pv = df.pivot_table(index=['Scenario', 'Year'], columns='Attribute', values='Value', aggfunc='sum', sort=False)
    pv.rename(columns={k: v['label'] for k, v in data_emis_map.items()}, inplace=True)
    pv = pv.reset_index()

    # Ordered labels & colors
    column_order = [v['label'] for v in data_emis_map.values()]
    color_order = [v['color'] for v in data_emis_map.values()]
    
    # Plot
    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    pv[column_order].plot(kind='bar', stacked=True, figsize=(14, 6), color=color_order, ax=ax)
    ax, ax2 = plot_settings(pv, ax)

    # Labels and titles
    if region != 'global':
        title_region = ', '.join(region_names) if isinstance(region_names, pd.Series) else region_names
        plt.title(f"Emissions profile of {title_region}", weight='bold')
    else:
        plt.title("Global emissions profile", weight='bold')

    ax.set_ylabel("Emissions [GtCO₂eq]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    plt.tight_layout()
    plt.gca().yaxis.set_major_locator(MultipleLocator(5))

    if saveplt:
        plt.savefig(Path(f"{region}_emission_profile.png"), dpi=300, bbox_inches='tight')

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

def plot_grt(attr, sector, region, dfs, horizon=2100, index=False, draw='bar'):
    '''
    Compare across scenarios parameters definef by their sector G, region R, and time such as agy(g,r,t)
    '''
    
    df = grt(attr, sector, region, dfs, horizon)

    x_labels = df['t'].astype(str)
    scenario_columns = [col for col in df.columns if col != 't']
    x = np.arange(len(x_labels))
    
    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)

    if index==True:
        for scen in scenario_columns:    
            df[scen] = df[scen] / df[scen].iloc[0] * 100
    else:
        None

    if draw == 'line':
        for scenario in scenario_columns:
            ax.plot(x, df[scenario], label=scenario)
    elif draw == 'bar':
        n_scenarios = len(scenario_columns)
        bar_width = 0.8 / n_scenarios
        for i, scenario in enumerate(scenario_columns):
            offset = (i - n_scenarios / 2) * bar_width + bar_width / 2
            ax.bar(x + offset, df[scenario], bar_width, label=scenario)
    else:
         raise ValueError("Parameter 'draw' must be 'line' or 'bar'")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1))
    ax.set_title(f"{lib['Yaxis'][attr]} of {sectors['name'][sector]} in {region}")
    ax.set_xlabel('year')
    ax.set_ylabel(f"{lib['Yaxis'][attr]} {f"Index=100" if index == True else f"in {lib['Unit'][attr]}"}")
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
    
def sci(sector, region, dfs, dfd, horizon=2100, scope2: bool=False, ghg: bool=False, saveplt: bool=False):
    '''
    Plot sectoral carbon intensity pathways across scenarios
    '''
        
    bco2 = dfs['BCO2'].copy()
    ghgky = dfs['ghgky'].copy()
    ghgky['Value_CO2eq'] = ghgky['Value'] * ghgky['GHG'].map(gwp_100y) / 1000

    emis_scope1 = sum([grt('sco2', sector, region, dfs, horizon=horizon)])
    
    if sector in ['EINT', 'NMM', 'OIL', 'GAS']:
        emis_scope1 += grt('etotco2', sector, region, dfs, horizon=horizon)
    
    if ghg:
        ghg_sector = ghgky[(ghgky['R'] == region) & (ghgky['*'] == sector)]
        ghg_df = ghg_sector.pivot_table(index='t', columns='Scenario', values='Value_CO2eq', aggfunc='sum', sort=False).reset_index()
        emis_scope1 += ghg_df
    
    emis_scope2 = emis_scope1.copy()
    if scope2:
        emis_scope2 += s2(sector, region, dfs, horizon=horizon)
    if sector == 'ELEC':
        bco2_sector = bco2[bco2['R'] == region].pivot_table(index='t', columns='Scenario', values='Value', aggfunc='sum', sort=False).reset_index()
        emis_scope2 += bco2_sector


    prod = grt('agy', sector, region, dfs, horizon=horizon)
    ci_t = prod['t']  # the timeline to keep in sync
    
    ci_scope1 = compute_intensity(emis_scope1, prod, dfs, dfd, sector, region, conv_R, ci_t)
    ci_scope2 = compute_intensity(emis_scope2, prod, dfs, dfd, sector, region, conv_R, ci_t)

    ci = ci_scope1.copy()

    scenarios = [col for col in ci.columns if col != 't']

    x_labels = ci['t'].astype(str)
    x = np.arange(len(x_labels))

    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)

    for scenario in scenarios:
        if scope2:
            ax.plot(x, ci_scope1[scenario], linestyle='--', color='gray', label=f"{scenario} Scope 1")
            ax.plot(x, ci_scope2[scenario], label=f"{scenario} Scope 1+2")
            ax.fill_between(x, ci_scope1[scenario], ci_scope2[scenario],
                        where=ci_scope2[scenario] > ci_scope1[scenario],
                        interpolate=True, alpha=0.3, label=f"{scenario} Scope 2")
        else:
            ax.plot(x, ci_scope1[scenario], label=f"{scenario} (Scope 1)")
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend(loc='upper left', bbox_to_anchor=(1.005, 1))
    ax.set_title(f"Carbon intensity (scope 1{f"+2" if scope2 == True else ''}) of {sectors.loc[sector, 'name']} in {regions.loc[region, 'name']}")
    ax.set_xlabel('year')
    if sector == 'ELEC':
        ax.set_ylabel(f"carbon intensity in kgCO₂{f"eq" if ghg == True else ''}/MWh")
    elif sector in ['I_S', 'NMM']:
        ax.set_ylabel(f"carbon intensity in kgCO₂{f"eq" if ghg == True else ''}/t")
    else:
        ax.set_ylabel(f"carbon intensity in kgCO₂{f"eq" if ghg == True else ''}/GJ")
    ax.set_ylim(0)
    plt.tight_layout()
    plt.savefig(f"sci_{sector}_{region}_CO2{f"eq" if ghg is True else ''}_scope1{f"&2" if scope2 is True else ''}.png") if saveplt is True else None
    plt.show()
    
def trade(sector, region, flow, dfs, agg='region', net=False, index=False, horizon=2100, percent_stack=False):

    df = dfs['trad_t'].copy()
    df = df[(df['G'] == sector) & (df['t'] <= horizon)]
    df = df.rename(columns={'t': 'Year'})

    if agg == 'scenario':
        impo = df[df['R'] == region].pivot_table(index=['Year'], columns='Scenario', values='Value', aggfunc='sum', sort=False).reset_index()
        expo = df[df['RR'] == region].pivot_table(index=['Year'], columns='Scenario', values='Value', aggfunc='sum', sort=False).reset_index()
        df = df.pivot_table(index=['Year'], columns='Scenario', values='Value', aggfunc='sum', sort=False).reset_index()
        if net == True:
            for scen in df.drop(columns=['Year']).columns:
                if flow == 'imports':
                    df[scen] = impo[scen] if index == True else impo[scen] - expo[scen]
                elif flow =='exports':
                    df[scen] = expo[scen] if index == True else expo[scen] - impo[scen]
                else:
                    raise("Error: flow should be either imports' or 'exports'")

        x = df['Year'].astype(str)
    
        width, height = mpl.rcParams["figure.figsize"]
        fig, ax = plt.subplots(figsize=(width, height), dpi=300)

        scenarios = [col for col in df.columns if col not in ['Year']]

        for scen in scenarios:
            if index==True:
                df[scen] = df[scen] / df[scen].iloc[0] * 100
            else:
                None
            ax.plot(x, df[scen], label=scen)

    else:
        if net == True:
            impo = df[df['R'] == region].pivot_table(index=['Scenario', 'Year'], columns='RR', values='Value', aggfunc='sum', sort=False).reset_index()
            expo = df[df['RR'] == region].pivot_table(index=['Scenario', 'Year'], columns='R', values='Value', aggfunc='sum', sort=False).reset_index()
            df = df[df['R'] == region].pivot_table(index=['Scenario', 'Year'], columns='RR', values='Value', aggfunc='sum', sort=False).reset_index()
            for reg in df.drop(columns=['Scenario', 'Year']).columns:
                if flow == 'imports':
                    df[reg] = impo[reg] - expo[reg]
                elif flow =='exports':
                    df[reg] = expo[reg] - impo[reg]
                else:
                    raise("Error: flow should be either imports' or 'exports'")
    
        else:
            if flow == 'imports':
                df = df[df['R'] == region]
                df = df.pivot_table(index=['Scenario', 'Year'], columns='RR', values='Value', aggfunc='sum', sort=False)
            elif flow =='exports':
                df = df[df['RR'] == region]
                df = df.pivot_table(index=['Scenario', 'Year'], columns='R', values='Value', aggfunc='sum', sort=False)
            else:
                raise("Error: flow should be either imports' or 'exports'")

            df = df.reset_index()

        if percent_stack:
            non_numeric_cols = ['Scenario', 'Year', 'Index']
            numeric_cols = [col for col in df.columns if col not in non_numeric_cols]
            row_sums = df[numeric_cols].sum(axis=1)
            df[numeric_cols] = df[numeric_cols].div(row_sums, axis=0) * 100

            
        df.rename(columns={k: v['name'] for k, v in regions_dict.items()}, inplace=True)

        width, height = mpl.rcParams["figure.figsize"]
        fig, ax = plt.subplots(figsize=(width, height), dpi=150)
        ax, ax2 = plot_settings(df, ax)
    
        x = np.arange(len(df))

        bottom_pos = np.zeros(len(df))
        bottom_neg = np.zeros(len(df))
        
        for col in df.drop(columns=['Scenario', 'Year', 'Index'], errors='ignore').columns:
            meta = next((v for v in regions_dict.values() if v['name'] == col), {})
            
            values = np.nan_to_num(df[col].values)
            positive = np.where(values > 0, values, 0)
            negative = np.where(values < 0, values, 0)
        
            ax.bar(x,
                positive,
                bottom=bottom_pos,
                label=col,
                color=meta.get('color', '#999999'),
                hatch=meta.get('hatch', '') or '',
                edgecolor='white' if meta.get('hatch') else meta.get('color'),
                alpha=0.7 if meta.get('hatch') else 1.0,
                linewidth=1)
            ax.bar( x,
                negative,
                bottom=bottom_neg,
                #label=col,
                color=meta.get('color', '#999999'),
                hatch=meta.get('hatch', '') or '',
                edgecolor='white' if meta.get('hatch') else meta.get('color'),
                alpha=0.7 if meta.get('hatch') else 1.0,
                linewidth=1)
            
            bottom_pos += positive
            bottom_neg += negative
        
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    if percent_stack == True:
        ax.set_ylabel(f"Relative {flow} [%]")
        plt.title(f"Breakdown of {f"net " if net == True else f"gross "}{sectors.loc[sector, 'name']} {flow} in {regions.loc[region, 'name']}")
    else:
        ax.set_ylabel(f"{f"Relative" if percent_stack == True else f""} Trade{f" [BUS$]" if index == False else f" Index=100"}")
        plt.title(f"{f"Net " if net == True else f"Gross "}{sectors.loc[sector, 'name']} {flow} in {regions.loc[region, 'name']}")
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set_title(ax.get_title(), fontsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
    ax.tick_params(axis='both', labelsize=8)
    
    legend = ax.get_legend()
    if legend:
        legend.set_title(agg, prop={'size': 12})
        for text in legend.get_texts():
            text.set_fontsize(11)
            

    plt.show()

def nrj(dfd, region='global', horizon=2100, saveplt=False):
    '''
    Plot global or regional primary energy use.
    '''

    # Filter and prepare data
    df = dfd[dfd['Attribute'].isin(data_nrj_map)].copy()
    df = df[pd.to_numeric(df['Year'], errors='coerce') <= horizon]

    if region != 'global':
        region_names = regions.loc[region, 'name']
        if isinstance(region_names, pd.Series):
            df = df[df['Region'].isin(region)]
        else:
            df = df[df['Region'] == region]

    # Pivot and rename
    pv = df.pivot_table(index=['Scenario', 'Year'], columns='Attribute', values='Value', aggfunc='sum', sort=False)
    pv.rename(columns={k: v['label'] for k, v in data_nrj_map.items()}, inplace=True)
    pv = pv.reset_index()

    # Prepare colors in column order
    column_order = [v['label'] for v in data_nrj_map.values()]
    color_order = [data_nrj_map[k]['color'] for k in data_nrj_map]

    fig, ax = plt.subplots(dpi=300, constrained_layout=True)
    df_plot = pv[column_order]  # ensure order matches color list
    df_plot.plot(kind='bar', stacked=True, ax=ax, color=color_order)
    ax, ax2 = plot_settings(pv, ax)

    # Labels and title
    if region != 'global':
        region_names = regions.loc[region, 'name']
        title_region = ', '.join(region_names) if isinstance(region_names, pd.Series) else region_names
        plt.title(f"Primary energy in {title_region}", weight='bold')
    else:
        plt.title("Global primary energy", weight='bold')

    ax.set_ylabel("Energy [EJ]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(50))

    # Save if needed
    if saveplt:
        plt.savefig(Path(f"{region}_primary_energy.png"), dpi=300, bbox_inches='tight')

    plt.show()
    
def elec(dfd, region='global', horizon=2100, saveplt=False):

    '''
    Plot electricity generation.
    '''

    # Select energy variables
    df = dfd[dfd['Attribute'].isin(data_elec_map)].copy()
    df = df[pd.to_numeric(df['Year'], errors='coerce') <= horizon]

    if region != 'global':
        region_names = regions.loc[region, 'name']
        if isinstance(region_names, pd.Series):
            df = df[df['Region'].isin(region)]
        else:
            df = df[df['Region'] == region]

    pv = df.pivot_table(index=['Scenario', 'Year'], columns='Attribute', values='Value', aggfunc='sum', sort=False)
    pv.rename(columns={k: v['label'] for k, v in data_elec_map.items()}, inplace=True)
    pv = pv.reset_index()
    
    # Plot
    fig, ax = plt.subplots(dpi=300, constrained_layout=True)
    ax, ax2 = plot_settings(pv, ax)

    bottom = np.zeros(len(pv))
    x = np.arange(len(pv))

    pv['Index'] = pv['Scenario'] + '-' + pv['Year'].astype(str)
    scenarios = pv['Scenario'].unique().tolist()

    for col in pv.drop(columns=['Scenario', 'Year', 'Index']).columns:
        meta = next((v for v in data_elec_map.values() if v['label'] == col), {})
        values = np.nan_to_num(pv[col].values)
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=col,
            color=meta.get('color', '#999999'),
            hatch=meta.get('hatch', '') or '',
            edgecolor='white' if meta.get('hatch') else meta.get('color'),
            alpha=0.7 if meta.get('hatch') else 1.0,
            linewidth=1
        )
        bottom += values

    # Labels, legend, etc.
    if region != 'global':
        region_names = regions.loc[region, 'name']
        if isinstance(region_names, pd.Series):
            title_region = ', '.join(region_names)
        else:
            title_region = region_names
        plt.title(f"Electricity generation mix in {title_region}", weight='bold')
    else:
        plt.title("Global electricity generation mix", weight='bold')
    ax.set_ylabel("Electricity [TWh]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    
    plt.savefig(Path(f"{region}_electricity.png"), dpi=300, bbox_inches='tight') if saveplt == True else None
    
    plt.show()
    
def ggdp(dfd, agg="scenario", region='global', horizon=2100):
    '''
    Plot global GDP by scenario and year.
    When aggregating over scenarios (agg='scenario') the plot compares the global GDP across scenarios
    When aggregating over regions, the plot stacked regions GDP (to be used for a datafrane with a single scenario
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
            ax.set_title(f"GDP in {regions_dict[region]['name']}")

    elif agg=='region':
        pv = df.pivot_table(index=['Scenario', 'Year'], columns='Region', values='Value', aggfunc='sum', sort=False)
        pv = pv.reset_index()
        pv.rename(columns={k: v['name'] for k, v in regions_dict.items()}, inplace=True)

        region_columns = pv.drop(columns=['Scenario', 'Year']).columns
        name_to_color = {v['name']: v['color'] for v in regions_dict.values()}
        colors = [name_to_color.get(name, '#CCCCCC') for name in region_columns]

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

def plot_egrt(sector, region, dfs, horizon=2100):

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

    df = dfs['ee_sector'].copy()
    df = df[pd.to_numeric(df['t'], errors='coerce') <= horizon]
    
    if region == 'global':
        df = df[df['G'] == sector]
        df = df.groupby(['t', 'Scenario', 'e'], as_index=False, sort=False)['Value'].sum()
        df = df.pivot_table(index=['Scenario', 't'], columns='e', values='Value', sort=False).reset_index()
    else:
        df = df[(df['R'] == region) & (df['G'] == sector)]
        df = df.pivot_table(index=['Scenario', 't', 'R'], columns='e', values='Value', sort=False).reset_index()
        df = df.drop(columns=['R'])
   
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

def nmm(dfs, region='global', horizon=2100):

    df = dfs['nmm_eppa'].copy()
    df = df[pd.to_numeric(df['t'], errors='coerce') <= horizon]
    
    if region == 'global':
        df = df.groupby(['t', 'Scenario', '*'], as_index=False, sort=False)['Value'].sum()
        df = df.pivot_table(index=['Scenario', 't'], columns='*', values='Value', sort=False).reset_index()
    else:
        df = df[(df['R'] == region)]
        df = df.pivot_table(index=['Scenario', 't', 'R'], columns='*', values='Value', sort=False).reset_index()
        df = df.drop(columns=['R'])
   
    df.rename(columns={'noccs': 'Conventional', 'ccs': 'CCS'}, inplace=True)
    df.rename(columns={'t': 'Year'}, inplace=True)

   # Plot
    fig, ax = plt.subplots(dpi=300, constrained_layout=True)
    df_plot = df.drop(columns=['Scenario', 'Year']).plot(kind='bar', stacked=True, ax=ax)
    ax, ax2 = plot_settings(df, ax)

    # Labels, legend, etc.
    plt.title(f"{f"Global cement production" if region == 'global' else f"Cement production in {regions.loc[region, 'name']}"}")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel(f"Cement [Mt]")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
    
    plt.show()

def compute_intensity(emis_df, prod, dfs, dfd, sector, region, conv_R, ci_t):
    ejoe = dfs['ejoe'].copy()
    ci = emis_df.copy()
    ci['t'] = ci_t  # restore correct timeline
    ci = ci[pd.to_numeric(ci['t'], errors='coerce') <= 2100]

    scenarios = [col for col in ci.columns if col != 't']

    for scen in scenarios:
        if sector == 'ELEC':
            elec = dfd[(dfd['Attribute'] == '28a_TOTAL ELEC (TWh)') & 
                       (dfd['Region'] == region) & 
                       (dfd['Scenario'] == scen)].copy()
            elec = elec.sort_values('Year').reset_index(drop=True)
            elec = elec[elec['Year'].isin(ci['t'])]
            ci[scen] = ci[scen] / elec['Value'].values * 1000
        elif sector in ['NMM', 'I_S']:
            ci[scen] = ci[scen] / (prod[scen] / conv_R.loc[sector, 'USA']) * 1000
        else:
            factor = ejoe[(ejoe['*'] == sector) & (ejoe['R'] == region)]['Value']
            factor = factor.iloc[0] if not factor.empty else 1
            ci[scen] = ci[scen] / prod[scen] * factor

    return ci

def steel(region, dfs, plot_dim='1d', comm=['supply','demand'], flow=['output', 'imports', 'exports', 'demand'], index=False, horizon=2100, years=[], saveplt=False):

    df = dfs['track_is'].copy()
    df.columns=['Region','Year','comm','flow', 'Value', 'Scenario']
    df = df[(df['Year'] <= horizon) & (df['Region'] == region)]
    if years:
        df = df[df['Year'].isin(years)]
    if comm:
        df = df[df['comm'].isin(comm)]
    if flow:
        df = df[df['flow'].isin(flow)]

    width, height = mpl.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)

    if plot_dim == '1d':
        df = df.pivot_table(index=['Year'], columns='Scenario', values = 'Value', sort=False).reset_index(drop=False)

        x = df['Year'].astype(str)
    
        scenarios = [col for col in df.columns if col not in ['Year']]
    
        for scen in scenarios:
            if index==True:
                df[scen] = df[scen] / df[scen].iloc[0] * 100
            else:
                None
            ax.plot(x, df[scen], label=scen)

        ax.set_ylabel(f"Value [B USD]")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
        ax.set_ylim(0)
        plt.tight_layout()
        plt.title(f"Steel {flow[0]} in {regions.loc[region, 'name']} across scenarios")
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        plt.gca().yaxis.set_major_locator(MultipleLocator(5))
        
    elif plot_dim == '2d':
        pv = df.pivot_table(index=['Scenario', 'Year'], columns='flow', values='Value', sort=False).reset_index(drop=False)
    
        ax, ax2 = plot_settings(pv, ax)
    
        demand = pv['demand'].values # extract the demand to be plotted seperately
        pv['exports'] *= -1
        
        x = np.arange(len(pv))
    
        bar_cols_pos = ['output', 'imports']
        bar_cols_neg = ['exports']
  
        # Positive flows stacked upwards
        bottom = np.zeros(len(pv))
        for col in bar_cols_pos:
            values = np.nan_to_num(pv[col].values)
            ax.bar(x, values, bottom=bottom, label=col)
            bottom += values
        
        # Negative flows stacked downward (separately)
        bottom_neg = np.zeros(len(pv))
        for col in bar_cols_neg:
            values = np.nan_to_num(pv[col].values)
            ax.bar(x, values, bottom=bottom_neg, label=col)
            bottom_neg += values
    
        ax.scatter(x, demand, color='red', label='Demand', zorder=10)
    
        ax.set_xticks(x)
        ax.set_xticklabels(pv['Year'], rotation=45, ha='right')
    
        ax.legend()
        fig.tight_layout()

        ax.set_ylabel(f"Value [B USD]")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.005, 1))
        plt.title(f"Steel supply and demand in {regions['name'][region]} across scenarios")
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        plt.gca().yaxis.set_major_locator(MultipleLocator(5))

    else:
        print("Error: the attribute 'plot' should be iether '1d' or '2d'")

    if saveplt:
        plt.savefig(Path(f"steel_{region}_{flow}.png"), dpi=300, bbox_inches='tight')
    