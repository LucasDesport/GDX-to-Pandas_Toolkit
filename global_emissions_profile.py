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

# # Global emissions profile

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path


def gemis():

    with open("dfd.pkl", "rb") as f:
        df = pickle.load(f)

    # Select emissions variables
    df = df[df['Attribute'].isin(['14a_GHGinCO2eq (million ton)',
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

    #return fig, ax


