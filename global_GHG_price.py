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
from matplotlib.ticker import MultipleLocator


def emis_price(emis_type: str): #emis_type can only be 'co2' or 'ghg'

    with open("dfd.pkl", "rb") as f:
        df = pickle.load(f)

    if emis_type == 'co2':
        df = df[df['Attribute'].isin(['46_CO2 price (US$ per ton CO2)'])]
    elif emis_type == 'ghg':
        df = df[df['Attribute'].isin(['46a_CO2eq price (US$ per ton CO2eq)'])]
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
        
    return df



