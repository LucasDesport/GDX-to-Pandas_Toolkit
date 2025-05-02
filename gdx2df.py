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
import pickle

with open("dfs.pkl", "wb") as f:
    pickle.dump(dfs, f)
# +
# Pre-treatment to create a subcategory of the dataframe including only the parameter 'data' 
dfd = dfs['data']
dfd.columns = ['Attribute', 'Year', 'Region', 'Value', 'Scenario']
dfd['Year'] = pd.to_numeric(dfd['Year'], errors='coerce')
dfd = dfd[dfd['Year'] <= 2100]

dfd.to_csv('output.csv', index=False)
# -





