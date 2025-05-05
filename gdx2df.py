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
import scenmap


def load_gdx_dfs(
    scenario_paths: Dict[str, str],
    time_range: Tuple[int, int] = (2014, 2100),
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Load parameters from multiple GDX files into pandas DataFrames,
    tagging each row with its scenario and optionally pivoting time dimensions.
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

    return dfs


importlib.reload(scenmap)
print(scenmap.scenario_map)

dfs = load_gdx_dfs(scenmap.scenario_map)

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





