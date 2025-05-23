# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Import and reload functions and scenarios anytime there is a change

# %% [markdown]
# ## Important: you need to create a file named *scenmap.py* which should contain a dictionary of your scenarios' names and GDX paths, such as:
#
# ```
# myscen = {'vref': 'C:\\Users\\username\\EPPA7\\results\\all_v-ref_p0_r0_gdpg-m_aeeg-m_sekl-m.gdx',
#           'ParisForever': 'C:\\Users\\username\\EPPA7\\results\\all_ParisForever.gdx'
# }
# ```
#
# The *scenmap.py* file will be ignored by Git to avoid user conflicts when merging.

# %%
import importlib

import scenmap

import functions as fn

# %%
importlib.reload(fn)

# %%
importlib.reload(scenmap)

# %% [markdown]
# # Import your results with the *gdx2dfs* function

# %%
dfs, dfd = fn.gdx2dfs(scenmap.myscen) # the function returns dfs with all parameters and dfd which is a filter on data

# %% [markdown]
# # Generate your plots

# %%
fn.plot_grt('agy', 'NMM', 'USA', 'bar', dfs)

# %%
fn.gemis(dfd, 2050)

# %%
fn.pemis(dfd, 'co2')

# %%
fn.plot_sci('I_S','USA',dfs)

# %%
fn.plot_leak('I_S','USA',dfs)

# %%
fn.nrj(dfd, 2100)

# %%
fn.gelec(dfd,2100)

# %%
fn.ggdp(dfd, 2050)

# %%
fn.plot_egrt('NMM', 'USA', dfs)

# %%
