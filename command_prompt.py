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

# %%
import importlib

import scenmap

import functions
#from functions import gdx2dfs #if you want to avoid calling 'functions' each time you compute a plot

# %%
importlib.reload(functions)

# %%
importlib.reload(scenmap)

# %% [markdown]
# # Import your results with the *gdx2dfs* function

# %%
dfs, dfd = functions.gdx2dfs(scenmap.myscen) # the function returns dfs with all parameters and dfd which is a filter on data

# %% [markdown]
# # Generate your plots

# %%
grt = functions.plot_grt('agy', 'NMM', 'USA', 'bar', dfs)

# %%
functions.gemis(dfd)

# %%
functions.pemis(dfd, 'co2')

# %%
