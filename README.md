# GDX-to-Pandas Toolkit

This project transforms GAMS `.gdx` data files into `pandas` DataFrames and provides plotting functions for visual analysis. The project was built under Jupyter Lab but should work fine with VS Code or other python editors.

## Installation

Clone the repository or download the zipped project. 

```
git clone https://github.com/LucasDesport/GDX-to-Pandas_Toolkit.git
```

Install the requirements.

```
pip install -r requirements.txt
```

Switch to your own branch directly

```
git checkout -b my-branch
```

## Usage

1. Define your scenarios, names and their path under `scenmap.py`, such as

```
myscen = {'vref': 'C:\\Users\\username\\EPPA7\\results\\all_v-ref_p0_r0_gdpg-m_aeeg-m_sekl-m.gdx',
          'ParisForever': 'C:\\Users\\username\\EPPA7\\results\\all_ParisForever.gdx'
}
```

Save the file.

2. Browse `functions.py` to see the available functions and what can be plotted.

3. Open `notebook.py` or convert it into a real notebook

Either directly by opening Jupyter Lab if jupytext is enabled or by doing the following

```
pip install jupytext
jupytext notebook.py --to notebook
```

This will create a file named notebook.ipynb

4. Plot your graphs

Import the ackages first, then refresh them before running any function if you have changed them in the meantime.
Convert you GDX files into Dataframes using `gdx2dfs`.
Plot your graphs

5. Contributing

Please work in your own branch and open a pull request to merge changes into `main` or directly push your own branch.

## Project structure

├── scenmap.py # Lists all GDX scenarios - ignored
├── library.py # Metadata for GDX parameters
├── functions.py # GDX-to-DataFrame and plotting functions
├── notebooks.py/ # Analysis notebooks
├── requirements.txt # dependency list
└── README.md # You're here
