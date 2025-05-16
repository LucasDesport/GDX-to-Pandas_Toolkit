''' library '''

import pandas as pd

lib = pd.DataFrame({
    'agy': {
        'Yaxis': 'Production',
        'EPPA_units': '10B USD',
        'Converter': 1.0,
        'Unit': 'billion USD',
        'type': 'monetary'
    },
    'sco2': {
        'Yaxis': 'Emissions',
        'EPPA_units': 'MtCO2',
        'Converter': 1.0,
        'Unit': 'MtCO2',
        'type': 'emission'
    },
    'DPD': {
        'Yaxis': 'Domestic price',
        'EPPA_units': '(relative value)',
        'Converter': 1.0,
        'Unit': '(relative value)',
        'type': 'price'
    },
    'APA': {
        'Yaxis': 'Armington price',
        'EPPA_units': '(relative value)',
        'Converter': 1.0,
        'Unit': '(relative value)',
        'type': 'price'
    },
    'imflow': {
        'Yaxis': 'Imports',
        'EPPA_units': 'B USD',
        'Converter': 1.0,
        'Unit': 'billion USD',
        'type': 'monetary'
    },
    'exflow': {
        'Yaxis': 'Exports',
        'EPPA_units': 'B USD',
        'Converter': 1.0,
        'Unit': 'billion USD',
        'type': 'monetary'
    }
}).T

sectors = {'NMM': 'cement',
           'I_S': 'steel',
           'CROP': 'crops',
           'LIVE': 'livestock',
           'FORS': 'forestry',
           'FOOD': 'food',
           'COAL': 'coal',
           'ROIL': 'refined oil',
           'OIL': 'oil',
           'GAS': 'gas',
           'ELEC': 'eletricitry',
           'EINT': 'energy-intensive industries',
           'OTHR': 'other industries',
           'SERV': 'services',
           'TRAN': 'transport',
           'DWE': 'households'
          }