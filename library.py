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
           'ELEC': 'eletricity',
           'EINT': 'energy-intensive industries',
           'OTHR': 'other industries',
           'SERV': 'services',
           'TRAN': 'transport',
           'DWE': 'households'
          }

regions = {'IDZ': 'Indonesia',
           'KOR': 'Korea',
           'REA': 'Rest of Asia',
           'LAM': 'Latin America',
           'MES': 'Middle-East',
           'AFR': 'Africa',
           'BRA': 'Brazilia',
           'IND': 'India',
           'CHN': 'China',
           'ASI': 'South Asia',
           'RUS': 'Russia',
           'ROE': 'Rest of Europe',
           'EUR': 'Europe',
           'ANZ': 'Australia and New-Zealand',
           'JPN': 'Japan',
           'MEX': 'Mexico',
           'CAN': 'Canada',
           'USA': 'USA'
          }

