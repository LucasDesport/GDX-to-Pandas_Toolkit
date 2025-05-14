''' library '''

import pandas as pd

lib = pd.DataFrame({
    'agy': {
        'Yaxis': 'Production',
        'EPPA_units': '10B USD',
        'Converter': 1.0,
        'Unit': 'billion USD'
    },
    'sco2': {
        'Yaxis': 'Emissions',
        'EPPA_units': 'MtCO2',
        'Converter': 1.0,
        'Unit': 'MtCO2'
    },
    'DPD': {
        'Yaxis': 'Domestic price',
        'EPPA_units': '(relative value)',
        'Converter': 1.0,
        'Unit': '(relative value)'
    },
    'APA': {
        'Yaxis': 'Armington price',
        'EPPA_units': '(relative value)',
        'Converter': 1.0,
        'Unit': '(relative value)'
    },
    'imflow': {
        'Yaxis': 'Imports',
        'EPPA_units': 'B USD',
        'Converter': 1.0,
        'Unit': 'billion USD'
    },
    'exflow': {
        'Yaxis': 'Exports',
        'EPPA_units': 'B USD',
        'Converter': 1.0,
        'Unit': 'billion USD'
    }
}).T