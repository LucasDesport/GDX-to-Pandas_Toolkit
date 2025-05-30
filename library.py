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

sectors = pd.DataFrame({
    'NMM': {'name': 'cement', 'color': '#0C5DA5'},
    'I_S': {'name': 'steel', 'color': '#FF9500'},
    'CROP': {'name': 'crops', 'color': '#B03AC2'},
    'LIVE': {'name': 'livestock', 'color': '#52CE02'},
    'FORS': {'name': 'forestry', 'color': '#FF2400'},
    'FOOD': {'name': 'food', 'color': '#38A7A2'},
    'COAL': {'name': 'coal', 'color': '#D2691E'},
    'ROIL': {'name': 'refined oil', 'color': '#6495ED'},
    'OIL': {'name': 'oil', 'color': '#FFD700'},
    'GAS': {'name': 'gas', 'color': '#40E0D0'},
    'ELEC': {'name': 'electricity', 'color': '#FF69B4'},
    'EINT': {'name': 'energy-intensive industries', 'color': '#808080'},
    'OTHR': {'name': 'other industries', 'color': '#A52A2A'},
    'SERV': {'name': 'services', 'color': '#20B2AA'},
    'TRAN': {'name': 'transport', 'color': '#9370DB'},
    'DWE': {'name': 'households', 'color': '#FF6347'},
}).T



regions = {'IDZ': {'name': 'Indonesia',
                   'color': '#B03AC2'
                  },
           'KOR': {'name': 'Korea',
                   'color': '#52CE02'
                  },
           'REA': {'name': 'Rest of Asia',
                   'color': '#CCBE2C'
                  },
           'LAM': {'name': 'Latin America',
                   'color': '#38A7A2'
                  },
           'MES': {'name': 'Middle-East',
                   'color': '#D6D092'
                  },
           'AFR': {'name': 'Africa',
                   'color': '#16824D'
                  },
           'BRA': {'name': 'Brazilia',
                   'color': '#16824D'
                  },
           'IND': {'name': 'India',
                   'color': '#979576'
                  },
           'CHN': {'name': 'China',
                   'color': '#725D7A'
                  },
           'ASI': {'name': 'South Asia',
                   'color': '#493B82'
                  },
           'RUS': {'name': 'Russia',
                   'color': '#2B4739'
                  },
           'ROE': {'name': 'Rest of Europe',
                   'color': '#679C82'
                  },
           'EUR': {'name': 'Europe',
                   'color': '#679C82'
                  },
           'ANZ': {'name': 'Australia and New-Zealand',
                   'color': '#1B344A'
                  },
           'JPN': {'name': 'Japan',
                   'color': '#6E36A4'
                  },
           'MEX': {'name': 'Mexico',
                   'color': '#80CDDF'
                  },
           'CAN': {'name': 'Canada',
                   'color': '#1D4971'
                  },
           'USA': {'name': 'USA',
                   'color': '#5492C5'
                  }
          }

