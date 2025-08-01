''' library '''

import pandas as pd

lib = pd.DataFrame({
    'agy': {
        'Yaxis': 'Production',
        'EPPA_units': '10B$',
        'Converter': 1.0,
        'Unit': 'B$',
        'type': 'quantity'
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
        'EPPA_units': '',
        'Converter': 1.0,
        'Unit': '',
        'type': 'price'
    },
    'imflow': {
        'Yaxis': 'Imports',
        'EPPA_units': 'B$',
        'Converter': 1.0,
        'Unit': 'B$',
        'type': 'monetary'
    },
    'impo_t': {
        'Yaxis': 'Imports',
        'EPPA_units': 'B$',
        'Converter': 1.0,
        'Unit': 'B$',
        'type': 'monetary'
    },
    'exflow': {
        'Yaxis': 'Exports',
        'EPPA_units': 'B$',
        'Converter': 1.0,
        'Unit': 'B$',
        'type': 'monetary'
    },
    'expo_t': {
        'Yaxis': 'Exports',
        'EPPA_units': 'B$',
        'Converter': 1.0,
        'Unit': 'B$',
        'type': 'monetary'
    },
    'ACCA': {
        'Yaxis': 'Consumption',
        'EPPA_units': 'B$',
        'Converter': 1.0,
        'Unit': 'B$',
        'type': 'monetary'
    },
    'cons_is': {
        'Yaxis': 'Consumption',
        'EPPA_units': 'B$',
        'Converter': 1.0,
        'Unit': 'B$',
        'type': 'quantity'
    },
    'prod_is': {
        'Yaxis': 'Production',
        'EPPA_units': 'B$',
        'Converter': 1.0,
        'Unit': 'B$',
        'type': 'quantity'
    },
    'etotco2': {
        'Yaxis': 'Process CO2 emissions',
        'EPPA_units': 'MtCO2',
        'Converter': 1.0,
        'Unit': 'MtCO2',
        'type': 'emission'
    },
    'dk_t': {
        'Yaxis': 'Capital',
        'EPPA_units': 'BUS$',
        'Converter': 1.0,
        'Unit': 'BUS$',
        'type': 'monetary'
    },
    'dl_t': {
        'Yaxis': 'Labor',
        'EPPA_units': 'BUS$',
        'Converter': 1.0,
        'Unit': 'BUS$',
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



regions = pd.DataFrame({'IDZ': {'name': 'Indonesia',
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
           'BRA': {'name': 'Brazil',
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
          }).T

regions_dict = {
    'IDZ': {'name': 'Indonesia', 'color': '#B03AC2'},
    'KOR': {'name': 'Korea', 'color': '#52CE02'},
    'REA': {'name': 'Rest of Asia', 'color': '#CCBE2C'},
    'LAM': {'name': 'Latin America', 'color': '#38A7A2'},
    'MES': {'name': 'Middle-East', 'color': '#D6D092'},
    'AFR': {'name': 'Africa', 'color': '#1A5A2D'},
    'BRA': {'name': 'Brazil', 'color': '#16824D'},
    'IND': {'name': 'India', 'color': '#979576'},
    'CHN': {'name': 'China', 'color': '#725D7A'},
    'ASI': {'name': 'South Asia', 'color': '#493B82'},
    'RUS': {'name': 'Russia', 'color': '#2B4739'},
    'ROE': {'name': 'Rest of Europe', 'color': '#91C96E'},
    'EUR': {'name': 'Europe', 'color': '#679C82'},
    'ANZ': {'name': 'Australia and New-Zealand', 'color': '#1B344A'},
    'JPN': {'name': 'Japan', 'color': '#6E36A4'},
    'MEX': {'name': 'Mexico', 'color': '#80CDDF'},
    'CAN': {'name': 'Canada', 'color': '#1D4971'},
    'USA': {'name': 'USA', 'color': '#5492C5'}
}

conv_R = pd.DataFrame({'NMM': 
                        {'AFR': 0.0751,
                        'ANZ': 0.2041,
                        'ASI': 0.0546,
                        'BRA': 0.0600,
                        'CAN': 0.1208,
                        'CHN': 0.0470,
                        'EUR': 0.5529,
                        'IDZ': 0.0483,
                        'IND': 0.0261,
                        'JPN': 0.1139,
                        'KOR': 0.0691,
                        'LAM': 0.0781,
                        'MES': 0.0732,
                        'MEX': 0.0611,
                        'REA': 0.0433,
                        'ROE': 0.0572,
                        'RUS': 0.0642,
                        'USA': 0.1904
                        },
                       'I_S':
                       {'AFR': 0.2188,
                        'ANZ': 0.4140,
                        'ASI': 0.2914,
                        'BRA': 0.2224,
                        'CAN': 0.2067,
                        'CHN': 0.1919,
                        'EUR': 0.2976,
                        'IDZ': 0.2160,
                        'IND': 0.1399,
                        'JPN': 0.3833,
                        'KOR': 0.2356,
                        'LAM': 0.4004,
                        'MES': 0.1538,
                        'MEX': 0.1928,
                        'REA': 0.0639,
                        'ROE': 0.1258,
                        'RUS': 0.1087,
                        'USA': 0.2932
                        }
                       }).T

gwp_100y = {
    'CH4': 28,
    'N2O': 265,
    'PFC': 6630,
    'SF6': 23500,
    'HFC': 1300
}

data_elec_map = {
    '22a_coal_no CCS (TWh)': {
        'label': 'Coal',
        'color': '#3B3B3B',
        'hatch': None
    },
    '22b_coal_CCS (TWh)': {
        'label': 'Coal with CCS',
        'color': '#3B3B3B',
        'hatch': '...'
    },
    '23_oil (TWh)': {
        'label': 'Oil',
        'color': '#7F4F24',
        'hatch': None
    },
    '24a_gas_no CCS (TWh)': {
        'label': 'Gas',
        'color': '#C44536',
        'hatch': None
    },
    '24b_gas_CCS (TWh)': {
        'label': 'Gas with CCS',
        'color': '#C44536',
        'hatch': '...'
    },
    '25_nuclear (TWh)': {
        'label': 'Nuclear',
        'color': '#F2C849',
        'hatch': None
    },
    '27a_bioelectricity and other (TWh)': {
        'label': 'Bioenergy',
        'color': '#287D57',
        'hatch': None
    },
    '27a_bioelectricity_CCS (TWh)': {
        'label': 'BECCS',
        'color': '#287D57',
        'hatch': '...'
    },
    '26_hydro (TWh)': {
        'label': 'Hydro',
        'color': '#2E86AB',
        'hatch': None
    },
    '27b_renewables_wind (TWh)': {
        'label': 'Wind',
        'color': '#66E8E5',
        'hatch': None
    },
    '27c_renewables_solar (TWh)': {
        'label': 'Solar',
        'color': '#F2EA74',
        'hatch': None
    }
}

data_nrj_map = {
    '15_coal (EJ)':             {'label': 'Coal',       'color': '#3B3B3B'},
    '16_oil (EJ)':              {'label': 'Oil',        'color': '#7F4F24'},
    '17_gas (EJ)':              {'label': 'Gas',        'color': '#C44536'},
    '18b_bioenergy (EJ)':       {'label': 'Bioenergy',  'color': '#287D57'},
    '19_nuclear (EJ)':          {'label': 'Nuclear',    'color': '#F2C849'},
    '19b_hydro (EJ)':           {'label': 'Hydro',      'color': '#2E86AB'},
    '20_renewables (wind&solar) (EJ)': {'label': 'Renewables', 'color': '#91C499'}
}

data_emis_map = {
    '06a_Pos_CO2_fossil (million ton)':       {'label': 'Fossil CO₂',       'color': '#d73027'},
    '07_CO2_industrial (million ton)':        {'label': 'Process CO₂',      'color': '#f46d43'},
    '08_CO2_land use change (million ton)':   {'label': 'AFOLU CO₂',        'color': '#fdae61'},
    '08b_NE_bioccs':                          {'label': 'CDR - BECCS',      'color': '#4575b4'},
    '08c_NE_daccs':                           {'label': 'CDR - DACCS',      'color': '#74add1'},
    '14a_GHGinCO2eq (million ton)':           {'label': 'Non-CO₂ emissions','color': '#66c2a5'}
}
