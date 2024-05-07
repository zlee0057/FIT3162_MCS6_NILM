import pandas as pd
import numpy as np
from copy import deepcopy
from os.path import join, isdir, isfile, exists
from os import listdir
import fnmatch
import os
import re
from sys import stdout
from nilmtk.utils import get_datastore
from nilmtk.datastore import Key
from nilmtk.timeframe import TimeFrame
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import get_module_directory, check_directory_exists
from nilm_metadata import convert_yaml_to_hdf5, save_yaml_to_datastore


def convert_mimos(mimos_path, output_filename, input_sec, format='HDF'):
    """
    Parameters
    ----------
    mimos_path : str
        The root path of the MIMOS 1_sec interval dataset.
    output_filename : str
        The destination filename (including path and suffix).
    format : str
        format of output. Either 'HDF' or 'CSV'. Defaults to 'HDF'
    """
    
    # Open DataStore
    store = get_datastore(output_filename, format, mode='w')

    # Convert raw data to DataStore
    _convert(mimos_path, store, 'Asia/Kuala_Lumpur')

    # Add metadata
    save_yaml_to_datastore(join(os.getcwd(), 'metadata/' + input_sec), store)
    store.close()

    print("Done converting MIMOS to HDF5 - " + input_sec + "!")

column_mapping = {
    'Active (W)': ('power', 'active'),
    'fridge': ('power', 'active'),
    'aircond': ('power', 'active'),
    'washing_machine': ('power', 'active'),
    'dryer': ('power', 'active'),
    'kettle': ('power', 'active'),
    'vacuum': ('power', 'active'),
    'water_heater': ('power', 'active'),
    'oven': ('power', 'active')
}
# column_mapping = {
#     'Active (W)': ('power', 'active'),
#     'Apparent (VA)': ('power', 'apparent'),
#     'fridge': ('power', 'active'),
#     'aircond': ('power', 'active'),
#     'washing_machine': ('power', 'active'),
#     'dryer': ('power', 'active'),
#     'kettle': ('power', 'active'),
#     'vacuum': ('power', 'active'),
#     'water_heater': ('power', 'active'),
#     'oven': ('power', 'active')
# }


def _convert(input_path, store, tz, sort_index=True):
    """
    Parameters
    ----------
    input_path : str
        The root path of the REFIT dataset.
    store : DataStore
        The NILMTK DataStore object.
    measurement_mapping_func : function
        Must take these parameters:
            - house_id
            - chan_id
        Function should return a list of tuples e.g. [('power', 'active')]
    tz : str 
        Timezone e.g. 'US/Eastern'
    sort_index : bool
    """

    check_directory_exists(input_path)

    # Iterate though all buildings
    houses = [1,2,3,4,5,6,7]
    nilmtk_house_id = 0
    
    for house_id in houses:
        nilmtk_house_id += 1
        print("Loading building", house_id, end="... ")
        stdout.flush()
        csv_filename = join(input_path, 'Building_' + str(house_id) + '.csv')
           
        if not exists(csv_filename):
            raise RuntimeError('Could not find the file. Please check the provided folder.')
#         usecols = ['UNIX', 'Apparent (VA)', 'Active (W)', 'fridge', 'aircond', 'washing_machine', 'dryer', 'kettle', 'vacuum', 'water_heater', 'oven']
        usecols = ['UNIX', 'Active (W)', 'fridge', 'aircond', 'washing_machine', 'dryer', 'kettle', 'vacuum', 'water_heater', 'oven']
        
        df = _load_csv(csv_filename, usecols, tz)
        if sort_index:
            df = df.sort_index() # might not be sorted...
        chan_id = 0
        for col in df.columns:
            chan_id += 1
            print(chan_id, end=" ")
            stdout.flush()
            key = Key(building=nilmtk_house_id, meter=chan_id)
            
            chan_df = pd.DataFrame(df[col])
            chan_df.columns = pd.MultiIndex.from_tuples([column_mapping[x] for x in chan_df.columns])
            
            # Modify the column labels to reflect the power measurements recorded.
            chan_df.columns.set_names(LEVEL_NAMES, inplace=True)
            
            store.put(str(key), chan_df)
            
        print('')


def _load_csv(filename, usecols, tz):
    """
    Parameters
    ----------
    filename : str
    usecols : list of columns to keep
    tz : str e.g. 'US/Eastern'

    Returns
    -------
    dataframe
    """
    # Load data
    df = pd.read_csv(filename, usecols=usecols)
    
    # Convert the integer index column to timezone-aware datetime 
    df['UNIX'] = pd.to_datetime(df['UNIX'], unit='s', utc=True)
    df.set_index('UNIX', inplace=True)
    df = df.tz_convert(tz)
    
    return df