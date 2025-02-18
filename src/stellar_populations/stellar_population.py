from .initialize_ssp import initialize_ssp
from .initialize_isochrones import initialize_isochrones
from .populate_isochrones import populate_isochrones
from .create_cmd_grid import create_cmd_grid
from .fit_cmd import fit_cmd

import configparser

import time
from tqdm import tqdm
import numpy as np
import pandas as pd

class StellarPopulationModel:
    """A class template for Stellar Population Modeling."""

    def __init__(self):
        """Initialize the StellarPopulationModel class."""
        self.parameters = {}
        self.config_file_name = ''
        
        # self.pts_df = pd.DataFrame()
        # self.df_gaia = pd.DataFrame()
        
    def read_config(self, config_file):
        """
        Read input parameters from a configuration file.
        :param config_file: Path to the configuration file.
        """
        config = configparser.ConfigParser()
        config.read(config_file)

        # Convert config sections and options into a dictionary
        for section in config.sections():
            self.parameters[section] = {}
            for key, value in config.items(section):
                self.parameters[section][key] = value

        self.config_file_name = config_file

        file_mask = \
            self.parameters['AMR_grid']['age_scale']+'.a'+ \
            self.parameters['AMR_grid']['age_min']+'.a'+ \
            self.parameters['AMR_grid']['age_max']+'.n'+ \
            self.parameters['AMR_grid']['n_age']+'.met.'+ \
            self.parameters['AMR_grid']['met_scale']+'.m'+ \
            self.parameters['AMR_grid']['met_min']+'.m'+ \
            self.parameters['AMR_grid']['met_max']+'.n'+ \
            self.parameters['AMR_grid']['n_met']
        
        tmp = \
            '.bf'+self.parameters['SSP']['binary_frac']+ \
            '.phot_err'+self.parameters['SSP']['phot_err'] + \
            '.SN' + self.parameters['CMD_grid']['sn'] + \
            '.SCALE.' + self.parameters['CMD_grid']['scale'] + \
            '.'+self.parameters['SSP']['imf_type'] +\
            '.'+self.parameters['SSP']['ssp_mass'] +\
            '.xs.'+self.parameters['SSP']['xs'] +\
            '.ys.'+self.parameters['SSP']['ys']

        path = self.parameters['General']['path']
        self.isochrones_download_file_name = path+'/dat/isochrones_download/iso.age.'+ file_mask +'.'+self.parameters['AMR_grid']['model']+'.h5'
        self.isochrones_sampled_file_name = path+'/dat/isochrones_sampled/iso_vor.age.'+ file_mask + tmp +'.'+self.parameters['AMR_grid']['model']+'.h5'
        self.isochrones_sampled_figs_folder = path+'/figs/isochrones.'+ file_mask + tmp + '.'+self.parameters['AMR_grid']['model']+'/'        
    
    
    # # Define methods as placeholders to call external implementations
    def initialize_ssp(self):
        initialize_ssp(self)

    def initialize_isochrones(self):
        initialize_isochrones(self)

    def populate_isochrones(self):
        populate_isochrones(self)

    def create_cmd_grid(self):
        create_cmd_grid(self)

    def fit_cmd(self):
        fit_cmd(self)
