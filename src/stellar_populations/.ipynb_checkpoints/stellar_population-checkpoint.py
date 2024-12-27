from .initialize_ssp import initialize_ssp
from .initialize_isochrones import initialize_isochrones
from .populate_isochrones import populate_isochrones
from .create_cmd_grid import create_cmd_grid

# from .fit_cmd import fit_cmd
import configparser

import time
from tqdm import tqdm
import numpy as np


class StellarPopulationModel:
    """A class template for Stellar Population Modeling."""

    def __init__(self):
        """Initialize the StellarPopulationModel class."""
        self.parameters = {}

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

    # # Define methods as placeholders to call external implementations
    def initialize_ssp(self):
        initialize_ssp(self)

    def initialize_isochrones(self):
        initialize_isochrones(self)

    def populate_isochrones(self):
        populate_isochrones(self)

    def create_cmd_grid(self):
        create_cmd_grid(self)

    # def fit_cmd(self):
    #     fit_cmd(self)
