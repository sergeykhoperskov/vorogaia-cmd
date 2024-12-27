import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from tools import file_exists

def initialize_ssp(model):
    """
    Generate a Simple Stellar Population (SSP).
    :param model: Instance of StellarPopulationModel.
    
    """
    print("Generating SSP...")

    fn = model.parameters['General']['path']+'/dat/ssps/sampled_'+model.parameters['SSP']['imf_type']+model.parameters['SSP']['ssp_mass']+'.h5'

    print(fn)

    if file_exists(fn):
        print('File... '+fn+' already exists')
        return
    else:
        print('Existing file not found. A new file will be generated')
    
    if model.parameters['SSP']['imf_type'] == 'kroupa':
        print("generating Kroupa IMF and total mass of 10e",model.parameters['SSP']['ssp_mass'])        
        df_ssp = pd.DataFrame()

        df_ssp['primary_mass'] = sample_kroupa_imf(10**float(model.parameters['SSP']['ssp_mass']))
        df_ssp.to_hdf(fn,key='dat')

        return

    if model.parameters['SSP']['imf_type'] == 'salpeter':
        print("generating Salpeter IMF and total mass of 10e",model.parameters['SSP']['ssp_mass'])        
        df_ssp = pd.DataFrame()

        df_ssp['primary_mass'] = sample_salpeter_imf(10**float(model.parameters['SSP']['ssp_mass']))
        df_ssp.to_hdf(fn,key='dat')

        return
    
    if model.parameters['SSP']['imf_type'] == 'chabrier':
        print("generating Chabrier IMF and total mass of 10e",model.parameters['SSP']['ssp_mass'])        
        df_ssp = pd.DataFrame()

        df_ssp['primary_mass'] = sample_chabrier_imf(10**float(model.parameters['SSP']['ssp_mass']))
        df_ssp.to_hdf(fn,key='dat')

        return
            
    print('wrong IMF type')
    sys.exit(1)
        
        
def sample_kroupa_imf(total_mass,m_min=0.01,m_max=100):
    """
    Sample stellar masses according to the Kroupa IMF.
    :param total_mass: Total stellar mass to sample (in solar masses).
    :return: Array of sampled stellar masses.
    """
    # Kroupa IMF parameters
    def kroupa_pdf(m):
        if m < 0.08:
            return m**-0.3
        elif m < 0.5:
            return m**-1.3
        else:
            return m**-2.3

    progress_bar = tqdm(total=total_mass, desc="Processing")

    # Rejection sampling
    masses = []
    current_mass = 0
    while current_mass < total_mass:
        progress_bar.update(current_mass/total_mass)
        m = np.random.uniform(m_min, m_max)
        p = np.random.uniform(0, 1)
        if p < kroupa_pdf(m):
            masses.append(m)
            current_mass += m
    return np.array(masses)

def sample_salpeter_imf(total_mass, m_min=0.1, m_max=100):
    """
    Sample stellar masses according to the Salpeter IMF.
    :param total_mass: Total stellar mass to sample (in solar masses).
    :param m_min: Minimum stellar mass.
    :param m_max: Maximum stellar mass.
    :return: Array of sampled stellar masses.
    """
    # Salpeter IMF parameters
    def salpeter_pdf(m):
        return m**-2.35

    progress_bar = tqdm(total=total_mass, desc="Processing")
    
    # Rejection sampling
    masses = []
    current_mass = 0
    while current_mass < total_mass:
        progress_bar.update(current_mass/total_mass)
        m = np.random.uniform(m_min, m_max)
        p = np.random.uniform(0, 1)
        if p < salpeter_pdf(m) / salpeter_pdf(m_min):  # Normalize PDF
            masses.append(m)
            current_mass += m
    return np.array(masses)


def sample_chabrier_imf(total_mass, m_min=0.01, m_max=100):
    """
    Sample stellar masses according to the Chabrier IMF.
    :param total_mass: Total stellar mass to sample (in solar masses).
    :param m_min: Minimum stellar mass.
    :param m_max: Maximum stellar mass.
    :return: Array of sampled stellar masses.
    """
    # Chabrier IMF parameters
    mu = np.log(0.2)  # Log-normal mean
    sigma = 0.55      # Log-normal standard deviation

    def chabrier_pdf(m):
        if m <= 1:
            return (1 / (m * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(m) - mu)**2) / (2 * sigma**2))
        else:
            return m**-2.3

    progress_bar = tqdm(total=total_mass, desc="Processing")
    # Rejection sampling
    masses = []
    current_mass = 0
    while current_mass < total_mass:
        progress_bar.update(current_mass/total_mass)
        m = np.random.uniform(m_min, m_max)
        p = np.random.uniform(0, 1)
        if p < chabrier_pdf(m) / chabrier_pdf(m_min):  # Normalize PDF
            masses.append(m)
            current_mass += m
    return np.array(masses)
