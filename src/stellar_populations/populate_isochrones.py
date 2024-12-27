import sys
import tracemalloc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from tools import file_exists, mkdir, make_vor_density, plot_vor_density2, myjet
from ezpadova import parsec, get_one_isochrone
from IPython.display import clear_output


def interpolate_magnitudes(iso,df,err):

    binary_indices = df['secondary_mass']>0
    
    ind = iso['Gmag']<20
    for o,i in enumerate(['G_BPmag','G_RPmag','Gmag']):
        df['primary_'+i] = np.interp(df['primary_mass'], iso['Mini'][ind], iso[i][ind], left=np.nan, right=np.nan)
        df['secondary_'+i] = np.interp(df['secondary_mass'], iso['Mini'][ind], iso[i][ind], left=np.nan, right=np.nan)
        
        df[i] = df['primary_'+i]
        
        df[i][binary_indices] = -2.5 * np.log10(10**(-0.4 * df['primary_'+i][binary_indices]) + 10**(-0.4 * df['secondary_'+i][binary_indices]))

        rng = np.random.default_rng(42+o)
        df[i] = df[i] + rng.normal(0, err, len(df))
        
    ind = (~np.isnan(df['Gmag'])) & (~np.isnan(df['G_RPmag'])) & (~np.isnan(df['G_BPmag'])) & \
                (df['Gmag']<5) & (df['G_BPmag']-df['G_RPmag']>-0.5) & (df['G_BPmag']-df['G_RPmag']<2.5)
    
    print('final number of stars=',sum(ind))
    return df['Gmag'][ind].values, df['G_BPmag'][ind].values-df['G_RPmag'][ind].values, df['secondary_mass'][ind].values/df['primary_mass'][ind].values

def check_isochrones(fn):
    if file_exists(fn):
        with pd.HDFStore(fn, mode='r') as store:
            keys = store.keys()
       
        if '/grid' in keys:
            grid = pd.read_hdf(fn,key='grid')

        print('Number of isochrones in the file:',len(keys)-1,'Number of isochones required',len(grid))
        
        if len(keys)-1<len(grid):
            print('Isochrones file is incomplete')
            return False
    else:
        print(fn,'does not exists')
        return False

def get_binary_ssp(model):
    
    fn2 = model.parameters['General']['path']+'/dat/ssps/sampled_'+model.parameters['SSP']['imf_type']+\
                                    model.parameters['SSP']['ssp_mass']+\
                                    '.bf.'+model.parameters['SSP']['binary_frac']+\
                                    '.qmin.'+model.parameters['SSP']['min_binary_mass_ratio']+'.h5'

    if file_exists(fn2):
        print('File with binary fraction of ',model.parameters['SSP']['binary_frac'],'and minimum mass ratio',model.parameters['SSP']['min_binary_mass_ratio'],' exists')        
    else:
        fn1 = model.parameters['General']['path']+'/dat/ssps/sampled_'+model.parameters['SSP']['imf_type']+model.parameters['SSP']['ssp_mass']+'.h5'
        print(fn1)
        if file_exists(fn1):
            df = pd.read_hdf(fn1)
                   
            num_binaries = int(len(df) * float(model.parameters['SSP']['binary_frac']))
            
            binary_indices = np.random.choice(len(df), num_binaries, replace=False)
                
            tmp = np.zeros(len(df))
            tmp[binary_indices] = 1
            
            df['secondary_mass'] = df['primary_mass']*np.random.uniform(float(model.parameters['SSP']['min_binary_mass_ratio']), 1.0, len(df))*tmp
            
            print('number of binaries=',len(binary_indices),'primary_mass=',sum(df['primary_mass']),'secondary_mass=',np.nansum(df['secondary_mass']))

            df.to_hdf(fn2,key='ssp2')
        else:
            print('Need to generate SSP file')
            sys.exit(1)
            
    return pd.read_hdf(fn2)
        
            
def populate_isochrones(model):
    tracemalloc.start()
    
    file_mask = \
    model.parameters['AMR_grid']['age_scale']+'.a'+ \
    model.parameters['AMR_grid']['age_min']+'.a'+ \
    model.parameters['AMR_grid']['age_max']+'.n'+ \
    model.parameters['AMR_grid']['n_age']+'.met.'+ \
    model.parameters['AMR_grid']['met_scale']+'.m'+ \
    model.parameters['AMR_grid']['met_min']+'.m'+ \
    model.parameters['AMR_grid']['met_max']+'.n'+ \
    model.parameters['AMR_grid']['n_met']+'.h5'

    fn = model.parameters['General']['path']+'/dat/isochrones_download/iso.age.'+ file_mask
    fno = model.parameters['General']['path']+'/dat/isochrones_sampled/iso_vor.age.'+ file_mask

    mkdir(model.parameters['General']['path']+'/dat/isochrones_sampled')
    
    file_is_ready = check_isochrones(fn)

    if file_is_ready:
        print('Going to bin isochrones to voronoi grid')
        grid = pd.read_hdf(fn,key='grid')    
    else:
        print('Please re-run initialize_isochrones()')
        grid = pd.read_hdf(fn,key='grid')    
        # self.initialize_isochrones(model)

    if len(model.pts_df)>0:
        print('Gaia CMD grid ready and contains',len(model.pts_df),' vornoi cells')
    else:
        # model.create_cmd_grid(model)
        print('No Gaia CMD and grid are available')
        sys.exit(1)
    
    df_ssp2 = get_binary_ssp(model)

    mn = 'mn'+\
            '.SN' + model.parameters['CMD_grid']['sn']+\
            '.SCALE.' + model.parameters['CMD_grid']['scale']+\
            '.IMF.' + model.parameters['SSP']['imf_type']+\
            '.mas.'+model.parameters['SSP']['ssp_mass']+\
            '.bf.'+model.parameters['SSP']['binary_frac']+\
            '.qmin.'+model.parameters['SSP']['min_binary_mass_ratio']+\
            '.phot_err' + model.parameters['SSP']['phot_err'] +'.'+str(len(grid))

    mkdir(model.parameters['General']['path']+'/figs/'+mn)

    if file_exists(fno):
         with pd.HDFStore(fno, mode='r') as store:
            keys = store.keys()
    else:
        keys = []

    print(keys)
    
    grid.to_hdf(fno,key='grid',mode='a')
    model.pts_df.to_hdf(fno,key='grid_cmd',mode='a')
    
    dftmp = pd.DataFrame()
    pts = model.pts_df[['pts_x', 'pts_y']].to_numpy()
    
    progress_bar = tqdm(total=len(grid), desc="Processing")

    o=-1
    for lab,age,met in zip(grid['labels'],grid['ages'],grid['mets']):
        o=o+1
        progress_bar.update(1)
        print(lab)
        if '/'+lab in keys:
            print('already exists')
            continue
        
            
        df = pd.read_hdf(fn,key=lab,mode='a')
        gmag, bprp, mass_ratio = interpolate_magnitudes(df,df_ssp2,float(model.parameters['SSP']['phot_err']))

        zz = make_vor_density(pts,bprp,gmag)

        dftmp['zz'] = zz

        dftmp.to_hdf(fno,key=lab)
        
        if o%int(model.parameters['AMR_grid']['save_isochrone_figs'])==0:
            
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            tit = 'Age = '+str(int(age*100)/100)+'  Gyr   Z = '+str(int(met*100)/100)
            
            plot_vor_density2(axes[0],pts,zz,[0.001*zz.max(),zz.max()],tit, scale='log')
    
            ind = df['Gmag']<5
            axes[0].plot(df['G_BPmag'][ind]-df['G_RPmag'][ind], df['Gmag'][ind], c='k', linewidth=2)        
    
            tit = ' binary fraction = '+model.parameters['SSP']['binary_frac']
            plot_vor_density2(axes[1],pts,zz*np.nan,[0,1],tit, scale='lin')
            tpc=axes[1].scatter(bprp, gmag, c=mass_ratio, linewidth=1, alpha=1, marker='.', s=2,cmap=myjet())            
            axes[1].set_xlim(-0.5,2.5)
            axes[1].set_ylim(-5,5)
            plt.gca().invert_yaxis()         
            axes[1].set_xlabel("BP-RP [mag]", fontsize=14)
            axes[1].set_ylabel("Gmag [mag]", fontsize=14)
            # cbar = plt.colorbar(tpc, ax=axes[1])
    
            plt.savefig(model.parameters['General']['path']+'/figs/'+mn+'/isochrone.'+lab+'.jpg') 
            plt.close()

