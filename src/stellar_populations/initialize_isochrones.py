import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from tools import file_exists, mkdir
from ezpadova import parsec, get_one_isochrone
from IPython.display import clear_output
from isochrones.mist import MIST_Isochrone
import ezbasti


def interpolate_isochrone(x,y):
    fn = '../dat/isochrones_download/All.Basti.h5'
    
    grid = pd.read_hdf(fn,key='grid')
    
    X = grid['ages'].unique()
    Y = grid['mets'].unique()

    i = np.searchsorted(X, x) - 1
    j = np.searchsorted(Y, y) - 1
    
    i = max(0, min(i, len(X) - 2))
    j = max(0, min(j, len(Y) - 2))
    
    labs = []
    for o in range(0,4):
        labs.append(grid['labels'].iloc[np.where(grid['ij']==(i,j))[0][0]])
        
    w11 =  (X[i+1]-x)/(X[i+1]-X[i])* (Y[j+1]-y)/(Y[j+1]-Y[j])
    w12 =  (x-X[i])/(X[i+1]-X[i])* (Y[j+1]-y)/(Y[j+1]-Y[j])
    w21 =  (X[i+1]-x)/(X[i+1]-X[i])* (y-Y[j])/(Y[j+1]-Y[j])
    w22 =  (x-X[i])/(X[i+1]-X[i])* (y-Y[j])/(Y[j+1]-Y[j])

    x1 = -1e6
    x2 = 1e6
    for o in range(0,4):
        df = pd.read_hdf(fn,key=labs[o])
        x1 = np.max([x1,df['Mini'].min()])
        x2 = np.min([x2,df['Mini'].max()])
    
    W = [w11,w12,w21,w22]

    dd = pd.DataFrame()
    n = 2100
    dd['Mini'] = np.linspace(x1,x2,n)

    for o in range(0,4):
        df = pd.read_hdf(fn,key=labs[o])
        for i in df.columns:
            if i == 'Mini':
                continue
            if o==0:
                dd[i] = np.zeros(n)
            dd[i] = dd[i] + W[o] * np.interp(dd['Mini'],df['Mini'],df[i]) 

    return dd



def initialize_isochrones(model):
    print('\n ###############################################################')
    print('    Checking '+ model.parameters['AMR_grid']['model'] +' isochrones file')
    print(' ###############################################################\n')

    fn = model.parameters['General']['path']+'/dat/isochrones_download/iso.age.'+ \
    model.parameters['AMR_grid']['age_scale']+'.a'+ \
    model.parameters['AMR_grid']['age_min']+'.a'+ \
    model.parameters['AMR_grid']['age_max']+'.n'+ \
    model.parameters['AMR_grid']['n_age']+'.met.'+ \
    model.parameters['AMR_grid']['met_scale']+'.m'+ \
    model.parameters['AMR_grid']['met_min']+'.m'+ \
    model.parameters['AMR_grid']['met_max']+'.n'+ \
    model.parameters['AMR_grid']['n_met']+'.'+model.parameters['AMR_grid']['model']+'.h5'

    if model.parameters['AMR_grid']['model'] == 'MIST':    
        mist = MIST_Isochrone()
    
    print(fn)

    if file_exists(fn):
        print('File... '+fn+' already exists')
        with pd.HDFStore(fn, mode='r') as store:
            keys = store.keys()
       
        if '/grid' in keys:
            grid = pd.read_hdf(fn,key='grid')
            file_is_ok = True
        else:
            print('grid does not exists. Exit now!')
            file_is_ok = False 
            sys.exit(1)
            os.delete(fn)            
    else:
        file_is_ok = False

    if file_is_ok == False:
        print('Existing file not found. A new file will be generated')
        mkdir(model.parameters['General']['path']+'/dat/isochrones_download')

        x1 = float(model.parameters['AMR_grid']['age_min'])
        x2 = float(model.parameters['AMR_grid']['age_max'])    

        if model.parameters['AMR_grid']['model'] == 'Basti':
            print('Changing the lower age limit for Basti isochrones',x1*1e9,x2*1e9)
            if x1*1e9<1.5e7:
                x1 = 1.5e7/1e9
            if x2>13:
                x2 = 12.99
                
        if model.parameters['AMR_grid']['age_scale'] == 'log':                
            ages=np.logspace( np.log10(x1* 1e9),np.log10(x2*1e9),int(model.parameters['AMR_grid']['n_age']))
    
        if model.parameters['AMR_grid']['age_scale'] == 'lin':    
            ages=np.linspace(x1,x2,int(model.parameters['AMR_grid']['n_age']))* 1e9
    
        x1 = float(model.parameters['AMR_grid']['met_min'])
        x2 = float(model.parameters['AMR_grid']['met_max'])

        if model.parameters['AMR_grid']['model'] == 'Basti':
            print('Changing the lower metallicity limit for Basti isochrones',x1,x2)
            if x1<-2:
                x1 = -2
            if x2>0.45:
                x2 = 0.45

        
        if model.parameters['AMR_grid']['met_scale'] == 'log':    
            mets=np.logspace(x1,x2,int(model.parameters['AMR_grid']['n_met']))
    
        if model.parameters['AMR_grid']['met_scale'] == 'lin':    
            mets=np.linspace(x1,x2,int(model.parameters['AMR_grid']['n_met']))

        mets = 10**mets*float(model.parameters['AMR_grid']['met_sun'])
     
        mns = []
        met2rec = []
        age2rec = []
        met2rec_u = []
        age2rec_u = []
        o=-1
        for age in ages:
            clear_output()
            for met in mets:
                o=o+1
                lab = 'iso'+str(o)

                met2rec_u.append(met)
                age2rec_u.append(age)
                
                met2rec.append(np.log10(met/float(model.parameters['AMR_grid']['met_sun'])))
                age2rec.append(age/1e9)
                mns.append(lab)    
            
        grid = pd.DataFrame()
        grid['ages'] = age2rec
        grid['mets'] = met2rec
        grid['ages_u'] = age2rec_u
        grid['mets_u'] = met2rec_u
        grid['labels'] = mns
        grid.to_hdf(fn,key='grid',mode='a')

        keys = []
        
        print('Age-Met grid saved to a new file',fn)


    if len(grid) == len(keys)-1:
        print('Number of raw isochrones needed',len(grid))
        print('Number of raw isochrones in the file',len(keys)-1)
        print('All isochrones are in the file')
        
        return

    progress_bar = tqdm(total=len(grid), desc="Processing")

    for age,met,lab in (zip(grid['ages_u'],grid['mets_u'],grid['labels'])):
        progress_bar.update(1)

        # print(age/1e9,met,lab,len(grid))
        
        if '/'+lab in keys:
            print(lab+' is already here')
        else:
            if model.parameters['AMR_grid']['model'] == 'MIST':
                r = mist.isochrone(np.log10(age), np.log10(met/float(model.parameters['AMR_grid']['met_sun'])))
                r = r.rename(columns={'initial_mass': 'Mini', 'G_mag': 'Gmag', 'BP_mag': 'G_BPmag', 'RP_mag': 'G_RPmag'})

            if model.parameters['AMR_grid']['model'] == 'Padova':
                r = get_one_isochrone(age, met, photsys_file='YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat')

            if model.parameters['AMR_grid']['model'] == 'Basti':
               
                FE_H = np.log10(met/float(model.parameters['AMR_grid']['met_sun']))
                r = interpolate_isochrone(age/1e9,FE_H)
               
            r.to_hdf(fn,key=lab,mode='a')

