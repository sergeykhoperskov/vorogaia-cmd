import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from tools import file_exists, mkdir
from ezpadova import parsec, get_one_isochrone
from IPython.display import clear_output


def initialize_isochrones(model):
    fn = model.parameters['General']['path']+'/dat/isochrones_download/iso.age.'+ \
    model.parameters['AMR_grid']['age_scale']+'.a'+ \
    model.parameters['AMR_grid']['age_min']+'.a'+ \
    model.parameters['AMR_grid']['age_max']+'.n'+ \
    model.parameters['AMR_grid']['n_age']+'.met.'+ \
    model.parameters['AMR_grid']['met_scale']+'.m'+ \
    model.parameters['AMR_grid']['met_min']+'.m'+ \
    model.parameters['AMR_grid']['met_max']+'.n'+ \
    model.parameters['AMR_grid']['n_met']+'.h5'
    
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
        if model.parameters['AMR_grid']['age_scale'] == 'log':                
            ages=np.logspace( np.log10(x1* 1e9),np.log10(x2*1e9),int(model.parameters['AMR_grid']['n_age']))
    
        if model.parameters['AMR_grid']['age_scale'] == 'lin':    
            ages=np.linspace(x1,x2,int(model.parameters['AMR_grid']['n_age']))* 1e9
    
        x1 = float(model.parameters['AMR_grid']['met_min'])
        x2 = float(model.parameters['AMR_grid']['met_max'])
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

    progress_bar = tqdm(total=len(grid), desc="Processing")

   
    for age,met,lab in (zip(grid['ages_u'],grid['mets_u'],grid['labels'])):
        progress_bar.update(1)

        print(age/1e9,met,lab,len(grid))
        
        if '/'+lab in keys:
            print(lab+' is already here')
        else:
            r = get_one_isochrone(age, met, photsys_file='YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat')
            r.to_hdf(fn,key=lab,mode='a')
            