
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools import file_exists, mkdir, make_voronoi, myjet, make_vor_density, plot_vor_density2, plot_solution, solver, copy_file

from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def read_isochrones(model):
    # file_mask = \
    # model.parameters['AMR_grid']['age_scale']+'.a'+ \
    # model.parameters['AMR_grid']['age_min']+'.a'+ \
    # model.parameters['AMR_grid']['age_max']+'.n'+ \
    # model.parameters['AMR_grid']['n_age']+'.met.'+ \
    # model.parameters['AMR_grid']['met_scale']+'.m'+ \
    # model.parameters['AMR_grid']['met_min']+'.m'+ \
    # model.parameters['AMR_grid']['met_max']+'.n'+ \
    # model.parameters['AMR_grid']['n_met'] +'.bf'+ \
    # model.parameters['SSP']['binary_frac'] +'.phot_err'+ \
    # model.parameters['SSP']['phot_err']+ '.SN' + \
    # model.parameters['CMD_grid']['sn']+'.SCALE.' + \
    # model.parameters['CMD_grid']['scale']+'.h5'

    # fn = model.parameters['General']['path']+'/dat/isochrones_sampled/iso_vor.age.'+ file_mask
    
    fn = model.isochrones_sampled_file_name
    
    with pd.HDFStore(fn, mode='r') as store:
        keys = store.keys()

    amr_grid = pd.read_hdf(fn,key='grid');
    cmd_grid = pd.read_hdf(fn,key='grid_cmd');

    print(len(amr_grid))
    print(len(keys))
    
    if len(amr_grid) == len(keys)-2:
        print('The file contains all isochrones')
        isochrones = np.zeros((len(amr_grid),len(cmd_grid)))
        
        for o,lab in enumerate(amr_grid['labels']):
            tmp = pd.read_hdf(fn,key=lab)
            isochrones[o,:] = tmp['zz'].values

        print('Shape of isochrones',isochrones.shape)

        return isochrones, amr_grid['ages'].values, amr_grid['mets'].values

def read_cmd_to_fit(model):

    if model.parameters['Fitting']['cmd_to_fit']=='default':
        fn = model.parameters['General']['path']+'/dat/cmd_grid/gaia3_cmd_full_all_voron.SN.'+\
          model.parameters['CMD_grid']['sn']+'.SCALE.'+model.parameters['CMD_grid']['scale']+'.h5'
    else:
        fn = model.parameters['Fitting']['cmd_to_fit']

    if file_exists(fn):
        print('CMD to fit:',fn)

        tmp = pd.read_hdf(fn,key='dat')

        return tmp['pts_x'].values,tmp['pts_y'].values,tmp['dat'].values/tmp['dat'].values.sum()*len(tmp['dat'])
    else:
        print('ERROR, no CMD to fit')


def save_solution(iter, parameters,fnout,pts_x,pts_y,gaia_CMD_to_fit,w0,weights,isochrones2,age,met):

    sol = pd.DataFrame(parameters)
    sol.to_hdf(fnout,key='config')
    
    sol = pd.DataFrame()
    sol['GaiaCMD'] = gaia_CMD_to_fit
    sol['pts_x'] = pts_x
    sol['pts_y'] = pts_y
    sol.to_hdf(fnout,key='gaia')
    
    sol = pd.DataFrame()
    sol['w0'] = w0
    sol['age'] = age
    sol['met'] = met
    sol.to_hdf(fnout,key='grid')
    
    sol = pd.DataFrame()
    sol['w'] = weights
    sol.to_hdf(fnout,key='sol'+str(iter))
    
    sol = pd.DataFrame(isochrones2)
    sol.to_hdf(fnout,key='isochrones')

    

def fit_cmd(model):
    now = datetime.now()    
    date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

    fn0 = model.parameters['General']['path']+'/figs/'+model.parameters['Fitting']['model_name']+'.'+date_time_str
    fn1 = model.parameters['General']['path']+'/results/'+model.parameters['Fitting']['model_name']+'.'+date_time_str

    
    mkdir(model.parameters['General']['path']+'/figs/')
    mkdir(fn0)
    mkdir(fn1)
    copy_file(model.config_file_name,fn1)

    figname_out0 = fn0+'/solution'+date_time_str 
    sol_file_name = fn1+'/solution'+date_time_str+'.h5'
    
    isochrones, AGE, MET = read_isochrones(model)

    all_isochrones = np.sum(isochrones, axis=0)

    pts_x,pts_y,gaia_cmd = read_cmd_to_fit(model)

    if model.parameters['Fitting']['initial_guess']=='none':     
        print('Uniform initial guess')
        w0 = AGE/AGE
        # w0 = 1+AGE
        e_w0 = np.exp(w0)
        test_CMD = isochrones.T @ e_w0
        e_w0 = e_w0 / test_CMD.sum() * len(test_CMD)
        test_CMD = isochrones.T @ e_w0
        w0 = np.log(e_w0)
    else:
        with pd.HDFStore(model.parameters['Fitting']['initial_guess'], mode='r') as store:
            keys = store.keys()        

        print('Will use an initial guess from ',model.parameters['Fitting']['initial_guess'],'iteration',keys[-1])
        a = pd.read_hdf(model.parameters['Fitting']['initial_guess'],key=keys[-1])
        w0 = a['w'].values
        test_CMD = isochrones.T @ np.exp(w0)
    
    figname_out = figname_out0+'.initial.jpg' 

    plot_solution(pts_x, pts_y, gaia_cmd, test_CMD,AGE,MET,w0,w0,[0],figname_out)

    ind = (all_isochrones==0) | (gaia_cmd/gaia_cmd.max()< 1e-3)
    gaia_cmd[ind] = 0
    isochrones2 = isochrones.copy()
    isochrones2[:,ind] = 0

    nsave = int(model.parameters['Fitting']['nsave'])
    hist = []
    

    eps = float(model.parameters['Fitting']['eps'])
    
    for i in range(0,int(model.parameters['Fitting']['max_step'])):
        print(model.parameters['Fitting']['model_name']+'.'+date_time_str,'running iteration ',i,' out of ',model.parameters['Fitting']['max_step'])
        figname_out = figname_out0 +'.'+ str(i).zfill(7)+'.jpg'
        current_weights,hist0 = solver(pts_x, pts_y, isochrones2.T, w0, gaia_cmd, eps, nsave,fittype=model.parameters['Fitting']['fittype'])
        hist.extend(hist0)
        plot_solution(pts_x, pts_y, gaia_cmd, isochrones2.T @ np.exp(current_weights),AGE,MET,np.exp(w0),np.exp(current_weights),hist,figname_out)
        save_solution(i,model.parameters,sol_file_name,pts_x,pts_y,gaia_cmd,w0,current_weights,isochrones2,AGE,MET)
        w0 = current_weights
        
    

    

    