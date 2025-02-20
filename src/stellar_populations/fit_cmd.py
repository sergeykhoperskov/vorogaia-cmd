import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools import file_exists, mkdir, make_voronoi, myjet, make_vor_density, plot_vor_density2, plot_solution, solver, copy_file, yes_no_input

from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from scipy.ndimage import gaussian_filter1d

def read_isochrones(model):

    fn = model.isochrones_sampled_file_name
    
    print(fn)
    
    with pd.HDFStore(fn, mode='r') as store:
        keys = store.keys()

    amr_grid = pd.read_hdf(fn,key='grid');
    cmd_grid = pd.read_hdf(fn,key='grid_cmd');
    
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

        print('Size of the CMD:',len(tmp))
        
        return tmp['pts_x'].values,tmp['pts_y'].values,tmp['dat'].values/tmp['dat'].values.sum()*len(tmp['dat'])
    else:
        print('ERROR, no CMD to fit')

def read_solution(fn, parameters):

    iteration = pd.read_hdf(fn,key='iteration').values[0]
    
    if iteration == int(parameters['Fitting']['max_step']):
        print('The desired solution already exists. Exit')
        sys.exit(1)
    else:
        print('The file containes iteration=',iteration)
        print('Will continue from the current solution')
        
        if parameters['Fitting']['save_each_step']=='yes':
            dd = pd.read_hdf(fn,key='sol'+str(iteration))
        else:
            dd = pd.read_hdf(fn,key='sol')

        w0 = dd['w'].values

        dd = pd.read_hdf(fn,key='history')
        
    return iteration, w0, dd['history'].values.tolist()

def save_solution(iter, parameters,fnout,pts_x,pts_y,gaia_CMD_to_fit,w0,weights,isochrones2,age,met,hist):
    
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
    
    if parameters['Fitting']['save_each_step']=='yes':
        sol.to_hdf(fnout,key='sol'+str(iter))
    else:
        sol.to_hdf(fnout,key='sol')
    
    sol = pd.DataFrame(isochrones2)
    sol.to_hdf(fnout,key='isochrones')

    sol = pd.DataFrame()
    sol['history'] = hist
    sol.to_hdf(fnout,key='history')

    with pd.HDFStore(fnout, mode="a") as store:
        store.put("iteration", pd.Series([iter]), format="table")
              
    

def fit_cmd(model):
    print('\n ###############################################################')
    print('    Fitting routine starting')
    print(' ###############################################################\n')

    isochrones, AGE, MET = read_isochrones(model)

    all_isochrones = np.sum(isochrones, axis=0)

    pts_x,pts_y,gaia_cmd = read_cmd_to_fit(model)

    if len(gaia_cmd) == isochrones.shape[1]:
        print('CMD size and isochrones size match')
    else:        
        print('CMD size ',len(gaia_cmd),' and isochrones size',isochrones.shape[1],'do not match. Exit now!')
        return
    
    if model.parameters['Fitting']['add_time_stamp']=='no':
        date_time_str = ''
        fn0 = model.parameters['General']['path']+'/results/'+model.parameters['Fitting']['model_name']
        fn1 = model.parameters['General']['path']+'/results/'+model.parameters['Fitting']['model_name']
    else:
        now = datetime.now()    
        date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        
        fn0 = model.parameters['General']['path']+'/results/'+model.parameters['Fitting']['model_name']+'.'+date_time_str
        fn1 = model.parameters['General']['path']+'/results/'+model.parameters['Fitting']['model_name']+'.'+date_time_str

    
    mkdir(model.parameters['General']['path']+'/figs/')
    mkdir(fn0)
    mkdir(fn1)
    if model.parameters['Fitting']['add_time_stamp']=='yes':
        figname_out0 = fn0+'/solution'+date_time_str 
        sol_file_name = fn1+'/solution'+date_time_str+'.h5'
    else:
        figname_out0 = fn0+'/solution' 
        sol_file_name = fn1+'/solution.h5'

            

    print('RUNNING THE MODEL:',sol_file_name)

    new_run_flag = True
    
    if file_exists(sol_file_name):
        print('The model already exists')

        # if yes_no_input("Do you want to rerun model with the same name y/n?"):
        #     print(" Rerunning model ",model.parameters['Fitting']['model_name'])
        
        #     new_run_flag = True
            
        # else:
        if True:
            initial_iteration, w0, hist = read_solution(sol_file_name, model.parameters)
            
            test_CMD = isochrones.T @ np.exp(w0)
            
            initial_iteration = initial_iteration + 1
            
            new_run_flag = False
            
    
    if new_run_flag:
        
        print('Running a new model')
        
        copy_file(model.config_file_name,fn1)
        
        initial_iteration = 1
        
        if file_exists(model.parameters['Fitting']['initial_guess']):
            with pd.HDFStore(model.parameters['Fitting']['initial_guess'], mode='r') as store:
                keys = store.keys()        
    
            print('Will use an initial guess from ',model.parameters['Fitting']['initial_guess'],'iteration',keys[-1])
            a = pd.read_hdf(model.parameters['Fitting']['initial_guess'],key=keys[-1])
            w0 = a['w'].values
            test_CMD = isochrones.T @ np.exp(w0)
        else:    
            w0 = []
            if model.parameters['Fitting']['initial_guess']=='uniform':     
                print('Uniform initial guess')
                w0 = AGE/AGE * np.gradient(AGE)
                w0 = np.log10(1+AGE)
            
            if model.parameters['Fitting']['initial_guess']=='blob':  
                print('Blob initial guess')
                w0 = np.log10( np.exp(-((AGE-11)/2.4)**2) * np.exp(-((MET+0.5)/0.1)**2) )
            
            if model.parameters['Fitting']['initial_guess']=='gradient':  
                print('Gradient initial guess')
                w0 = (1+10*AGE)**2 * np.gradient(AGE)
                w0 = np.log10(1+AGE)
    
            if len(w0)==0:
                print('!! ERROR in the initial guess !!')
                print('Will proceed with the Uniform weights')
                w0 = AGE/AGE * np.gradient(AGE)
                w0 = np.log10(1+AGE)
                
            e_w0 = np.exp(w0)
            test_CMD = isochrones.T @ e_w0
            e_w0 = e_w0 / test_CMD.sum() * len(test_CMD)
            test_CMD = isochrones.T @ e_w0
            w0 = np.log(e_w0)            
    
        figname_out = figname_out0+'.0000000.jpg' 
        plot_solution(pts_x, pts_y, gaia_cmd, test_CMD,AGE,MET,np.exp(w0),np.exp(w0),[0],figname_out)
        hist = []    
    
    ind = (all_isochrones==0) | (gaia_cmd/gaia_cmd.max()< float(model.parameters['Fitting']['cmd_density_range']))
    gaia_cmd[ind] = 0
    isochrones2 = isochrones.copy()
    isochrones2[:,ind] = 0
    
    for i in range(initial_iteration,int(model.parameters['Fitting']['max_step'])+1):

        # ampl = np.median(10**w0)*0.01
        # extra =  ampl * (2*np.random.rand(len(w0))-1)
        # extra = gaussian_filter1d(extra, sigma=3)
        # w0 = w0 + np.exp(extra)
        
        # noise_amplitude = (w0.max()-w0.min())*0.02
        # w0 = w0 + np.random.uniform(-noise_amplitude, noise_amplitude, size=w0.shape)
        

        print(model.parameters['Fitting']['model_name']+'.'+date_time_str,'running iteration ',i,' out of ',model.parameters['Fitting']['max_step'])
        figname_out = figname_out0 +'.'+ str(i).zfill(7)+'.jpg'
        current_weights,hist0 = solver(pts_x, pts_y, isochrones2.T, w0, gaia_cmd, float(model.parameters['Fitting']['eps']), 
                                       int(model.parameters['Fitting']['nsave']),fittype=model.parameters['Fitting']['fittype'])
        hist.extend(hist0)
        plot_solution(pts_x, pts_y, gaia_cmd, isochrones2.T @ np.exp(current_weights),AGE,MET,np.exp(w0),np.exp(current_weights),hist,figname_out)
        save_solution(i,model.parameters,sol_file_name,pts_x,pts_y,gaia_cmd,w0,current_weights,isochrones2,AGE,MET,hist)
        w0 = current_weights
        

    return 1

    

    