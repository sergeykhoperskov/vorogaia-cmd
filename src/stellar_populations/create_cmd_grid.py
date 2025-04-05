import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools import file_exists, mkdir, make_voronoi, myjet, make_vor_density, plot_vor_density2


def create_cmd_grid(model):
    print('\n ######################################')
    print('    Reading/writing CMD to fit ')
    print(' ########################################\n')

    fno = model.grid_file_name
    
    if file_exists(fno+'.h5'):
        print('Reading binned CMD...')
        model.pts_df = pd.read_hdf(fno+'.h5',key='dat')
               
    else:
        tmp = pd.read_hdf(model.parameters['CMD_grid']['gaia_file'],
                         key=model.parameters['CMD_grid']['gaia_file_key'])
        print(tmp.columns)
        print('Size of the Gaia file: ',len(tmp))
        for i in ['bp_col','rp_col','bprp_col']:
            print(model.parameters['CMD_grid'][i],model.parameters['CMD_grid'][i] in tmp.columns)

        if (model.parameters['CMD_grid']['bp_col'] == 'none') | (model.parameters['CMD_grid']['rp_col'] == 'none'): 
            model.df_gaia = tmp[[model.parameters['CMD_grid']['gmag_col'],model.parameters['CMD_grid']['bprp_col']]]
        else: 
            tmp['bprp_col'] = tmp[model.parameters['CMD_grid']['bp']]-tmp[model.parameters['CMD_grid']['rp_col']]
            model.df_gaia = tmp[[model.parameters['CMD_grid']['gmag_col'],'bprp_col']]
        
        model.df_gaia = model.df_gaia.rename(columns={model.parameters['CMD_grid']['gmag_col']: "M_G", 'bprp_col': "BP_RP"})

        print(len(model.df_gaia))
    
        xx = np.linspace(-0.6,2.5,int(model.parameters['CMD_grid']['nx']))
        yy = np.linspace(-5,5, int(10 * len(xx) / (xx.max()-xx.min())))
        print('number of initial cells',len(xx)*len(yy))
        pts = make_voronoi(xx,yy,model.df_gaia['BP_RP'],model.df_gaia['M_G'],
                           SCALE=float(model.parameters['CMD_grid']['scale']),
                           targetSN=float(model.parameters['CMD_grid']['sn']))
    
        model.pts_df = pd.DataFrame(pts, columns=['pts_x', 'pts_y'])
        ind = (np.abs(model.df_gaia['M_G'])<5) & (model.df_gaia['BP_RP']>-0.5) & (model.df_gaia['BP_RP']<2.5)    
        dat = make_vor_density(pts,model.df_gaia['BP_RP'][ind],model.df_gaia['M_G'][ind])

        print('Number of >0 density values',sum(dat>0))
            
        mkdir(model.parameters['General']['path']+'/dat/cmd_grid')
        model.pts_df.to_hdf(fno+'.h5', key='dat')

        
        print('File',model.parameters['CMD_grid']['gaia_file'],'binned correctly as')
        print(fno)
    
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        plot_vor_density2(axes,model.pts_df[['pts_x', 'pts_y']].to_numpy(),
                          dat,[1e-3*dat.max(),dat.max()],
                          'Gaia CMD used to make grid',scale='log')
        plt.title(model.parameters['CMD_grid']['gaia_file']+' '+str(sum(dat>0)))
        
        mkdir(model.parameters['General']['path']+'/figs/_cmds_to_fit/')
        
        plt.savefig(model.grid_figure_file_name+'.jpg')
