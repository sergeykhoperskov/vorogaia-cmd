import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import pandas as pd
import matplotlib.tri as tri

from scipy.spatial import cKDTree
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from scipy.spatial import Voronoi, voronoi_plot_2d

from scipy.optimize import minimize


def yes_no_input(prompt):
    """
    Prompt the user for a Yes/No input and return True for 'Y', False for 'N'.
    :param prompt: The question or message to display.
    :return: True for 'Y', False for 'N'.
    """
    while True:
        answer = input(f"{prompt} (Y/N): ").strip().lower()
        if answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")


def myjet(m=256):

    cdict1 = {'red':((0.0, 0.0, 0.0),
                   ( 0.4, 0.0, 0.0),
                   ( 0.6, 1.0, 1.0),
                   ( 1, 0.4, 0.4)),

         'green': ((0.0, 0.0, 0.0),
                   (0.1, 0.0, 0.0),
                   (0.5, 1.0, 1.0),
                   (0.9, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.4, 0.4),
                   (0.4, 1.0, 1.0),
                   (0.6, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }

    return mpl.colors.LinearSegmentedColormap('myjet', cdict1, m)

def copy_file(file, destination):
    print('copy file',file,' to ',destination)
    
    os.system('cp '+ file + ' ' + destination + '/')


def file_exists(file_path):
    if os.path.exists(file_path):
        # print(f"The file '{file_path}' exists.")
        return True
    else:
        # print(f"The file '{file_path}' does not exist.")
        return False

def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        # print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")


    
def plot_vor_density(pts,zz,cmap_loc = myjet()):
    nv = len(pts[:,0])
    vor = Voronoi(pts)
    kdtree = cKDTree(pts)

    norm = LogNorm(vmin=min(zz[zz>0]), vmax=max(zz), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap_loc)
    
    for j in range(0,nv):
        region = vor.regions[vor.point_region[j]]
        if not -1 in region:
            if zz[j]>0:
                polygon = [vor.vertices[i] for i in region]
                plt.fill(*zip(*polygon), color=mapper.to_rgba(zz[j]))
                
    voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_colors='k',
                line_width=1, line_alpha=0.6, point_size=0)


def make_voronoi(xx,yy, x0,y0, SCALE=1e6, targetSN=10):

    
    tmp,xedges,yedges = np.histogram2d(x0,y0,  bins = [xx,yy])
    xx = np.array( (xedges[:-1] + xedges[1:]) / 2 )
    yy = np.array( (yedges[:-1] + yedges[1:]) / 2 )

    plt.figure(figsize=(8, 6), dpi=80)    
    plt.pcolor(xx,yy,tmp.T,norm=LogNorm(),cmap=myjet())
    
    zz = np.array(tmp.T)+1

    zz = SCALE*zz/np.sum(zz)

    xx,yy = np.meshgrid(xx,yy)

    xx = xx[zz>0]
    yy = yy[zz>0]
    zz = zz[zz>0]

    signal = zz**0.5
    noise = signal**0.5 


    out  = voronoi_2d_binning(xx, yy, signal, noise, targetSN, plot=0, quiet=1, wvt=True, pixelsize=1)
    binNum, xx1, yy1, xBar, yBar, sn, nPixels, scale = out

    pts = np.vstack((xx1,yy1)).T

    print('voronoi binning done',pts.shape)

    vor = Voronoi(pts)
    

    voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_colors='k',
                    line_width=1, line_alpha=0.6, point_size=0)

    plt.xlim(-0.6,2.5)
    plt.ylim(-5,5)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()

    return pts


def make_vor_density(pts,x0,y0,m=[]):

    nv = len(pts[:,0])

    vor = Voronoi(pts)
    kdtree = cKDTree(pts)
    test_point_dist, test_point_regions = kdtree.query(np.vstack((x0,y0)).T)

    
    zz = np.zeros(nv).copy()*0
    if len(m)==len(x0):
        unique_values = np.unique(test_point_regions)
        indexes_dict = {val: np.where(test_point_regions == val)[0] for val in unique_values}
    
        for val, indexes in indexes_dict.items():
            zz[val] = np.sum(m[indexes])
    else:
        val, num = np.unique(test_point_regions, return_counts=True)
        zz[val] = num

    return zz

def plot_vor_density2(ax,pts,zz,vals,tit,scale='log',nc=100):

    nv = len(pts[:,0])

    vor = Voronoi(pts)
    kdtree = cKDTree(pts)

    if scale=='log':
        norm = LogNorm(vmin=vals[0], vmax=vals[1], clip=True)
    else:
        norm =  mpl.colors.Normalize(vmin=vals[0], vmax=vals[1], clip=True)

    zz[zz<vals[0]] = vals[0]
    zz[zz>vals[1]] = vals[1]
    # zz[np.isnan(zz)] = vals[0] # ??
    
    mapper = cm.ScalarMappable(norm=norm, cmap=myjet(nc))
  
    for j in range(0,nv):
        region = vor.regions[vor.point_region[j]]
        if not -1 in region:
            if np.abs(zz[j])>=0:
                polygon = [vor.vertices[i] for i in region]
                im = ax.fill(*zip(*polygon), color=mapper.to_rgba(zz[j]))

    mapper.set_array(zz)
    cbar = plt.colorbar(mapper, ax=ax)

    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='k',
                line_width=0.5, line_alpha=0.2, point_size=0)

    
    ax.set_xlabel("BP-RP [mag]", fontsize=14)
    ax.set_ylabel("Gmag [mag]", fontsize=14)
    ax.set_title(tit, fontsize=16)
    ax.set_xlim(-0.6,2.5)
    ax.set_ylim(-5,5)
    ax.invert_yaxis()

def plot_solution(px,py, y0, y1,age,met,w0,w1,hist,fig_name):
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    cmd_density_scale = y0[y0>0].min()/y0.max()

    y0[y0<y0.max()*cmd_density_scale] = 0.0 # ???
    plot_vor_density2(axes[0, 0],np.column_stack((px, py)),y0,[y0.max()*cmd_density_scale,y0.max()],'Gaia CMD '+str(int(sum(y0>0))),scale='log')

    y1[y0==0] = 0.0 # ???
    plot_vor_density2(axes[0, 1],np.column_stack((px, py)),y1,[y0.max()*cmd_density_scale,y0.max()],'Current solution',scale='log')
    
    tmp = y0-y1
    # ind = np.abs(tmp)>0
    plot_vor_density2(axes[0, 2],np.column_stack((px, py)),tmp,[-np.max(np.abs(tmp)), np.max(np.abs(tmp))],'Gaia-Solution',scale='lin')
   
    tmp = tmp/np.sqrt(y0)
    tmp[y0/y0.max()<cmd_density_scale] = np.nan
    # ind = np.abs(tmp)>0
    plot_vor_density2(axes[1, 0],np.column_stack((px, py)),tmp,[-np.max(np.abs(tmp)), np.max(np.abs(tmp))],'(Gaia-Solution)/sqrt(Gaia)',scale='lin',nc=21)

    triang = tri.Triangulation(age, met)

    tmp = np.log10(w1)
    tmp[tmp.max()-tmp>4] = tmp.max()-4
    tpc = axes[1, 1].tripcolor(triang, tmp, shading='flat', cmap=myjet())
    cbar = plt.colorbar(tpc, ax=axes[1, 1])

    data = pd.DataFrame({'age': age, 'weight': w1})
    summed_weights = data.groupby('age')['weight'].sum().reset_index()
    values = summed_weights['weight']/np.gradient(summed_weights['age'].values)
    values = values/np.max(values)+met.min()
    
    axes[1,1].plot(summed_weights['age'].values,values,c='k',linewidth=3)
    axes[1,1].set_xlim(0,14)
    axes[1,1].set_ylim(-2,0.5)
    axes[1, 1].set_xlabel('Age [Gyr]', fontsize=14)
    axes[1, 1].set_ylabel('Metallicity [dex]', fontsize=14)
    axes[1, 1].set_title('log10(weights)', fontsize=14)
    # axes[1, 1].clim(np.log10(w1.max()*1e-5),np.log10(w1.max()))
    
    im1 = axes[1, 2].plot(hist[-50000:-1])
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_title("Residuals: "+str(hist[-1])+'  '+str(len(hist)), fontsize=14)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(fig_name) 
    
    # Show the plot
    # plt.show()
    plt.close()



def function_rel(X, m, y):
    mask = y > 0
    return (y[mask] - X[mask] @ np.exp(m)) / np.sqrt(y[mask])

def partial_derivative_rel(m_stat, X, y):
    n = len(X)
    mask = y > 0  
    fun = function_rel(X[mask], m_stat, y[mask])
    weight = 1 / np.sqrt(y[mask])
    df_dm = (-2 / n) * (X[mask].T @ (fun * weight)) * np.exp(m_stat)

    return df_dm

        
def mean_squared_error_rel(m_stat, X, y):
    mask = y > 0  
    return np.sum(function_rel(X[mask], m_stat, y[mask]) ** 2, axis=0) / len(X[mask])


def function(m,X,y):
  
    return (y - X @ np.exp(m) )

def partial_derivative(m_stat, X, y):

    n = len(X)

    fun = function(m_stat,X,y)
    
    df_dm =  (-2/n) * (X.T @ fun) * np.exp(m_stat)

    return df_dm

def mean_squared_error(m_stat,X,y):
                 
    return np.sum((function(m_stat,X,y)**2),axis = 0) / len(X)

def solver(pts_x,pts_y, X, m_stat, y, eps, max_counter,fittype='abs'):
  
    hist = []
    
    if fittype == 'default':
        options = {
            'disp': True,          # Enable verbosity
            'maxiter': 100}
        
        result = minimize(mean_squared_error_rel, m_stat, method='L-BFGS-B', options=options, args=(X, y))
        m_stat = result.x
        hist = [result.fun]

    if fittype == 'abs':
        for counter in range(0,max_counter):
            m_stat = m_stat - eps * partial_derivative(m_stat, X,  y)
            err = mean_squared_error(m_stat, X, y)
            hist.append(err)
            
    if fittype == 'rel':
        for counter in range(0,max_counter):
            m_stat = m_stat - eps * partial_derivative_rel(m_stat, X, y)    
            err = mean_squared_error_rel(m_stat, X, y)
            hist.append(err)        
                       
    return m_stat, hist

def nbt2den(x,y,z,x1,x2,y1,y2,n1,n2):
    xx = np.linspace(x1,x2,n1+1)
    yy = np.linspace(y1,y2,n2+1)
    
    if len(z)<len(x):
        zz,xx,yy = np.histogram2d(x,y,bins=[xx,yy])
    else:
        # z = np.reshape(z,len(z))
        zz,xx,yy = np.histogram2d(x,y,bins=[xx,yy],weights=z)

    xx = (xx[:-1] + xx[1:]) / 2
    yy = (yy[:-1] + yy[1:]) / 2
    
    return xx,yy,zz.T

