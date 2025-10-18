# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:05:14 2023

@author: Noga Mudrik
pip install pylops==1.18.2

"""


"""
CREIMBO: Cross-Regional Ensemble Interactions in Multi-view Brain Observations (ICLR 2025) 
@code: Noga 
"""


#%%%%%%%%%%%%%%%%%%%%%%%%
# IMPORTS

"""
Imports
"""
import scipy.io as sio
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime as datetime2
from scipy.sparse import coo_matrix  
from numpy.linalg import matrix_power
from scipy.linalg import expm
from math import e
from numpy.core.shape_base import stack
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import random
from pathlib import Path
import os
import pickle
from tkinter.filedialog import askopenfilename
from datetime import date

import scipy.io

import warnings
import statsmodels as stats
from importlib import reload  
import statsmodels.stats as st
sep = os.sep

from importlib import reload  
from scipy.interpolate import interp1d

from scipy import interpolate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pylops

from statistics import mode
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn import svm
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.linalg import block_diag
from scipy.optimize import nnls
from matplotlib import colors
from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
        
global ss, today, full_date
full_date = str(datetime2.now())
ss = int(str(datetime2.now()).split('.')[-1])
full_date = full_date.replace('-','_').replace(':', '_').replace('.','_')
today = full_date.split()[0]
#%%% MODEL FUNCTIONS

def name_to_rgb(color):
    """Converts a color name to its RGB representation as a tuple of RGBA values in the range [0, 1]."""
    return colors.to_rgba(color)







#%% create specific dynamics
def create_lorenz_mat(t_val = 10, initial_conds = (0., 1., 1.05), alpha = 10 , beta  = 25 , gamma = 2.67, dt = 0.01,
                      direction = 'reg', multi_min_max = [0.9, 1.1]):
    """
    Create a Lorenz system dynamics.
    
    Args:
        t_val (float, optional): The maximum time value (default: 10).
        initial_conds (tuple, optional): The initial conditions (default: (0., 1., 1.05)).
        alpha (float, optional): The parameter alpha (default: 10).
        beta (float, optional): The parameter beta (default: 25).
        gamma (float, optional): The parameter gamma (default: 2.67).
        dt (float, optional): The time step size (default: 0.01).
        direction (str, optional): The direction of the system (default: 'reg').
        multi_min_max (list, optional): The minimum and maximum values for the multiplier (default: [0.9, 1.1]).
    
    Returns:
        numpy.ndarray: The dynamic matrix.
        numpy.ndarray: The Jacobian matrix.
        numpy.ndarray: The coefficients matrix.
        list: The sub-dynamics matrices.
    
    """
    cur_lorenz = np.array(initial_conds).reshape((-1,1))

    t = np.arange(0,t_val, dt)
    real_jac = []
    
    real_cs = [ np.array([1, cur_lorenz[0],cur_lorenz[1], cur_lorenz[2]]).reshape((-1,1))  ]
    f0 = np.array([[-alpha,alpha,0], [beta , -1,0], [0,0,-gamma]])*dt + np.eye(3)
    f1 = np.array([[0,0,0], [0 , 0 , -1], [0,1, 0]])*dt
    f2 = np.array([[0,0,0], [0, 0, 0], [1,0,0]])*dt
    f3 = np.array([[0,0,0], [-1, 0, 0], [0,0,0]])*dt
    for t_spec in range(len(t)):
        y = cur_lorenz[1,t_spec]
        x = cur_lorenz[0,t_spec]
        z = cur_lorenz[2,t_spec]       
        real_cs.append(np.array([1,x,y,z]).reshape((-1,1)))
        
        new_mat = np.sum(np.dstack([f0, f1*x, f2*y, f3*z   ]),2)

        if direction == 'reg':
            next_jac =  new_mat #(new_mat*dt   + np.eye(3) )
            next_d = next_jac @ cur_lorenz[:,-1]
        else:
            multi = np.random.rand()*(multi_min_max[1] - multi_min_max[0]) + multi_min_max[0]
            next_jac = multi*(new_mat*dt   + np.eye(3) )
            next_d = next_jac @ cur_lorenz[:,-1]
        cur_lorenz = np.hstack([ cur_lorenz, next_d.reshape((-1,1))])
        real_jac.append(next_jac)
    fs = [f0, f1,f2,f3]
    cs =  np.hstack(real_cs)[:,1:]

    cur_lorenz = build_reco_again(cur_lorenz[:,0], fs, cs)
    
    return cur_lorenz, np.dstack(real_jac), cs,  fs 


def create_van_der_pol_mat(t_val = 10, initial_conds = (0., 1.), dt = 0.01,  direction = 'reg', mu = 4):
    """
    Create the Van der Pol matrix and corresponding trajectory.
    
    Args:
        t_val (float, optional): Total time value (default: 10).
        initial_conds (tuple, optional): Initial conditions for the trajectory (default: (0., 1.)).
        dt (float, optional): Time step size (default: 0.01).
        direction (str, optional): Direction of the matrix:
                                   - 'reg': Regular direction (default).
                                   - 'rev': Reversed direction.
        mu (float, optional): Parameter value for the Van der Pol equation (default: 4).
    
    Returns:
        numpy.ndarray: The trajectory matrix.
        numpy.ndarray: The stack of Jacobian matrices.
    
    """
    cur_pol = np.array(initial_conds).reshape((-1,1))
    t = np.arange(0,t_val, dt)
    real_jac = []
    for t_spec in range(len(t)):
        
        x =  cur_pol[0,t_spec]
        y =  cur_pol[1,t_spec]
       
        new_mat = np.array([[mu - mu*x*x/3,-mu], [1/mu, 0]])

        next_jac =  (new_mat*dt   + np.eye(2) )
        next_d = next_jac @  cur_pol[:,-1]

        cur_pol = np.hstack([  cur_pol, next_d.reshape((-1,1))])
        real_jac.append(next_jac)
        
    return  cur_pol, np.dstack(real_jac)


def create_smooth_trans(dyn1 = [], dyn2 = [], coeffs = [], max_time = 200, dt = 0.1, sig_e = 5):
    """
    Create a smooth transition between two dynamics.
    
    Args:
        dyn1 (numpy.ndarray, optional): The first dynamic matrix (default: []).
        dyn2 (numpy.ndarray, optional): The second dynamic matrix (default: []).
        coeffs (numpy.ndarray, optional): The coefficients for blending the dynamics (default: []).
        max_time (float, optional): The maximum time value (default: 200).
        dt (float, optional): The time step size (default: 0.1).
        sig_e (float, optional): The range of the sigmoid function (default: 5).
    
    Returns:
        numpy.ndarray: The blended dynamic matrix.
        numpy.ndarray: The first dynamic matrix.
        numpy.ndarray: The second dynamic matrix.
        numpy.ndarray: The coefficients used for blending.
    
    """
    if dyn1 == [] or dyn2 == []:
        dyn_perm = create_dynamics('f_spiral' , max_time= max_time, 
                                   dt = dt, return_3d=True)

        dyn1 = dyn_perm
        dyn2 = np.vstack([dyn_perm[2,:], dyn_perm[1,:], dyn_perm[0,:]])
    if coeffs == []:
        c1 = 0.4*sigmoid(np.linspace(-sig_e ,sig_e ,dyn_perm.shape[1]))
        c2 = 0.4*(1-sigmoid(np.linspace(-sig_e ,sig_e ,dyn_perm.shape[1])))
        coeffs = np.vstack([c1,c2])
    v2 = c1.reshape((1,-1)) * dyn1 + c2.reshape((1,-1)) * dyn2#[c1[t]*dyn1 + c2[t]*dyn2 for t in range(dyn_perm.shape[1])]
    return v2, dyn1, dyn2, coeffs

def create_duffing(t_val = 10, initial_conds = (0., 1.), dt = 0.01,  direction = 'reg',
                   gamma = 0.02, delta = 0.001, alpha = 1, beta = 50, 
                   w = 0.5):
    """
    Create a Duffing oscillator dynamics.

    Args:
        t_val (float, optional): The maximum time value (default: 10).
        initial_conds (tuple, optional): The initial conditions (default: (0., 1.)).
        dt (float, optional): The time step size (default: 0.01).
        direction (str, optional): The direction of the oscillator (default: 'reg').
        gamma (float, optional): The damping coefficient (default: 0.02).
        delta (float, optional): The offset amplitude (default: 0.001).
        alpha (float, optional): The stiffness coefficient (default: 1).
        beta (float, optional): The nonlinearity coefficient (default: 50).
        w (float, optional): The angular frequency of the offset (default: 0.5).

    Returns:
        numpy.ndarray: The dynamic matrix.
        numpy.ndarray: The Jacobian matrix.
        list: The masks for the Jacobian matrices.
        numpy.ndarray: The offset values.
        list: The sub-dynamics matrices.

    """
    cur_pol = np.array(initial_conds).reshape((-1,1))
    t = np.arange(0,t_val, dt)
    real_jac = []
    offset_full = []
    for t_spec in range(len(t)):
        
        x =  cur_pol[0,t_spec]
        y =  cur_pol[1,t_spec]
        offset = np.array([0, delta * np.cos(w*t_spec)]).reshape((-1,1))
       
        new_mat = np.array([[0,1],[ -beta*x*x - alpha,-gamma]])

      
        next_jac =  (new_mat*dt   + np.eye(2) )
        next_d = (next_jac @  cur_pol[:,-1]).reshape((-1,1)) + offset

        cur_pol = np.hstack([  cur_pol, next_d.reshape((-1,1))])
        real_jac.append(next_jac)
        offset_full.append(offset)

    real_fs = [np.array([  [1,dt],  [-alpha*dt , 1-gamma*dt]]) , 
    np.array( [ [0,0],[-beta*dt, 0] ]   ) ]
    
    masks = [f_spec != 0 for f_spec in real_fs]
    
    return  cur_pol, np.dstack(real_jac), masks, np.hstack(offset_full), real_fs


#%% PRE-PROCESSING

def init_mat(size_mat, r_seed = 0, dist_type = 'norm', init_params = {'loc':0,'scale':1, 'k':0.8}, normalize = False):
  """
  This is an initialization function to initialize matrices like G_i and c. 
  Inputs:
    size_mat    = 2-element tuple or list, describing the shape of the mat
    r_seed      = random seed (should be integer)
    dist_type   = distribution type for initialization; can be 'norm' (normal dist), 'uni' (uniform dist),'inti', 'sparse', 'regional', 'zeros'
    init_params = a dictionary with params for initialization. The keys depends on 'dist_type'.
                  keys for norm -> ['loc','scale']
                  keys for inti and uni -> ['low','high']
                  keys for sparse -> ['k'] -> number of non-zeros in each row
                  keys for regional -> ['k'] -> repeats of the sub-dynamics allocations
    normalize   = whether to normalize the matrix
  Output:
      the random matrix with size 'size_mat'
  """
  np.random.seed(r_seed)
  random.seed(r_seed)
  if dist_type == 'norm':
      print(init_params)
      print(size_mat)
          
      rand_mat = np.random.normal(loc=init_params['loc'],scale = init_params['scale'], size= size_mat)
    
  elif dist_type == 'uni':
    init_params = {**{'low':0,'high':1 }, **init_params} 
    #if 'high' not in init_params.keys() or  'low' not in init_params.keys():
    #    raise KeyError('Initialization did not work since low or high boundries were not set')
    rand_mat = np.random.uniform(init_params['low'],init_params['high'], size= size_mat)
  elif dist_type == 'inti':
    if 'high' not in init_params.keys() or  'low' not in init_params.keys():
      raise KeyError('Initialization did not work since low or high boundries were not set')
    rand_mat = np.random.randint(init_params['low'],init_params['high'], size= size_mat)
  elif dist_type == 'sparse':
    if 'k' not in init_params.keys():
      raise KeyError('Initialization did not work since k was not set')

    k = init_params['k']
    if k < 1:
        k = int(size_mat[0]*k)
    b1 = [random.sample(list(np.arange(size_mat[0])),np.random.randint(1,np.min([size_mat[0],k]))) for i in range(size_mat[1])]
    b2 = [[i]*len(el) for i,el in enumerate(b1)]
    rand_mat = np.zeros((size_mat[0], size_mat[1]))
    rand_mat[np.hstack(b1), np.hstack(b2)] = 1
  elif dist_type == 'regional':
    if 'k' not in init_params.keys():
      raise KeyError('Initialization did not work since k was not set for regional initialization')

    k=init_params['k']
    splits = [len(split) for split in np.split(np.arange(size_mat[1]),k)]
    cur_repeats = [np.repeat(np.eye(size_mat[0]), int(np.ceil(split_len/size_mat[0])),axis = 1) for split_len in  splits]
    cur_repeats = np.hstack(cur_repeats)[:size_mat[1]]
    
    rand_mat = cur_repeats
  elif dist_type == 'zeros':
      rand_mat = np.zeros( size_mat)
  else:
    raise NameError('Unknown dist type!')
  if normalize:
    rand_mat = norm_mat(rand_mat)
  return rand_mat

def find_dominant_dyn(coefficients):
    """
    This function finds the # of the most dominant dynamics in each time point. It should be used when comparing to rsLDS
    Input:
        coefficients: np.array of kXT
    Output:
        an array with len T, containing the index of the most dominant sub-dynamic at each time point
    """
    domi = np.argmax(np.abs(coefficients),0)
    return domi  
  
def norm_mat(mat, type_norm = 'evals', to_norm = True):
  """
  This function comes to norm matrices by the highest eigen-value
  Inputs:
      mat       = the matrix to norm
      type_norm = what type of normalization to apply. Can be only 'evals' for now.
      to_norm   = whether to norm or not to.
  Output:  
      the normalized matrix
  """    
  if to_norm:
    if type_norm == 'evals':
        eigenvalues, _ =  linalg.eig(mat)
        mat = mat / np.max(np.abs(eigenvalues))
    elif type_norm == 'max_abs':
        mat = mat / np.max(np.abs(mat))
    elif type_norm == 'none':
        pass
  return mat

def norm_mat_by_max(mat, to_norm = True, type_norm = 'exp'):
  """
  normalize a matrix by dividing by its max value or by fixing the determinant to 1
  
  
  """  
  if to_norm:
    if type_norm == 'max':
      mat = mat / np.max(np.abs(mat))
    elif type_norm  == 'exp':
      mat = np.exp(-np.trace(mat))*expm(mat)
  return mat
# %% MODEL FUNCTIONS
def solve_lasso_problem(y, A, reg, update_type = 'inv', random_state = 0, params_update_c  = {}):

    params_update_c = {**{'update_c_type':'inv', 'smooth_term': 0, 'reg_term':0, 'num_iters':10},
                       **params_update_c}

    if update_type == 'inv' :
      try:
          coeffs = linalg.pinv(A) @ y.reshape((-1,1))
      except:
          raise NameError('A problem in taking the inverse of fx when looking for the model coefficients')
    
    elif update_type == 'nls' or reg == 0:       
        
        # Perform non-negative least squares optimization
        try:
            coeffs, _ = nnls(A, y)
        except:
            coeffs = np.linalg.pinv(A) @ y.reshape((-1,1))
        
   
          
    elif update_type  == 'lasso' :

      clf = linear_model.Lasso(alpha=reg,random_state=random_state)
      clf.fit(A,y.T )     
      coeffs = np.array(clf.coef_)

    elif update_type .lower() == 'fista' :
        Aop = pylops.MatrixMult(A)

        if 'threshkind' not in params_update_c: params_update_c['threshkind'] ='soft'

        coeffs = pylops.optimization.sparsity.FISTA(Aop, y.flatten(), niter=params_update_c['num_iters'],
                                                    eps = reg , threshkind =  params_update_c.get('threshkind') )[0]

    elif update_type.lower() == 'ista' :
        print('ista')
        
        if 'threshkind' not in params_update_c: params_update_c['threshkind'] ='soft'
        Aop = pylops.MatrixMult(A)
        coeffs = pylops.optimization.sparsity.ISTA(Aop, y.flatten(), niter=params_update_c['num_iters'] , 
                                                   eps = reg,threshkind =  params_update_c.get('threshkind'))[0]
   
        
        
    elif update_type.lower() == 'omp' :
       
       
     
        # Create an OMP object
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=int(np.ceil(reg))) 
        # Set k as the desired number of non-zero elements in x
        
        # Fit the OMP model
        omp.fit(A, y.flatten())
        
        
        # Get the estimated sparse coefficients
        coeffs = omp.coef_
        ss = ((A @ coeffs  - y)**2).sum()
        print('er here: %s'%str(ss))
        
        # The indices of non-zero elements in the solution x
        #nonzero_indices = np.where(x_sparse != 0)[0]
        
        # The non-zero elements in the solution x
        #onzero_elements = x_sparse[nonzero_indices]
                
        
    elif update_type.lower() == 'spgl1' :
        #print('spgl1')
        Aop = pylops.MatrixMult(A)
        coeffs = pylops.optimization.sparsity.SPGL1(Aop, y.flatten(),iter_lim = params_update_c['num_iters'],
                                                   tau = reg, verbosity = False)[0]
        
        
    elif update_type.lower() == 'irls' :
        print('irls')
        Aop = pylops.MatrixMult(A)
        
    
        coeffs = pylops.optimization.sparsity.IRLS(Aop, y.flatten(),  nouter=50, 
                                                   espI = reg)[0]
    return coeffs
def infer_c_x(data_i, D_i, F,  lambda_c = 1.2, lambda_x = 1.5, smooth_c = 0.9, num_iters = 100, smooth_x = 4):
    raise ValueError('depracated!')
    D = D_i.copy()
    num_times = data_i.shape[1]
    latent_dim = D_i.shape[1]
    num_subdyns = len(F)
    num_obs = data_i.shape[0]
    x = []
    c = []
    
    
    for t in range(num_times):
      
        """
        go over time
        """       
        
        if t == 0:
            Aop = pylops.MatrixMult(D)
            x_next = pylops.optimization.sparsity.SPGL1(Aop, data_i[:,t].flatten(), iter_lim = num_iters,
                                                       tau = lambda_x)[0]
            lambda_x = 1

            x.append(x_next)
        else:
          
            Fx = np.hstack([  (f_i @ x[-1].reshape((-1,1))).reshape((-1,1)) for f_i in F ])
            left = np.vstack([1/lambda_x * data_i[:,t].reshape((-1,1)), np.zeros(( latent_dim, 1))])
            right_top = 1/lambda_x * np.hstack(   [D_i, np.zeros((       num_obs, num_subdyns        )) ]      ) 
            right_bottom = np.hstack(   [np.eye(latent_dim)  ,  1/lambda_c * Fx ]      ) 
            right = np.vstack([right_top, right_bottom])
            if smooth_c != 0 and t > 1:
                right_new = np.hstack([   np.zeros((num_subdyns, latent_dim)),  1/lambda_c* np.eye(num_subdyns)    ])
                
            
                right = np.vstack([right, smooth_c * right_new ])
                
                left = np.vstack([left,  smooth_c * c[-1].reshape((-1,1)) ])
                    
                
                
            if smooth_x != 0 and t >= 1:
                right_new = np.hstack([   np.eye(latent_dim), np.zeros((latent_dim,num_subdyns))    ])
                
          
                right = np.vstack([right, smooth_x * right_new ])
                
                left = np.vstack([left,  smooth_x * x[-1].reshape((-1,1)) ])
                
            Aop = pylops.MatrixMult(right)
            x_and_coeffs = pylops.optimization.sparsity.SPGL1(Aop, left.flatten(), iter_lim = num_iters,
                                                       tau = 2,  verbosity = False)[0]
            x_next = x_and_coeffs[:latent_dim]
            c_next = x_and_coeffs[latent_dim:]
            c.append(c_next)
            x.append(x_next)
    c = np.vstack(c).T/lambda_c
    x = np.vstack(x).T
    
    return x, c    

    
        
def infer_under_mask(y, A, x_mask = [], k = 1, params_update_c = {}, indices_mask = [], vec = []): 
    # solve y = Ax for x under mask for x
    params_update_c = {**{'update_c_type':'inv', 'smooth_term': 0, 'reg_term':0, 'num_iters':10},
                       **params_update_c}
    
    """
    FIRST CHECK WHAT IS THE MASK / UPDATE THE MASK    
    """
    if checkEmptyList(x_mask) and  checkEmptyList(indices_mask):        
        if checkEmptyList(vec):
            vec = solve_lasso_problem(y, A,  params_update_c['reg_term'], params_update_c['update_type'] , 0,   params_update_c)
        _ , indices_mask = keep_max_vec(vec,k, return_indices = True) # GET THE INDICES OF THE MASK
    elif checkEmptyList(x_mask):
      
        indices_mask = np.where(x_mask != 0)[0]
    if checkEmptyList(indices_mask) or len(indices_mask) == 0:

        raise ValueError('empty!')
        # change to x inference if empty
        

    A_s = A[:,indices_mask ]    
    if len(indices_mask) == 1:
        A_s = A_s.reshape((-1,1))
    reg = params_update_c['reg_term']

    vec_new = solve_lasso_problem(y, A_s, 0, params_update_c.get('update_c_type', 'spgl1'),  0) 

    zers = np.zeros(A.shape[1])    
    zers[indices_mask] = vec_new.flatten()

    return zers    
            
            
def update_c(F, latent_dyn, 
             params_update_c = {'update_c_type':'inv','reg_term':0,'smooth_term':0, 'to_norm_fx' : False},clear_dyn = [], 
             direction = 'c2n',other_params = {'warm_start':False},random_state=0 , skip_error = False, coefficients = [],
             given_stacked_fx = False,
             stacked_fx_list  = [],
             include_identity = False,x_together = False, 
             data_i = [], D = [], 
             lambda_x = 0, weight_observation_eq = 150): # add coeffs as input
  """  
  The function comes to update the coefficients of the sub-dynamics, {c_i}, by solving the inverse or solving lasso.
  Inputs:
      F               = list of sub-dynamics. Should be a list of k X k arrays. 
      latent_dyn      = latent_dynamics (dynamics dimensions X time)
      params_update_c = dictionary with keys:
          update_c_type  = options:
               - 'inv' (least squares)
               - 'lasso' (sklearn lasso)
               - 'fista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.FISTA.html)
               - 'omp' (https://pylops.readthedocs.io/en/latest/gallery/plot_ista.html#sphx-glr-gallery-plot-ista-py)
               - 'ista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.ISTA.html)       
               - 'IRLS' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.IRLS.html)
               - 'spgl1' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.SPGL1.html)
               
               
               - . Refers to the way the coefficients should be claculated (inv -> no l1 regularization)
          reg_term       = scalar between 0 to 1, describe the reg. term on the coefficients
          smooth_term    = scalar between 0 to 1, describe the smooth term on the coefficients (c_t - c_(t-1))
      direction      = can be c2n (clean to noise) OR  n2c (noise to clean)
      other_params   = additional parameters for the lasso solver (optional)
      random_state   = random state for reproducability (optional)
      skip_error     = whether to skip an error when solving the inverse for c (optional)
      coefficients    = needed only if smooth_term > 0.
      This is the reference coefficients matrix to apply the constraint (c_hat_t - c_(t-1)) on. 
      include_identity = whether to include the identity term, s.t., x_(t+1) = \sum_i (I + c_it * f_i ) @ x_t
  Outputs: 
      coefficients matrix (k X T), type = np.array
      
  example:
  coeffs = update_c(np.random.rand(2,2), np.random.rand(2,15),{})
  
  future - just use the  solve_lasso_problem() instead of writing twice
  """  

  if x_together and (checkEmptyList(data_i) or checkEmptyList(D)):
      raise ValueError('if together with x, you must provide data')
      
  if x_together and lambda_x == 0:
      lambda_x = params_update_c['reg_term']*5
      
  if x_together and len(latent_dyn) == 0:
      latent_dyn = np.zeros((D.shape[1], data_i.shape[1]))
      
  if isinstance(latent_dyn,list):
    if len(latent_dyn) == 1: several_dyns = False
    else: several_dyns = True
  else:
    several_dyns = False
  if x_together:
      if isinstance(latent_dyn, list): 
        n_times = latent_dyn[0].shape[1]
      else:
        n_times = latent_dyn.shape[1]
  else:
          
      if isinstance(latent_dyn, list): 
        n_times = latent_dyn[0].shape[1]-1
      else:
        n_times = latent_dyn.shape[1]-1
    
    
  params_update_c = {**{'update_c_type':'inv', 'smooth_term': 0, 'reg_term':0, 'num_iters':10},**params_update_c}
  if len(clear_dyn) == 0:
    clear_dyn = latent_dyn

  if direction == 'n2c':
    latent_dyn, clear_dyn  =clear_dyn,  latent_dyn
  if isinstance(F,np.ndarray): F = [F]
  coeffs_list = []
  
  if given_stacked_fx and len(stacked_fx_list) ==0: 
      raise ValueError('If "given_stacked_fx" then stack_fx should not be empty')

  for time_point in np.arange(n_times):
    if x_together:
        lambda_c = params_update_c['reg_term']
        if not several_dyns:
            if  time_point  > 0:
                """
                find total_next_dyn_full
                """
                next_dyn1 = data_i[:,time_point].reshape((-1,1))*weight_observation_eq*lambda_x

                next_dyn2 = np.zeros((latent_dyn.shape[0],1))
                if params_update_c['smooth_term']>0:
                    next_dyn3 = coeffs_list[-1].reshape((-1,1))
                    total_next_dyn_full = np.vstack([next_dyn1, next_dyn2, next_dyn3 ]) 
                else:
                    total_next_dyn_full = np.vstack([next_dyn1, next_dyn2])
                
                
                """
                find stacked_fx_full 
                """
                cur_dyn = latent_dyn[:, time_point-1]
                f_x_mat = []
                for f_i in F:
                    f_x_mat.append(f_i @ cur_dyn)
                stacked_fx = np.vstack(f_x_mat).T 
                stacked_fx[stacked_fx> 10**8] = 10**8
                if len(F) == 1: stacked_fx = np.reshape(stacked_fx,[-1,1])
                
                block1 = np.hstack([D*weight_observation_eq, np.zeros((D.shape[0], len(F)))])
                block2 = np.hstack([1/lambda_x*np.eye(latent_dyn.shape[0]), 
                                    1/lambda_c*stacked_fx])
                if params_update_c['smooth_term']>0:
                    block3 = np.hstack([np.zeros((len(F), D.shape[1])), 
                                        1/lambda_c*params_update_c['smooth_term']*np.eye(len(F))])
                    stacked_fx_full = np.vstack([block1, block2, block3])
                else:
                    stacked_fx_full = np.vstack([block1, block2])
                    
            else: # namely t == 0
                """
                find total_next_dyn_full
                """
                total_next_dyn_full = data_i[:,time_point].reshape((-1,1))
                
                """
                find stacked_fx_full 
                """
                
                stacked_fx_full = 1/lambda_x*D
                                
            
            
            #reg = 1
            reg = params_update_c['reg_term']
        else:
            raise ValueError('future implementation')
    else: 
        reg = params_update_c['reg_term']
        if not several_dyns:
          cur_dyn = clear_dyn[:,time_point]
          next_dyn = latent_dyn[:,time_point+1]
          """
          define left side
          """
          if include_identity:
              total_next_dyn = next_dyn-cur_dyn
          else:
              total_next_dyn = next_dyn
              
          """
          right side
          """
          if not given_stacked_fx:
              f_x_mat = []
              for f_i in F:
                  f_x_mat.append(f_i @ cur_dyn)
              stacked_fx = np.vstack(f_x_mat).T 
              stacked_fx[stacked_fx> 10**8] = 10**8
          else: 
              stacked_fx = stacked_fx_list[time_point]
        else: 
          total_next_dyn = []
          for dyn_num in range(len(latent_dyn)):
            cur_dyn = clear_dyn[dyn_num][:,time_point]
            next_dyn = latent_dyn[dyn_num][:,time_point+1]
            if include_identity:
                total_next_dyn.extend((next_dyn-cur_dyn).flatten().tolist())
            else:
                total_next_dyn.extend(next_dyn.flatten().tolist())
            if not given_stacked_fx:
                f_x_mat = []
                for f_num,f_i in enumerate(F):
                  f_x_mat.append(f_i @ cur_dyn)
                if dyn_num == 0:
                  stacked_fx = np.vstack(f_x_mat).T 
                else:
                  stacked_fx = np.vstack([stacked_fx, np.vstack(f_x_mat).T ])
                stacked_fx[stacked_fx> 10**8] = 10**8
            else:
                stacked_fx = stacked_fx_list[time_point]
        
          total_next_dyn = np.reshape(np.array(total_next_dyn), (-1,1))
        if len(F) == 1: stacked_fx = np.reshape(stacked_fx,[-1,1])
        if params_update_c['smooth_term'] > 0 and time_point > 0 :
            if len(coefficients) == 0:
                #warnings.warn("Warning: you called the smoothing option without defining coefficients")
                pass
                
        """
        apply smoothness
        """
        if params_update_c['smooth_term'] > 0 and time_point > 0 and len(coefficients) > 0 :
            c_former = coeffs_list[-1].reshape((-1,1))
    
            total_next_dyn_full = np.vstack([total_next_dyn.reshape((-1,1)), np.sqrt(params_update_c['smooth_term'])*c_former])
            stacked_fx_full = np.vstack([stacked_fx, np.sqrt(params_update_c['smooth_term'])*np.eye(stacked_fx.shape[1])])
        else:
            total_next_dyn_full = total_next_dyn
            stacked_fx_full = stacked_fx   

    """
    cases
    """
    if (params_update_c['update_c_type'] == 'inv' or params_update_c['reg_term'] == 0):
      stacked_fx_full[np.isnan(stacked_fx_full)] = np.nanmean(stacked_fx_full)
      try:
          coeffs = linalg.pinv(stacked_fx_full) @ total_next_dyn_full.reshape((-1,1))
      except:
          print('stacked_fx_full')
          print(stacked_fx_full)
          print('total_next_dyn_full')
          print(total_next_dyn_full)
          
          if not skip_error:
              raise NameError('A problem in taking the inverse of fx when looking for the model coefficients')
          else:
              return 0*np.ones((len(F), latent_dyn.shape[1]))
          
    elif params_update_c['update_c_type'] == 'lasso' :

      clf = linear_model.Lasso(alpha=reg,random_state=random_state, **other_params)
      clf.fit(stacked_fx_full,total_next_dyn_full.flatten() )     
      coeffs = np.array(clf.coef_)

    elif params_update_c['update_c_type'].lower() == 'fista' :
        Aop = pylops.MatrixMult(stacked_fx_full)

        if 'threshkind' not in params_update_c: params_update_c['threshkind'] ='soft'

        coeffs = pylops.optimization.sparsity.FISTA(Aop, total_next_dyn_full.flatten(), niter=params_update_c['num_iters'],eps = reg , threshkind =  params_update_c.get('threshkind') )[0]

    elif params_update_c['update_c_type'].lower() == 'ista' :
        print('ista')
        
        if 'threshkind' not in params_update_c: params_update_c['threshkind'] ='soft'
        Aop = pylops.MatrixMult(stacked_fx_full)
        coeffs = pylops.optimization.sparsity.ISTA(Aop, total_next_dyn_full.flatten(), niter=params_update_c['num_iters'] , 
                                                   eps = reg,threshkind =  params_update_c.get('threshkind'))[0]
   
        
        
    elif params_update_c['update_c_type'].lower() == 'omp' :

        

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=int(np.ceil(reg))) 
        # Set k as the desired number of non-zero elements in x
        
        # Fit the OMP model
        omp.fit(stacked_fx_full,  total_next_dyn_full.flatten())
        
        # Get the estimated sparse coefficients
        coeffs = omp.coef_
     
        ss = ((stacked_fx_full @ coeffs  - total_next_dyn_full)**2).sum()
        print('er here: %s'%str(ss))
        
    elif params_update_c['update_c_type'].lower() == 'spgl1' :

        
        Aop = pylops.MatrixMult(stacked_fx_full)

        coeffs = pylops.optimization.sparsity.SPGL1(Aop, total_next_dyn_full.flatten(),iter_lim = params_update_c['num_iters'],
                                                   tau = reg, verbosity = False)[0]
        
        
    elif params_update_c['update_c_type'].lower() == 'irls' :
        print('irls')
        Aop = pylops.MatrixMult(stacked_fx_full)
        
    
        coeffs = pylops.optimization.sparsity.IRLS(Aop, total_next_dyn_full.flatten(),  nouter=50, 
                                                   espI = reg)[0]

        
        
    else:
        
        
      raise NameError('Unknown update c type')
    if x_together:
        if time_point == 0:
            latent_dyn[:,time_point] = coeffs.flatten()/lambda_x
            
            coeffs = np.zeros((len(F), 1))
        else:
            latent_dyn_local = coeffs[:latent_dyn.shape[0]].flatten()/lambda_x
            if np.abs(latent_dyn_local).sum() <= 10e-5:
                latent_dyn_local = np.linalg.pinv(D) @ data_i[:,time_point].reshape((-1,1))
            
            latent_dyn[:,time_point] = latent_dyn_local.flatten()
            coeffs = coeffs[latent_dyn.shape[0]:].flatten() 
            
         
    coeffs_list.append(coeffs.flatten())
  coeffs_final = np.vstack(coeffs_list)
  if x_together:
      coeffs_final =coeffs_final[1:,:]
      
      return coeffs_final.T, latent_dyn
          
  return coeffs_final.T


def create_next(latent_dyn, coefficients, F,time_point = -1, order = 1):
  """
  This function evaluate the dynamics at t+1 given the value of the dynamics at time t, the sub-dynamics, and other model parameters
  Inputs:
      latent_dyn    = the latent dynamics (can be either ground truth or estimated). [k X T]
      coefficients  = the sub-dynamics coefficients (used by the model)
      F             = a list of np.arrays, each np.array is a sub-dynamic with size kXk
      time_point    = current time point
      order         = how many time points in the future we want to estimate
  Outputs:
      k X 1 np.array describing the dynamics at time_point+1
  order 1 = only x_(t+1) is predicted using x_t. if order = k, x_(t+k) is predicted using x_t
  """

  if isinstance(F[0],list):
    F = [np.array(f_i) for f_i in F]

  if not  check_1d(latent_dyn) and latent_dyn.shape[1] > 1:
      if time_point < 0:
          raise ValueError('time point must be given is latent dyn is a matrix')
      cur_A = np.dstack([coefficients[i,time_point]*f_i @ latent_dyn[:, time_point] 
                             for i,f_i in enumerate(F)]).sum(2).T   
  else:
      if  check_1d(latent_dyn):
          latent_dyn = latent_dyn.reshape((-1,1))
      if not check_1d(coefficients):
          raise ValueError('coefficients cannot be a matrix in this case')

      cur_A  = np.dstack([coefficients[i]*f_i @ latent_dyn for i,f_i in enumerate(F)]).sum(2).T 

  if order > 1:
      raise ValueError('need to change!')

  return cur_A

def missing_samples(dynamic, missing_locs = 10, seed = 0):
    """
    Remove samples from a dynamic matrix at specified locations.
    
    Args:
        dynamic (numpy.ndarray): The dynamic matrix.
        missing_locs (int or list or numpy.ndarray, optional): The locations to remove samples from.
                                                              If an integer is provided, random locations will be chosen (default: 10).
        seed (int, optional): The seed for random number generation when selecting random locations (default: 0).
    
    Returns:
        numpy.ndarray: The dynamic matrix with samples removed.
    
    """
    if not isinstance(missing_locs, (list, np.ndarray)):
        np.random.seed(seed)
        missing_locs = np.random.choice(np.arange(dynamic.shape[1]-1)+1,missing_locs, replace=False)
    missing_locs = np.sort(missing_locs)[::-1]
    dynamic_without = dynamic.copy()
    for miss_loc in missing_locs:
        dynamic_without = np.hstack([dynamic_without[:,:miss_loc], dynamic_without[:,miss_loc+1:]]  )
    return dynamic_without
        
        
        
def find_coeff_for_multi(data, F, params_update_c = {}, to_update_c = True, return_dstack = False, type_coeffs_shared = 'opt',
                         coefficients = [],include_identity = False):
    """
    Parameters
    ----------
    data : list
        list of data samples. k X T
    F : list of kXk np.arrays
        sub-dynamics.

    Returns
    -------
    stacked_fx
    """
    if not (not return_dstack and type_coeffs_shared == 'mean'):
        
        params_update_c = {**{'update_c_type':'inv','reg_term':0,'smooth_term':0, 'to_norm_fx' : False}, **params_update_c}
        data_sum = np.sum(np.dstack(data), axis = 2)
        # Each element in the list is a time point
        list_of_stacked_fx = [np.sum(np.dstack([np.hstack([(f @ data_i[:,t]).reshape((-1,1)) for f in F]) for data_i in data]),2) 
                              for t in range(data[0].shape[1])]
        if return_dstack:
            list_of_fx = [np.dstack([np.hstack([(f @ data_i[:,t]).reshape((-1,1)) for f in F]) for data_i in data]) 
                                  for t in range(data[0].shape[1])]
   
    if to_update_c:
        if type_coeffs_shared == 'mean':
            if len(coefficients) == 0: raise ValueError('Coefficients is empty. You should provide coefficients as input if type_coeffs_shared is "mean"')
            else: coeffs = np.mean(np.dstack(coefficients),2)
        elif type_coeffs_shared == 'opt':
            coeffs = update_c(F, data_sum, 
                              params_update_c = params_update_c, given_stacked_fx = True,    stacked_fx_list  = list_of_stacked_fx,
                              include_identity = include_identity )
        else:
            raise NameError('Unknown type_coeffs_shared')
        if return_dstack:
            return coeffs, list_of_fx
        return coeffs
    else:
        if return_dstack:
            return list_of_stacked_fx, list_of_fx
        return list_of_stacked_fx
    


def update_f_all(latent_dyn,F,coefficients, step_f, normalize = False, acumulated_error = False,error_order = 1,
                 action_along_time = 'mean',
                 weights_power = 1.2, weights = [], normalize_eig = True, 
                 bias_val = [], include_identity = False,  type_norm = 'evals' ,dur_update = 3, min_T_update = 0):
    
    """
    Update all the sub-dynamics {f_i} using gradient descent (GD).

    Args:
        latent_dyn (numpy.ndarray): The latent dynamics.
        F (list): List of matrices representing the sub-dynamics {f_i}.
        coefficients (numpy.ndarray): Coefficient matrix.
        step_f (float): Step size for the GD update.
        normalize (bool, optional): Indicates whether to normalize the updated sub-dynamics (default: False).
        acumulated_error (bool, optional): Indicates whether to compute the cumulative error (default: False).
        error_order (int, optional): The error order for computing the cumulative error (default: 1).
        action_along_time (str, optional): The action to take along the time dimension:
                                           - 'mean': Compute the mean of the gradients (default).
                                           - 'median': Compute the median of the gradients.
        weights_power (float, optional): The power for the weights (default: 1.2).
        weights (list, optional): List of weight matrices.
        normalize_eig (bool, optional): Indicates whether to normalize the eigenvalues (default: True).
        bias_val (list, optional): List of bias values for each sub-dynamic (default: []).
        include_identity (bool, optional): Indicates whether to include an identity matrix in the gradient computation (default: False).
        type_norm (str, optional): The type of normalization to apply:
                                   - 'evals': Normalize using eigenvalues (default).
                                   - 'max': Normalize by dividing by the maximum absolute value.

    Returns:
        list: List of updated sub-dynamics {f_i}.

    Raises:
        NameError: If the action_along_time parameter is not 'mean' or 'median'.

    """
  
    if len(bias_val) == 0:
        bias_val = np.zeros((latent_dyn.shape[0], 1))
        
    if action_along_time == 'mean':
      
      if acumulated_error:
        all_grads = create_ci_fi_xt(latent_dyn,F,coefficients, cumulative = acumulated_error, error_order = error_order, 
                                    weights_power=weights_power,weights =weights, bias_val = bias_val,
                                    include_identity = include_identity )
        new_f_s = [norm_mat(f_i-2*step_f*norm_mat_by_max(np.mean(all_grads[:,:,:]*np.reshape(coefficients[i,:], [1,1,-1]), 2),
                                                         to_norm = normalize),to_norm = normalize_eig , type_norm = type_norm) for i,f_i in enumerate(F)] 
      
      else:
        all_grads = create_ci_fi_xt(latent_dyn,F,coefficients,error_order = error_order, weights_power=weights_power,
                                    weights =weights, bias_val = bias_val, include_identity = include_identity )
        new_f_s = [norm_mat(f_i-2*step_f*norm_mat_by_max(np.mean(all_grads[:,:,:]*np.reshape(coefficients[i,:], [1,1,-1]), 2),to_norm = normalize),to_norm = normalize_eig ) for i,f_i in enumerate(F)] 
    elif action_along_time == 'median':
        
      if acumulated_error:
        all_grads = create_ci_fi_xt(latent_dyn,F,coefficients, cumulative = acumulated_error, error_order = error_order, 
                                    weights_power=weights_power,weights =weights, bias_val = bias_val,  include_identity = include_identity )
        new_f_s = [norm_mat(f_i-2*step_f*norm_mat_by_max(np.median(all_grads[:,:,:]*np.reshape(coefficients[i,:], [1,1,-1]), 2),to_norm = normalize),
                            to_norm = normalize_eig ,  type_norm = type_norm) for i,f_i in enumerate(F)] 

      
      else:

        
        all_grads = create_ci_fi_xt(latent_dyn,F,coefficients,error_order = error_order,
                                    weights_power=weights_power,
                                    weights =weights, bias_val = bias_val,  include_identity = include_identity )
        
        max_T_update = all_grads.shape[2] - dur_update      
        rand_start = np.random.randint(0, max_T_update - 1)
        rand_end = rand_start  + dur_update
        all_grads = all_grads[:,:,rand_start:rand_end] 

        
        
        
        new_f_s = [norm_mat(f_i-2*step_f*norm_mat_by_max(np.mean(all_grads*np.reshape(coefficients[i,rand_start:rand_end ], [1,1,-1]), 2),
                                                         to_norm = normalize),to_norm = normalize_eig ,  type_norm = type_norm ) 
                   for i,f_i in enumerate(F)] 

                
                
                
    else:
      raise NameError('Unknown action along time. Should be mean or median')
    for f_num in range(len(new_f_s)):
        rand_mat = np.random.rand(new_f_s[f_num].shape[0],new_f_s[f_num].shape[1])
        new_f_s[f_num][np.isnan(new_f_s[f_num])] = rand_mat[np.isnan(new_f_s[f_num])] .flatten()
        
    return new_f_s

def create_ci_fi_xt(latent_dyn,F,coefficients, cumulative = False,error_order = 1, weights_power = 1.2, 
                    weights = [], mute_infs = 10**50, max_inf = 10**60, bias_val = [], include_identity = False):
    
  """
  An intermediate step for the reconstruction -
  Specifically - It calculated the error that should be taken in the GD step for updating f: 
  f - eta * output_of(create_ci_fi_xt)
  output: 
      3d array of the gradient step (unweighted): [k X k X time]
  """
  if len(bias_val) == 0: bias_val = np.zeros((latent_dyn.shape[0], 1))
  if max_inf <= mute_infs:
    raise ValueError('max_inf should be higher than mute-infs')
    
  if  error_order > 1:
    curse_dynamics = latent_dyn
    list_dyns = [curse_dynamics]; order_list = [1]

    for i in range(error_order):
      curse_dynamics =create_reco(curse_dynamics, coefficients, F) 
      curse_dynamics[curse_dynamics > max_inf] = max_inf
      curse_dynamics[curse_dynamics < -max_inf] = -max_inf
      list_dyns.append(curse_dynamics); order_list.append(i+2)

    if len(weights) == 0:
      weights = np.array(order_list)[::-1]**weights_power/np.sum(np.array(order_list)**weights_power)
    if mute_infs > 0:
      to_mute = np.array([np.median((list_dyn-latent_dyn)**2) > mute_infs for list_dyn in list_dyns]      )
    else:
      to_mute = np.array([False] * len(list_dyns))
    if (to_mute == False).any():
    
      weights[to_mute] = 0

    else:
      mute_vals = np.array([np.median((list_dyn-latent_dyn)**2)  for list_dyn in list_dyns]      )
      weights[mute_vals > np.min(mute_vals)] = 0
  
    weights = weights / np.sum(weights)
    curse_dynamics = np.average(np.dstack(list_dyns), axis = 2, weights = np.array(order_list)[::-1]/np.sum(np.array(order_list)**2))
  else:
    curse_dynamics = latent_dyn

  all_grads = []
  for time_point in np.arange(latent_dyn.shape[1]-1):
    if cumulative:
      if time_point > 0:
        previous_A = cur_A
      else:
        previous_A = curse_dynamics[:,0]
      if include_identity:
          cur_A = create_next(np.reshape(previous_A,[-1,1]), coefficients, F,time_point) + previous_A.reshape((-1,1)) # the prediction for the next 
      else:
          cur_A = create_next(np.reshape(previous_A,[-1,1]), coefficients, F,time_point) # the prediction for the next 
    else:
        if include_identity:
            cur_A = create_next(curse_dynamics, coefficients, F,time_point) + curse_dynamics[:,time_point]
        else:
            cur_A = create_next(curse_dynamics, coefficients, F,time_point)
    next_A = latent_dyn[:,time_point+1]
    
    """
    The actual step
    """

    if np.sum(bias_val) !=0:
        cur_A = cur_A + bias_val
    if cumulative:
      gradient_val = -(next_A - cur_A) @ previous_A.T
    else:

        gradient_val = -(next_A.flatten() - cur_A.flatten()).reshape((-1,1)) @ curse_dynamics[:, time_point].reshape((1,-1))
    all_grads.append(gradient_val)
  all_grads = np.dstack(all_grads)

  return all_grads

   






      
    
    

    
def build_reco_again(init_cond, fs, cs):
    """
     Build a reconstruction based on initial conditions, frequency spectra, and coefficients.
     
     Args:
         init_cond (list): Initial conditions for the reconstruction.
         fs (list): List of frequency spectra.
         cs (numpy.ndarray): Coefficients array.
     
     Returns:
         numpy.ndarray: Reconstructed array.
     
    """
    rec = [np.array(init_cond).reshape((-1,1))] 
    for t in range(cs.shape[1]):
        rec.append( np.sum(np.dstack([f_spec*cs[i,t] for i, f_spec in enumerate(fs)  ]) ,2) @ rec[-1])
    rec = np.hstack(rec)
    return rec    
  
    


def create_rotation_mat(theta = 0, axes = 'x', dims = 3):
    """
    Create a rotation matrix for a specified angle and axes.
    
    Args:
        theta (float, optional): The rotation angle in radians (default: 0).
        axes (str, optional): The axes of rotation:
                              - 'x': Rotation around the x-axis (default).
                              - 'y': Rotation around the y-axis.
                              - 'z': Rotation around the z-axis.
        dims (int, optional): The number of dimensions for the rotation matrix:
                              - 3: 3D rotation matrix (default).
                              - 2: 2D rotation matrix.
    
    Returns:
        numpy.ndarray: The rotation matrix.
    
    Raises:
        ValueError: If dims is not 2 or 3.
    
    """
    if dims == 3:
        if axes.lower() == 'x':
            rot_mat = np.array([[1,0,0],
                                [0,np.cos(theta), -np.sin(theta)], 
                                [0, np.sin(theta), np.cos(theta)]])
        elif axes.lower() == 'y':
            rot_mat = np.array([[np.cos(theta),0,np.sin(theta)],
                                [0,1, 0], 
                                [-np.sin(theta),0, np.cos(theta)]])
        elif  axes.lower() == 'z':
            rot_mat = np.array([[np.cos(theta),-np.sin(theta),0],
                                [np.sin(theta),np.cos(theta), 0], 
                                [0, 0, 1]])
    elif dims == 2:
        if axes.lower() == 'x':
            rot_mat = np.array([[0,np.cos(theta), -np.sin(theta)], 
                                [0, np.sin(theta), np.cos(theta)]])
        elif axes.lower() == 'y':
            rot_mat = np.array([[np.cos(theta),0,np.sin(theta)],                            
                                [-np.sin(theta),0, np.cos(theta)]])
        else:
            raise ValueError('axes is invalid')
        
    else: 
        raise ValueError('dims should be 2 or 3')
    return rot_mat

def flip_power(x1,x2)        :
    """
    Compute the power of x2 raised to x1.
    
    Args:
        x1 (float): The exponent.
        x2 (float): The base.
    
    Returns:
        float: The result of x2 raised to the power of x1.
    
    """
    return np.power(x2,x1)
    

        
def sigmoid(x, std = 1):
    """
    Compute the sigmoid function.
    
    Args:
        x (float or numpy.ndarray): Input value(s) to the sigmoid function.
        std (float, optional): Standard deviation parameter (default: 1).
    
    Returns:
        float or numpy.ndarray: Sigmoid function output(s).
    
    """
    return 1 / (1 + np.exp(-x/std))        


  
def plot_effect_of_one_dyn(f, c_basis = 0.99, len_t = 1000, to_return = True, init = 50, to_plot = True, x0 = []):
    """
    Plot the effect of a single dynamic on the initial condition.
    
    Args:
        f (numpy.ndarray): Dynamic matrix.
        c_basis (float, optional): Coefficient basis (default: 0.99).
        len_t (int, optional): Length of time to compute (default: 1000).
        to_return (bool, optional): Whether to return the computed trajectory (default: True).
        init (int, optional): Index to start plotting the trajectory (default: 50).
        to_plot (bool, optional): Whether to plot the trajectory (default: True).
        x0 (numpy.ndarray, optional): Initial condition (default: empty list).
    
    Returns:
        numpy.ndarray or None: Computed trajectory if `to_return` is True, else None.
    
    """
    if checkEmptyList(x0):
        if f.shape[0] == 3: x0 = np.array([1,-1,1]).reshape((-1,1))
        elif f.shape[0] == 2:  x0 = np.array([1,-1]).reshape((-1,1))
    if x0.shape[1] > 1: x0 = x0[:,0].reshape((-1,1))
    for i in range(len_t):
        x0 = np.hstack([x0, c_basis*f @ x0[:,-1].reshape((-1,1))])
        
    if to_plot:
        visualize_dyn(x0[:,init:], remove_back = False)
    if to_return:
        return x0
    
def checkEmptyList(obj):
    """
    Check if the given object is an empty list.

    Args:
        obj (object): Object to be checked.

    Returns:
        bool: True if the object is an empty list, False otherwise.

    """    
    return isinstance(obj, list) and len(obj) == 0


def create_dynamics(type_dyn = 'cyl', max_time = 1000, dt = 0.01, change_speed = False, t_speed = np.exp, 
                    axis_speed = [], t_speed_params = {}, to_cent = False,return_3d = False, return_additional = False,
                    params_ex = {}, add_I = False,  initial_conds = (0., 1., 1.05), direction = 'reg', addition = '', single_session = -1, 
                    all_regs_together = False):
  """
  
  Create ground truth dynamics
  dyn_type options:
      cyl
      f_spiral
      df_spiral
      
  """
  params_ex = { **{'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2,'theta':0, 'orientation_ax':'y', 'type_theta':'rotate',
                  'type_theta':'rotate','phi': np.pi/50, 'c_type_for_combined':'sig',
                                'x0':np.array([1,-1]).reshape((-1,1)), 'c_control' :0.999, 'dim':2}, **params_ex}  
  dim = params_ex['dim']
  if t_speed == np.power: 
      t_speed_params = {**{'pow':2}, **t_speed_params}
      t_speed = flip_power
  else:
      t_speed_params = {}

  t = np.arange(0, max_time, dt)

  if type_dyn == 'cyl':
    x = params_ex['radius']*np.sin(t)
    y = params_ex['radius']*np.cos(t)
    z = t     + params_ex['bias']

    if change_speed: 
      t_speed_vec = t_speed(params_ex['exp_power']*t, t_speed_params.get('pow'))

      if 0 in axis_speed: x = np.sin(t_speed_vec)
      if 1 in axis_speed: y = np.cos(t_speed_vec)
      if 2 in axis_speed: z = t_speed_vec
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
    
  elif type_dyn == 'comb_spirals':
    x0 = params_ex['x0']
    phi = params_ex['phi']
    c_control_joint = params_ex['c_control']
    if len(x0) == 2: x0 = np.vstack([x0.reshape((-1,1)), [-1]])
    if params_ex['c_type_for_combined'] == 'sig':
        sig_e = 5
        c_control = np.vstack([sigmoid(np.linspace(-sig_e ,sig_e ,len(t)), 2), (1-sigmoid(np.linspace(-sig_e ,sig_e , len(t)), 2))])
  

    else:
        if not isinstance(c_control, (list,np.ndarray, tuple)):
            c_control = [c_control, c_control]
    f1_basis = np.array([[np.cos(phi),np.sin(phi),0],[-np.sin(phi),np.cos(phi),0],[0,0,0.1]])
    f1 = f1_basis.copy()  
    f2 = np.array([[np.cos(phi),0,np.sin(phi)],
                        [0,0.1, 0], 
                        [-np.sin(phi),0, np.cos(phi)]]) 

    for i in range(len(t)):
        cur_mat = (c_control[0,i]*f1 +c_control[1,i]*f2)
        eigenvalues, eigenvectors =  linalg.eig(cur_mat)
        cur_mat = cur_mat / np.max(np.abs(eigenvalues))
        c_control[:,i] = c_control[:,i]/ np.max(np.abs(eigenvalues))
        x0 = np.hstack([x0, c_control_joint*cur_mat @ x0[:,-1].reshape((-1,1))])
    x = x0[0,:]
    y = x0[1,:]
    z = x0[2,:] 
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()])
    
  elif type_dyn == 'spiral' or  type_dyn == 'f_spiral' or  type_dyn == 'df_spiral' :
    """
    spiral = 1d spiral
    f_spiral = flat spiral
    df spiral = spiral in and out
    
    """
 
    x0 = params_ex['x0']
    phi = params_ex['phi']
    c_control = params_ex['c_control']
    
    f1 = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
    for i in range(len(t)-1):
        if add_I:
            x0 = np.hstack([x0,(np.eye(2) + c_control*f1) @ x0[:,-1].reshape((-1,1))])
        else:
            x0 = np.hstack([x0,c_control*f1 @ x0[:,-1].reshape((-1,1))])
    x = x0[0,:]
    y = x0[1,:]
    if dim == 3 or type_dyn == 'spiral':
        if dim < 3: print('Pay attention! if you want a 2d spiral type "f_spiral" as the dynamic type (not "spiral")')
        if type_dyn == 'spiral':           z = t         
        elif type_dyn == 'f_spiral': z = np.zeros(t.shape)
        elif type_dyn == 'df_spiral': z = np.zeros(t.shape)
    
    if change_speed: 
      t_speed_vec = t_speed(params_ex['exp_power']*t, t_speed_params.get('pow'))
      if 0 in axis_speed: x = t_speed_vec * np.sin(t_speed_vec)
      if 1 in axis_speed: y = t_speed_vec * np.cos(t_speed_vec)
      if 2 in axis_speed and dim == 3: z = t_speed_vec
    if dim == 3:
        dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
    else:
        dynamics = np.vstack([x.flatten(),y.flatten()])
    if type_dyn == 'df_spiral':
        dynamics = np.hstack([dynamics[:,::-1], dynamics])
        if not type_dyn == 'spiral' and not return_3d:
            dynamics = dynamics[:2,:]
  elif type_dyn == 'lorenz':    

    dynamics1, real_jac, _,fs =  create_lorenz_mat(np.max(t), initial_conds, alpha = 10 , beta  = 25 , gamma = 2.67, dt = dt, direction = direction)

  elif type_dyn == 'torus':
    R=5;    r=1;
    u=np.arange(0,max_time,dt);
    v=np.arange(0,max_time,dt);
    [u,v]=np.meshgrid(u,v);
    x=(R+r*np.cos(v)) @ np.cos(u);
    y=(R+r*np.cos(v)) @ np.sin(u);
    z=r*np.sin(v);
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
  elif type_dyn == 'circ2d':
    x = params_ex['radius']*np.sin(t)
    y = params_ex['radius']*np.cos(t)
    dynamics = np.vstack([x.flatten(),y.flatten()]) 
  elif type_dyn == 'trans':
      dynamics,_,_,_ = create_smooth_trans(max_time = max_time, dt = dt, sig_e = 5)
  elif type_dyn == 'duff':

      dynamics0, real_jac , masks, offset_full, real_fs = create_duffing(t_val = max_time, initial_conds = initial_conds, dt = dt,  direction = 'reg', 
                                            gamma = 0.02, delta = 0.001, alpha = 1, beta = 100,    w = 50)
      dynamics = [dynamics0, real_jac, masks, offset_full, real_fs]
      
      
      
      
  elif type_dyn == 'multi_cyl':
    dynamics0_str = create_dynamics('cyl',max_time = max_time ,dt = dt, params_ex = params_ex)
    dynamics0_str = dynamics0_str - dynamics0_str[:,0].reshape((-1,1))
    dynamics0_inv = dynamics0_str[:,::-1]
    dynamics0     = np.hstack([dynamics0_str, dynamics0_inv ])
    
    list_dyns = []
    for dyn_num in range(params_ex['num_cyls']):
        np.random.seed(dyn_num)
        random_trans = np.random.rand(dynamics0.shape[0],dynamics0.shape[0])-0.5
        transformed_dyn = random_trans @ dynamics0
        list_dyns.append(transformed_dyn)
    dynamics = np.hstack(list_dyns)
  elif type_dyn == 'c_elegans':
      mat_c_elegans = load_mat_file('WT_NoStim.mat','E:\CoDyS-Python-rep-\other_models') # 
      dynamics = mat_c_elegans['WT_NoStim']['traces'].T
  elif type_dyn == 'lorenz_2d':
    txy = t
    if change_speed: 

      t_speed_vec = t**params_ex['exp_power']
      if (0 and 1) in axis_speed: txy = t_speed_vec      
      if 2 in axis_speed: txy = t_speed_vec
    x,y,z  = create_lorenz_mat(t, txy = txy)
    dynamics = np.vstack([x.flatten(),z.flatten()]) 
    
  elif type_dyn.lower() == 'fhn':
    v_full, w_full = create_FHN(dt = dt, max_t = max_time, I_ext = 0.5, b = 0.7, a = 0.8 , tau = 20, v0 = -0.5, 
                                w0 = 0, params = {'exp_power' : params_ex['exp_power'], 'change_speed': change_speed}) 
    dynamics = np.vstack([v_full, w_full])
  elif type_dyn.lower()   == 'monkey_trial':
      dynamics = list(np.load('monkey_data_22_5_82_5.npy',allow_pickle = True))
      dynamics =  dynamics[0].T
  elif type_dyn.lower()   == 'monkey_trials':
      dynamics = list(np.load('monkey_data_22_5_82_5.npy',allow_pickle = True))
      dynamics = [dyn.T for dyn in dynamics]  

  elif type_dyn.lower()   == 'eeg_trial': # one 
      """
      EEG data
      """
      dynamics = red_mean(np.load('EEG_trial_control.npy',allow_pickle = True))

  elif type_dyn.lower()   == 'eeg_circle': # one circle, one trial
      dynamics = list(np.load('EEG_circle_control.npy',allow_pickle = True))
      dynamics = [red_mean(dyn) for dyn in dynamics]

  elif type_dyn.lower()   == 'eeg_circles': # 
      dynamics = list(np.load('EEG_circles_control.npy',allow_pickle = True))
      dynamics = [red_mean(dyn) for dyn in dynamics]
 
  elif type_dyn.lower()   == 'eeg_conditions': 
      dynamics = list(np.load('EEG_conditions.npy',allow_pickle = True))
      dynamics = [red_mean(dyn) for dyn in dynamics]

      
      
  if params_ex['theta'] > 0:
      if params_ex['type_theta'] == 'rotate':
          rot_mat = create_rotation_mat(theta = params_ex['theta'], axes = params_ex['orientation_ax'], dim = dynamics.shape[0])
          dynamics = rot_mat @ dynamics
      elif params_ex['type_theta'] == 'shift':
          if params_ex['orientation_ax'] == 'x':       dynamics =  dynamics+np.array([params_ex['theta'],0, 0]).reshape((-1,1))
          if params_ex['orientation_ax'] == 'y':       dynamics =  dynamics+np.array([0,params_ex['theta'], 0]).reshape((-1,1))
          if params_ex['orientation_ax'] == 'z':       dynamics =  dynamics+np.array([0,0,params_ex['theta']]).reshape((-1,1))
  if to_cent:
      dynamics = dynamics - np.mean(dynamics,1).reshape((-1,1))
      
  if type_dyn == 'comb_spirals' and return_additional:
     return dynamics, f1, f2, c_control
  if (type_dyn == 'f_spiral' or type_dyn == 'df_spiral') and return_additional:
     return  dynamics, f1, c_control
  if type_dyn == 'epi_log' :
      dynamics = np.load('vals_logs_source_sink.npy')
  elif type_dyn == 'epi_amir' or type_dyn == 'epi_amir_normalize':
      file_name = 'GD_1_SI'
      rr = load_mat_file(file_name)  
      dynamics = 10*rr['SI3_wins']
      if type_dyn =='epi_amir_normalize'  :
          dynamics = dynamics / ((dynamics**2).sum(1)**0.5).reshape((-1,1))
  elif type_dyn =='epi_amir_short'  :

    dynamics_full = np.load( 'save_epi_full.npy', allow_pickle = True).item()['neural_data']

    keys = list(dynamics_full.keys())
    dynamics = [10*dynamics_full[key] for key in keys]
    return dynamics, keys


  elif type_dyn == 'multi_reg':
      dynamics = np.load('data_ordered_info_dict.npy', allow_pickle = True).item()
  elif  type_dyn == 'hippocampus':
      dynamics = np.load('hippocampus_data_tcALLSEUDO.npy') #, allow_pickle = True).item()
  elif  type_dyn == 'hippocampus2':
      dynamics = np.load('hippocampus2') 
  elif  type_dyn == 'hippocampus2_avg':


      dynamics = np.load('hippocampus2_avg.npy')
  elif type_dyn == 'test':
      l = np.load('test_multi_trial.npy', allow_pickle=True).item()
      dynamics  = list(l.values())
  
  elif type_dyn.startswith('multi_reg_neuron'):

      dynamics = np.load('data_ordered_info_dict_neuronsB_%s.npy'%addition, allow_pickle = True).item()
  
  
 
  elif type_dyn.startswith('multi_reg_meso_large'):

      dynamics = np.load('meso_multi_reg_to_run.npy', allow_pickle = True).item()
     
      
  elif type_dyn.startswith( 'multi_reg_meso_small'):

      dynamics = np.load('meso_multi_reg_to_run_small.npy', allow_pickle = True).item()
      
            
  elif type_dyn.startswith('multi_reg_meso'): # i.e. specif number_ of trials

      spl = type_dyn.split('_')
      num_trials = int(spl[-2])
      num_files = int(spl[-1])
      file_load = r'mseo_reg_feb_2024_2_n_files_%d_n_trials_%d_2024_02_11.npy'%(num_files, num_trials)

      dynamics = np.load(file_load, allow_pickle = True).item()
      
  elif type_dyn.startswith('multi_reg_human_subject10'): # i.e. specif number_ of trials


      if 'small' in type_dyn.lower():
          file_load =  r'human_apr_apr19_2_human_data_subject_10_dandi_000469_2024_04_19_small.npy'
      else:
          file_load = r'human_apr_apr19_2_human_data_subject_10_dandi_000469_2024_04_19.npy'

      dynamics = np.load(file_load, allow_pickle = True).item()    
  


  elif type_dyn.startswith('multi_reg_human_all'): # i.e. specif number_ of trials
    

        if 'part' in  type_dyn.lower():
            if '50' in type_dyn.lower():

                raise ValueError('unavaialbe')
            elif '30' in type_dyn.lower():
                
                if 'small' in type_dyn.lower():
                    file_load = 'human_apr_apr19_2_human_all_sub_id_000469_2024_04_22_0.030_onlytwoFalse_part_True_small.npy'

                else:
                    file_load = 'human_apr_apr19_2_human_all_sub_id_000469_2024_04_22_0.030_onlytwoFalse_part_True.npy'
        elif 'only_two' in  type_dyn.lower():
            if '50' in type_dyn.lower():

                raise ValueError('unavaialbe')
            elif '30' in type_dyn.lower():
                file_load = 'human_apr_apr19_2_human_all_sub_id_000469_2024_04_22_0.030_onlytwoTrue.npy'

        else:
            if '50' in type_dyn.lower():

                raise ValueError('unavailable')
            elif '30' in type_dyn.lower():

                file_load = 'human_apr_apr19_2_human_all_sub_id_000469_2024_04_25_0.030_onlytwoFalse_part_False.npy'

        dynamics = np.load(file_load, allow_pickle = True).item()   
        if ('part' not in type_dyn) and  'small' in type_dyn.lower() and ('30' in type_dyn.lower() or '50' in type_dyn.lower()):
            dynamics['data_active'] = {key:val[:,:150] for key,val in dynamics['data_active'].items()}
            
        elif ('part' not in type_dyn) and  'small' not in type_dyn.lower() and ('30' in type_dyn.lower() or '50' in type_dyn.lower()):
            dynamics['data_active'] = {key:val[:,:1000] for key,val in dynamics['data_active'].items()}
            
        if single_session > -1:
            sessions = list(dynamics['data_active'].keys())
            chosen_session = sessions[single_session]
            dynamics['chosen_session'] = chosen_session
            dynamics['data_active'] = {chosen_session : dynamics['data_active'][chosen_session]}
            dynamics['H_dict'] =  {chosen_session : dynamics['H_dict'][chosen_session]}
       
            dynamics['D_masks'] =   {chosen_session : dynamics['D_masks'][chosen_session]}
            dynamics['labels'] =   {chosen_session : dynamics['labels'][chosen_session]}
            or_num = dynamics['keys_to_num'][chosen_session]

            dynamics['or_key_new_key'] = {single_session: or_num}
            dynamics['keys_to_num'] =  {chosen_session : 0}
            dynamics['num_to_keys'] =  {0 :chosen_session }
            dynamics['indices_regs'] =  {chosen_session : dynamics['indices_regs'][chosen_session]}
            
        elif single_session == -2:
            sessions = list(dynamics['data_active'].keys())
            
            chosen_sessions = sessions[:8]
            dynamics['chosen_session'] = chosen_sessions
            dynamics['data_active'] = {chosen_session : dynamics['data_active'][chosen_session] for chosen_session in chosen_sessions}
            dynamics['H_dict'] =  {chosen_session : dynamics['H_dict'][chosen_session] for chosen_session in chosen_sessions}

            dynamics['D_masks'] =   {chosen_session : dynamics['D_masks'][chosen_session] for chosen_session in chosen_sessions}
            dynamics['labels'] =   {chosen_session : dynamics['labels'][chosen_session] for chosen_session in chosen_sessions}
            or_nums = [dynamics['keys_to_num'][chosen_session] for chosen_session in chosen_sessions] 
          
            dynamics['or_key_new_key'] = {j: or_num for j, or_num in enumerate(or_nums)}
            dynamics['keys_to_num'] =  {ses:j for j,ses in enumerate(chosen_sessions)}
            dynamics['num_to_keys'] =  {j:ses for j, ses in enumerate(chosen_sessions) }
            dynamics['indices_regs'] =  {chosen_session : dynamics['indices_regs'][chosen_session] for chosen_session in chosen_sessions}          


    
  elif type_dyn.startswith('synth_multi'):      
      """
      MULTIPLE ENSEMBLES CASE
      """
      
      if 'three_ensemble' in type_dyn:

          path_load = os.getcwd() + os.sep + 'synth_three_ensembles_four_regions' + os.sep + 'three_ensbmeles_four_areas_False_2024_05_15.npy'
          dynamics = np.load(path_load, allow_pickle = True).item()   
          if 'MATERIALS' in os.getcwd():
              dynamics['ys'] = {key:dyn[:,:100] for key,dyn in dynamics['ys'].items()}
          
      elif 'simplesimple' in type_dyn:
            path_GT = os.getcwd() + os.sep + 'synth_simple' + os.sep + 'synth_multi_simple_ground_truth_noise_False_2023-12-14.npy'
            dynamics = np.load(path_GT, allow_pickle = True).item()     
            if all_regs_together:
                dynamics['Ds_masks'] = {key:np.ones(D_mask.shape) for key, D_mask in dynamics['Ds_masks'].items()}
                dynamics['num_per_region_full'] = {key: np.array([np.sum(val)]) for key, val in dynamics['num_per_region_full'].items()}
            
      else:
          type_dyn_short = type_dyn.split('synth_multi_')[1]
          today = '2024_01_04'
          try:
              dynamics = np.load(os.getcwd() + os.sep +  r'synth_%s'%type_dyn_short + os.sep + r'synth_multi_high_ground_truth_noise_False_2024-01-04.npy', allow_pickle = True).item()#'syntehtic_simple_ground_truth.npy'     
          except:
              dynamics = np.load(os.getcwd() + os.sep +  r'synth_%s'%type_dyn_short + os.sep + r'synth_multi_high_ground_truth_noise_False_2024_01_04.npy', allow_pickle = True).item()#'syntehtic_simple_ground_truth.npy'     
      if single_session > -1:
        sessions = list(dynamics['ys'].keys())
        chosen_session = sessions[single_session]
        dynamics['chosen_session'] = chosen_session
        dynamics['cs'] = {chosen_session : dynamics['cs'][chosen_session]}

        dynamics['Ds_masks'] =   {chosen_session : dynamics['Ds_masks'][chosen_session]}
        dynamics['D_masks'] = dynamics['Ds_masks']
        dynamics['labels'] =   {chosen_session : dynamics['labels'][chosen_session]}
        
        or_num = chosen_session
        dynamics['or_key_new_key'] = {single_session: or_num}
        
        dynamics['keys_to_num'] =  {chosen_session : 0}
        dynamics['num_to_keys'] =  {0 :chosen_session }
        dynamics['num_per_region_full'] =   {chosen_session : dynamics['num_per_region_full'][chosen_session]}
        dynamics['Ds'] =   {chosen_session : dynamics['Ds'][chosen_session]}
        dynamics['xs'] =   {chosen_session : dynamics['xs'][chosen_session]}
        dynamics['ys'] =   {chosen_session : dynamics['ys'][chosen_session]}
        

  else:
      print(type_dyn)
      raise ValueError('not found')
  if single_session > -1 and ('human' not in type_dyn and 'synth' not in type_dyn):
       raise ValueError('TODO')
  return    dynamics


def from_spike_times_to_rate(spike_dict, type_convert = 'discrete',
                             res = 0.01, max_min_val = [], return_T = False, 
                             T_max = np.inf, T_min = 0,  params_gauss = {}):
    """
    Converts spike times to firing rates.
    spike dict is dictionary of units vs spike times
    res is how much to mutiply it by
    Parameters:
    - spike_dict (dict): A dictionary of units vs spike times.
    - res (float): A value by which to multiply the spike times.
    - type_convert (str): Type of conversion to perform (default is 'discrete').
    - Ts (dict): Dictionary containing time indices.
    - Ns (dict): Dictionary containing neuron indices.
    - firings_rates_gauss (dict): Dictionary containing Gaussian-convolved firing rates.
    - firings_rates (dict): Dictionary containing firing rates.
    - max_min_val (list): List containing minimum and maximum values.
    - return_T (bool): Whether to return firing rate matrices (default is False).
    - T_max (float): Maximum time value (default is np.inf).
    - params_gauss (dict): Dictionary containing parameters for Gaussian convolution.
    
    Returns:
    - firing_rate_mat (ndarray): Matrix containing firing rates.
    - firing_rate_mat_gauss (ndarray): Matrix containing Gaussian-convolved firing rates.
    - return_T (bool): Whether to return firing rate matrices.    
    import numpy as np
    """  
    if isinstance(spike_dict , (np.ndarray, list)):
        spike_dict = {1: spike_dict}       
        
        
    if T_min >= T_max:
        raise ValueError('T_min must be larger than T_max')
    if res != 1:
        spike_dict = {key:np.array(val) / res for key,val in spike_dict.items()}
    if T_min > 0:
        spike_dict = {key:val - T_min for key,val in spike_dict.items()}
        spike_dict = {key : val[val > 0] for key,val in spike_dict.items()}
        print(len(list(spike_dict.values())[0] ))

        
    """
    make sure keys are continues
    """
    if set(np.arange(len(spike_dict))) != set(list(spike_dict.keys())):
        new_keys = np.arange(spike_dict)
        old_keys = list(spike_dict.keys())
        old2new = {old:new for old,new in zip(old_keys, new_keys)}
        spike_dict = {old2new[key]:val for key,val in spike_dict.items()}
    else:
        old2new = {}
    
    
    # FILL IN A SINGLE RANDOM VALUE FOR EMPTY FIRES
    
    
    if checkEmptyList(max_min_val):
        min_val = np.nanmin([np.min(val)  if len(val) > 0 else np.nan for val in list(spike_dict.values())   ])
        max_val = np.nanmax([np.max(val)  if len(val) > 0 else np.nan for val in list(spike_dict.values())   ])

        
        
    N = len(spike_dict)
        
    if T_min > 0:
        max_val = max_val - T_min     
    max_val = int(np.ceil(max_val))
    max_val = int(np.min([max_val, T_max]))
    firing_rate_mat = np.zeros((int(N) ,max_val))    

        
    if type_convert == 'discrete':         
        T_thres = T_max 
        tup_neurons_and_spikes = np.vstack([ np.hstack([np.array([neuron]*np.sum( times < T_thres )).reshape((-1,1)) , np.array(times[ times < T_thres]).reshape((-1,1)) ])
                                  for neuron, times  in spike_dict.items()])
        rows =  tup_neurons_and_spikes[:,0]
        cols =  tup_neurons_and_spikes[:,1]
        
        data = np.ones(len(rows))  # Assuming all values are 1
        sparse_mat = coo_matrix((data, (rows, cols)), shape=(N, max_val))
        

        firing_rate_mat = sparse_mat.toarray()
        firing_rate_mat_gauss = gaussian_convolve(firing_rate_mat,  **params_gauss)
            
    if T_min > 0 :     
        firing_rate_mat = firing_rate_mat[:, T_min:]
        firing_rate_mat_gauss = firing_rate_mat_gauss[:, T_min:]
    if return_T:
        return  firing_rate_mat, firing_rate_mat_gauss, return_T
    return  firing_rate_mat, firing_rate_mat_gauss, old2new
            
def plot_raster(dict_spike_time, ax = [], fig = [], max_T = np.inf, res = 0.05, plot_params = {}):
    # input is neuron num as key
    if checkEmptyList(ax):
        fig, ax = plt.subplots()
    T = np.max([np.max(times) for times in list(dict_spike_time.values())])
    keys = list(dict_spike_time.keys())
    yval = np.array(lists2list([[key]*len(val) for key, val in dict_spike_time.items() ]))
    xval = np.array(lists2list([list(val/res) for key, val in dict_spike_time.items() ]))
    
    if max_T < T:
        yval = yval[xval < max_T]
        xval = xval[xval < max_T]
        
    ax.scatter(xval, yval, marker = '$|$', **plot_params)
    
    

def create_simple_cbar(vmin = 0, vmax = 1, cmap = 'Reds', to_return = False, 
                       center = None, cbar_kws = {}, aspect = 10, params_heat = {}):
    fig, axs = plt.subplots()
    sns.heatmap(np.random.rand(3,3)*np.nan, vmin = vmin, vmax = vmax , 
                cmap = cmap, center = center, cbar_kws=cbar_kws, ax = axs, **params_heat)
    # Adjust the width of the colorbar
    cbar = axs.collections[0].colorbar
    cbar.ax.set_aspect(aspect)
    remove_edges(axs, left = False, bottom = False, include_ticks = False)
    if to_return:
        return fig
    
    
def take_mid(arr):
    arr = np.array(arr)
    return 0.5*(arr[1:] + arr[:-1])    
        
def create_legend(dict_legend, size = 30, save_formats = ['.png','.svg'], 
                  save_addi = 'legend' , dict_legend_marker = {}, 
                  marker = '.', style = 'plot', s = 500, to_save = True, plot_params = {'lw':15},
                  save_path = os.getcwd(), params_leg = {}):
    fig, ax = plt.subplots()
    if style == 'plot':
        [ax.plot([],[], 
                 c = dict_legend[area], label = area, marker = dict_legend_marker.get(area), **plot_params) for area in dict_legend]
    else:
        if len(dict_legend_marker) == 0:
            [ax.scatter([],[], s=s,c = dict_legend.get(area), label = area, marker = marker, **plot_params) for area in dict_legend]
        else:
            [ax.scatter([],[], s=s,c = dict_legend[area], label = area, marker = dict_legend_marker.get(area), **plot_params) for area in dict_legend]
    ax.legend(prop = {'size':size},**params_leg)
    remove_edges(ax, left = False, bottom = False, include_ticks = False)
    fig.tight_layout()
    if to_save:
        [fig.savefig(save_path + os.sep + 'legend_%s%s'%(save_addi,type_save)) 
         for type_save in save_formats]
          
    
def save_fig(name_fig,fig, save_path = '', formats = ['png','svg']) :
    if len(save_path) == 0:
        save_path = os.getcwd()
        
    [fig.savefig(save_path + os.sep + '%s.%s'%(name_fig, format_i)) for format_i in formats]
        
        
        
def plot_side_by_side(latent_dyn, reco, num = 15):
    """
    Plot the latent dynamics and reconstruction side by side.

    Args:
        latent_dyn (numpy.ndarray): Latent dynamics array.
        reco (numpy.ndarray): Reconstruction array.
        num (int, optional): Number of plots to display. Defaults to 15.

    """
    fig, ax = plt.subplots(3,5, sharex = True)
    ax = ax.flatten()
    [ax_spec.plot(latent_dyn[i,:], color = 'blue') for i,ax_spec in enumerate(ax)]
    [ax_spec.plot(reco[i,:], color = 'red') for i,ax_spec in enumerate(ax)]
    
def rgb_to_hex(rgb_vec):
  """
    Convert an RGB color vector to its corresponding hexadecimal representation.
    
    Args:
        rgb_vec (list): RGB color vector.
    
    Returns:
        str: Hexadecimal representation of the RGB color.
    
  """    
  r = rgb_vec[0]; g = rgb_vec[1]; b = rgb_vec[2]
  return rgb2hex(int(255*r), int(255*g), int(255*b))


def quiver_plot(sub_dyn = [], xmin = -5, xmax = 5, ymin = -5, ymax = 5, ax = [], chosen_color = 'red',
                alpha = 0.4, w = 0.02, type_plot = 'quiver', zmin = -5, zmax = 5, cons_color = False, cmap = None,
                return_artist = False,xlabel = 'x',ylabel = 'y',quiver_3d = False,inter=2, projection = [0,1]):
    
    if sub_dyn.shape[0] > 2: 
        f_proj = sub_dyn[:,projection]
        sub_dyn = f_proj[projection,:]
    """
    type_plot - can be either quiver or streamplot
    """
    
    if len(sub_dyn) == 0:
        sub_dyn =  np.array([[0,-1],[1,0]])

    
    if ymin >= ymax:
        raise ValueError('ymin should be < ymax')
    elif xmin >=xmax:            
        raise ValueError('xmin should be < xmax')
    else:

        if not quiver_3d:
            if isinstance(ax,list) and len(ax) == 0:
                fig, ax = plt.subplots()
            X, Y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin,ymax))

            new_mat = sub_dyn - np.eye(len(sub_dyn))

            U = new_mat[0,:] @ np.vstack([X.flatten(), Y.flatten()])
            V = new_mat[1,:] @ np.vstack([X.flatten(), Y.flatten()])

            if type_plot == 'quiver':
                h = ax.quiver(X,Y,U,V, color = chosen_color, alpha = alpha, width = w)
            elif type_plot == 'streamplot':

                
                x = np.linspace(xmin,xmax,100)
                y = np.linspace(ymin,ymax,100)
                X, Y = np.meshgrid(x, y)
                new_mat = sub_dyn - np.eye(len(sub_dyn))
                U = new_mat[0,:] @ np.vstack([X.flatten(), Y.flatten()])
                V = new_mat[1,:] @ np.vstack([X.flatten(), Y.flatten()])
                

                if cons_color:

                    if len(chosen_color[:]) == 3 and isinstance(chosen_color, (list,np.ndarray)): 
                        color_stream = rgb_to_hex(chosen_color)
                    elif isinstance(chosen_color, str) and chosen_color[0] != '#':
                        color_stream = list(name_to_rgb(chosen_color))
                    else:
                        color_stream = chosen_color

                else:
                    new_mat_color = np.abs(new_mat  @ np.vstack([x.flatten(), y.flatten()]))
                    color_stream = new_mat_color.T @ new_mat_color
                try:
                    h = ax.streamplot(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100),U.reshape(X.shape),V.reshape(Y.shape), color = color_stream, cmap = cmap) #chosen_color
                except:
                    h = ax.streamplot(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100),U.reshape(X.shape),V.reshape(Y.shape), color = chosen_color, cmap = cmap) #chosen_color
            else:
                raise NameError('Wrong plot name')
        else:
            if isinstance(ax,list) and len(ax) == 0:
                fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
            X, Y , Z = np.meshgrid(np.arange(xmin, xmax,inter), np.arange(ymin,ymax,inter), np.arange(zmin,zmax,inter))

            new_mat = sub_dyn - np.eye(len(sub_dyn))
            U = np.zeros(X.shape); V = np.zeros(X.shape); W = np.zeros(X.shape); 

            for xloc in np.arange(X.shape[0]):
                for yloc in np.arange(X.shape[1]):
                    for zloc in np.arange(X.shape[2]):
                        U[xloc,yloc,zloc] = new_mat[0,:] @ np.array([X[xloc,yloc,zloc] ,Y[xloc,yloc,zloc] ,Z[xloc,yloc,zloc] ]).reshape((-1,1))
                        V[xloc,yloc,zloc] = new_mat[1,:] @ np.array([X[xloc,yloc,zloc] ,Y[xloc,yloc,zloc] ,Z[xloc,yloc,zloc] ]).reshape((-1,1))
                        W[xloc,yloc,zloc] = new_mat[2,:] @ np.array([X[xloc,yloc,zloc] ,Y[xloc,yloc,zloc] ,Z[xloc,yloc,zloc] ]).reshape((-1,1))

            if type_plot == 'quiver':                    
                h = ax.quiver(X,Y,Z,U,V,W, color = chosen_color, alpha = alpha,lw = 1.5, length=0.8, normalize=True,arrow_length_ratio=0.5)#, width = w
                ax.grid(False)
            elif type_plot == 'streamplot':
                raise NameError('streamplot is not accepted for the 3d case')
         
            else:
                raise NameError('Wront plot name')
    if quiver_3d: zlabel ='z'
    else: zlabel = None
 
    add_labels(ax, zlabel = zlabel, xlabel = xlabel, ylabel = ylabel) 
    if return_artist: return h
            
            
            
    

    
    
def movmfunc(func, mat, window = 3, direction = 0, dist = 'uni'):
  """
  moving window with applying the function func on the matrix 'mat' towrads the direction 'direction'
  dist: can be 'uni' (uniform) or 'gaus' (Gaussian)
  """
  if len(mat.shape) == 1: 
      mat = mat.reshape((-1,1))
      direction = 0
  if np.mod(window,2) == 1:
      addition = int((window-1)/2)
  else:
      addition = int(window/2)
  if direction == 0:
    
    if dist == 'uni':
        mat_wrap = np.vstack([np.nan*np.ones((addition,np.shape(mat)[1])), mat, np.nan*np.ones((addition,np.shape(mat)[1]))])
        movefunc_res = np.vstack([func(mat_wrap[i-addition:i+addition+1,:],axis = direction) for i in range(addition, np.shape(mat_wrap)[0]-addition)])
        
    elif dist == 'gaus':
        mat_wrap = np.vstack([mat[:addition,:][::-1,:], mat, mat[-addition:,:][::-1,:]])
        if np.mod(window,2) == 1:
            wind = np.hstack([np.arange(np.floor(window/2)),np.floor(window/2),np.arange(np.floor(window/2))[::-1] ])+1
            addi = 1
        else:
            wind = np.hstack([np.arange(np.floor(window/2)),np.arange(np.floor(window/2))[::-1] ])+1
            addi = 0
        wind = wind**2
        wind = wind/np.sum(wind)

        movefunc_res = np.vstack([((wind.reshape((1,-1)) @ mat_wrap[i-addition:i+addition+addi,:]).reshape((1,-1)) ) 
                                  for i in range(addition, np.shape(mat_wrap)[0]-addition)])        
  elif direction == 1:
    if dist == 'uni':
        mat_wrap = np.hstack([np.nan*np.ones((np.shape(mat)[0],addition)), mat, np.nan*np.ones((np.shape(mat)[0],addition))])
        movefunc_res = np.hstack([func(mat_wrap[:,i-addition:i+addition+1],axis = direction).reshape((-1,1)) for i in range(addition, np.shape(mat_wrap)[1]-addition)])

    
    elif dist == 'gaus':
        mat_wrap = np.hstack([mat[:,:addition][:,::-1], mat, mat[:,-addition:][:,::-1]])
        if np.mod(window,2) == 1:
            wind = np.hstack([np.arange(np.floor(window/2)),np.floor(window/2),np.arange(np.floor(window/2))[::-1] ])+1
            addi = 1
        else:
            wind = np.hstack([np.arange(np.floor(window/2)),np.arange(np.floor(window/2))[::-1] ])+1
            addi = 0
        wind = wind**2
        wind = wind/np.sum(wind)

        movefunc_res = np.hstack([(( mat_wrap[:,i-addition:i+addition+addi] @ wind.reshape((-1,1)) ).reshape((-1,1)) ) 
                                  for i in range(addition, np.shape(mat_wrap)[1]-addition)])  
        
  return movefunc_res

def red_mean(mat, axis=1):
    """
    Subtract the mean along the specified axis from the input matrix.

    Args:
        mat (numpy.ndarray): Input matrix.
        axis (int, optional): Axis along which to compute the mean. Defaults to 1.

    Returns:
        numpy.ndarray: Matrix with the mean subtracted.

    """
    if axis == 1:
        return mat - mat.mean(axis=axis).reshape((-1, 1))
    else:
        return mat - mat.mean(axis=axis).reshape((1, -1))




def reco_accumulated(latent_dyn, coefficients, F, reverse = False, noisy = []):
    if reverse:        next_dyn = latent_dyn[:,-1].reshape((-1,1))
    else:              next_dyn = latent_dyn[:,0].reshape((-1,1))
    if len(noisy) == 0 : noisy = np.zeros((latent_dyn.shape[0],coefficients.shape[1]+1))
    next_dyn_all = next_dyn
    for t in np.arange(coefficients.shape[1]):
        if reverse:
            next_dyn = np.linalg.pinv(np.sum(np.dstack([F[j]*coefficients[j,t] for j in range(len(F))]),2)) @ next_dyn.reshape((-1,1)) 
            next_dyn_all = np.hstack([next_dyn, next_dyn_all])
        else:

            next_dyn = np.sum(np.dstack([F[j]*coefficients[j,t] for j in range(len(F))]),2) @ next_dyn.reshape((-1,1)) + noisy[:,t].reshape((-1,1))
            
            
        next_dyn_all = np.hstack([next_dyn_all, next_dyn])
    return next_dyn_all

def reco_accum(coefficients, F, latent_dyn , w_I = True     ):
    """
    Accumulate reconstructions based on coefficients, frequency spectra, and latent dynamics.

    Args:
        coefficients (numpy.ndarray): Coefficients array.
        F (list): List of frequency spectra.
        latent_dyn (numpy.ndarray): Latent dynamics array.
        w_I (bool, optional): Whether to include the identity matrix. Defaults to True.

    Returns:
        numpy.ndarray: Accumulated reconstructions.

    """    
    if len(coefficients.shape) == 1: coefficients = coefficients.reshape((1,-1))
    x0 = latent_dyn[:,0].reshape((-1,1))
    for t in range(coefficients.shape[1]):#
        next_one = np.sum(np.dstack([ coefficients[j,t] *(np.eye(F[0].shape[0])+ 0.5*F[j]) @ x0[:,-1].reshape((-1,1))
                                     for j in range(coefficients.shape[0])]),2)
        x0 = np.hstack([x0, next_one])
    return x0
    
    
def create_reco_with_identitiy(coefficients, F, latent_dyn                               )    :
    """
    Create reconstructions using coefficients, frequency spectra, and latent dynamics.

    Args:
        coefficients (numpy.ndarray): Coefficients array.
        F (list): List of frequency spectra.
        latent_dyn (numpy.ndarray): Latent dynamics array.

    Returns:
        numpy.ndarray: Reconstructed array.

    """
    return np.hstack([((np.eye(F[0].shape) + np.sum(np.dstack([coefficients[j,t] *F[j] for j in range(len(F)) ] ),2)
                       ) @ latent_dyn[:,t].reshape((-1,1)) ).reshape((-1,1)) for t in range(coefficients.shape[1])])


def create_reco(latent_dyn,coefficients, F, accumulation = False, step_n = 1,type_find = 'median',
                min_far =10, smooth_coeffs = False, smoothing_params = {'wind':5},enable_history = True, 
                bias_type = 'disable', bias_val = []):
  """
  This function creates the reconstruction 
  step_n: if accumulation -> how many previous samples to consider
          if accumulation == False -> the reconstruction order
  bias_type: can be:
      disable - no internal bias
      shift  - shift of the reconstructed dynamics by a fixed value
      each   - add the bias inside the reconstruction
  """
  if smooth_coeffs: # future
    coefficients = movmfunc(np.nanmedian, coefficients, window = smoothing_params['wind'], direction = 1)
  if accumulation and step_n > coefficients.shape[1] + 1: step_n =  coefficients.shape[1] + 1
  if accumulation:
    calcul_history = False
    step_n =  coefficients.shape[1] + 1
    cur_reco = latent_dyn[:,0].reshape((-1,1))
    for time_point in range(latent_dyn.shape[1]-1):
      next_dyn1 = create_next(cur_reco, coefficients, F,time_point)
      if step_n == 1:
        next_dyn = next_dyn1
      else:
        if (next_dyn1 < min_far).all():
          next_dyns = [next_dyn1]
        else:
          next_dyns = []

        for order in range(2,step_n+1):
          if time_point-order+1 >= 0:#cur_reco.shape[1]
            cur_next_dyn = create_next(latent_dyn, coefficients, F,time_point-order+1, order = order)
            if (cur_next_dyn < min_far).all():
              next_dyns.append(cur_next_dyn)
        if len(next_dyns) > 0:          
          if type_find == 'mean':
            next_dyn = np.dstack(next_dyns).mean(2)
          elif type_find == 'median':
            next_dyn = np.median(np.dstack(next_dyns),2)
          else:
            raise NameError('Unknown type find')
        else:
          calcul_history = True
      if enable_history and (((step_n == 1) and (not (next_dyn1 < min_far).all())) or (calcul_history)):
        addi = 1    
        while not (next_dyn < min_far).all():          
          if time_point-step_n+1-addi <=0:
            next_dyn = next_dyn1
            break
          next_dyn = create_next(latent_dyn, coefficients, F,time_point-step_n+1-addi, order = step_n+addi)
          addi += 1
      else:
        next_dyn = next_dyn1
      if bias_type == 'each'  and len(bias_val) > 0:
          cur_reco = np.hstack([cur_reco, next_dyn.reshape(-1,1) + bias_val.reshape(-1,1)])
      else:    
          cur_reco = np.hstack([cur_reco, next_dyn.reshape(-1,1)])
  else:
    if bias_type == 'each'  and len(bias_val) > 0:
        cur_reco = np.hstack([create_next(latent_dyn, coefficients, F,time_point)+ bias_val.reshape(-1,1) for time_point in range(latent_dyn.shape[1]-1)])
        cur_reco = np.hstack([latent_dyn[:,0].reshape((-1,1)),cur_reco])
    else:

        
        cur_reco = np.hstack([create_next(latent_dyn, coefficients, F,time_point) 
                              for time_point in range(latent_dyn.shape[1]-1)])
        cur_reco = np.hstack([latent_dyn[:,0].reshape((-1,1)),cur_reco])
    
    if step_n <= 1:
        pass
    else:
      cur_reco = create_reco(cur_reco,coefficients, F, accumulation = False, step_n = step_n-1,type_find = type_find, smooth_coeffs = smooth_coeffs, smoothing_params = smoothing_params)
  if bias_type == 'shift' and len(bias_val) > 0:
      cur_reco = cur_reco + bias_val.reshape(-1,1)
  return cur_reco


def create_reco_step_reverse(latent_dyn, coefficients, F):

    x_t_preds = np.vstack([np.linalg.pinv(np.sum(np.dstack([coefficients[i,t-1] * F[i] for i in range(len(F))]),2)) @ latent_dyn[:,t] 
                 for t in  np.arange(latent_dyn.shape[1]-1, 0, -1)])[::-1,:]

    x_t_preds = np.hstack([x_t_preds.T, latent_dyn[:,-1].reshape((-1,1))])
    return x_t_preds
    
    
#%% Post-Proc Functions
  
def add_labels(ax, xlabel='X', ylabel='Y', zlabel='', title='', xlim = None, ylim = None, zlim = None,xticklabels = np.array([None]),
               yticklabels = np.array([None] ), xticks = [], yticks = [], legend = [], 
               ylabel_params = {'fontsize':19},zlabel_params = {'fontsize':19}, xlabel_params = {'fontsize':19}, 
               title_params = {'fontsize':19}, format_xticks = 0, format_yticks = 0):
  """
  This function add labels, titles, limits, etc. to figures;
  Inputs:
      ax      = the subplot to edit
      xlabel  = xlabel
      ylabel  = ylabel
      zlabel  = zlabel (if the figure is 2d please define zlabel = None)
      etc.
  """
  if xlabel != '' and xlabel != None: ax.set_xlabel(xlabel, **xlabel_params)
  if ylabel != '' and ylabel != None:ax.set_ylabel(ylabel, **ylabel_params)
  if zlabel != '' and zlabel != None:ax.set_zlabel(zlabel,**zlabel_params)
  if title != '' and title != None: ax.set_title(title, **title_params)
  if xlim != None: ax.set_xlim(xlim)
  if ylim != None: ax.set_ylim(ylim)
  if zlim != None: ax.set_zlim(zlim)
  
  if (np.array(xticklabels) != None).any(): 
      if len(xticks) == 0: xticks = np.arange(len(xticklabels))
      ax.set_xticks(xticks);
      ax.set_xticklabels(xticklabels);
  if (np.array(yticklabels) != None).any(): 
      if len(yticks) == 0: yticks = np.arange(len(yticklabels)) +0.5
      ax.set_yticks(yticks);
      ax.set_yticklabels(yticklabels);
  if len(legend)       > 0:  ax.legend(legend)
  if format_xticks > 0:
      ax.xaxis.set_major_formatter(FormatStrFormatter('%.%df'%format_xticks))
  if format_yticks > 0:
      
      ax.yaxis.set_major_formatter(FormatStrFormatter('%.%df'%format_yticks))
      

def remove_background(ax, grid = False, axis_off = True):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    if not grid:
        ax.grid(grid)
    if axis_off:
        ax.set_axis_off()
    

def visualize_dyn(dyn,ax = [], params_plot = {},turn_off_back = False, marker_size = 10, include_line = False, 
                  color_sig = [],cmap = 'cool', return_fig = False, color_by_dominant = False, coefficients =[],
                  figsize = (5,5),colorbar = False, colors = [],vmin = None,vmax = None, color_mix = False, alpha = 0.4,
                  colors_dyns = np.array(['r','g','b','yellow']), add_text = 't ', text_points = [],fontsize_times = 18, 
                  marker = "o",delta_text = 0.5, color_for_0 =None, legend = [],fig = [],return_mappable = False,
                  remove_back = True, edgecolors='none', line_params = {}, view_init = [30,-60]):
   """
   Plot the multi-dimensional dynamics
   Inputs:
       dyn          = dynamics to plot. Should be a np.array with size k X T
       ax           = the subplot to plot in (optional)
       params_plot  = additional parameters for the plotting (optional). Can include plotting-related keys like xlabel, ylabel, title, etc.
       turn_off_back= disable backgroud of the plot? (optional). Boolean
       marker_size  = marker size of the plot (optional). Integer
       include_line = add a curve to the plot (in addition to the scatter plot). Boolean
       color_sig    = the color signal. if empty and color_by_dominant - color by the dominant dynamics. If empty and not color_by_dominant - color by time.
       cmap         = cmap
       colors       = if not empty -> pre-defined colors for the different sub-dynamics. Otherwise - colors are according to the cmap.
       color_mix    = relevant only if  color_by_dominant. In this case the colors need to be in the form of [r,g,b]
   Output:
       (only if return_fig) -> returns the figure      
      
   """
   if not isinstance(color_sig,list) and not isinstance(color_sig,np.ndarray): color_sig = [color_sig]

 

   if isinstance(ax,list) and len(ax) == 0:
       if dyn.shape[0] == 3:
           fig, ax = plt.subplots(figsize = figsize, subplot_kw={'projection':'3d'})  
       else:
           fig, ax = plt.subplots(figsize = figsize)  
           
       

   if include_line:

       if dyn.shape[0] == 3:
           ax.plot(dyn[0,:], dyn[1,:], dyn[2,:], **line_params)
       else:
           ax.plot(dyn[0,:], dyn[1,:], alpha = 0.2,**line_params)
   if len(legend) > 0:
       [ax.scatter([],[], c = colors_dyns[i], label = legend[i], s = 10) for i in np.arange(len(legend))]
       ax.legend()
   # Create color sig        
   if len(color_sig) == 0: 
       color_sig = np.arange(dyn.shape[1])      
   if color_by_dominant and (coefficients.shape[1] == dyn.shape[1]-1 or coefficients.shape[1] == dyn.shape[1]): 
       if color_mix:
           if len(colors) == 0 or not np.shape(colors)[0] == 3: raise ValueError('colors mat should have 3 rows')
           else:

               color_sig = ((np.array(colors)[:,:coefficients.shape[0]] @ np.abs(coefficients))  / np.max(np.abs(coefficients).sum(0).reshape((1,-1)))).T
               color_sig[np.isnan(color_sig) ] = 0.1
          
               dyn = dyn[:,:-1]
       else:
           
           color_sig_tmp = find_dominant_dyn(coefficients)
           if len(colors_dyns) > 0: 
               color_sig = colors_dyns[color_sig_tmp]
           elif len(color_sig) == 0:  
               color_sig=color_sig_tmp 
           else:        
               color_sig=np.array(color_sig)[color_sig_tmp] 
           if len(color_sig.flatten()) < dyn.shape[1]: dyn = dyn[:,:len(color_sig.flatten())]
           if color_for_0:

               color_sig[np.sum(coefficients,0) == 0] = color_for_0



   if dyn.shape[0] > 2:
       if len(colors) == 0:
           h = ax.scatter(dyn[0,:], dyn[1,:], dyn[2,:], marker = marker, s = marker_size,
                          c= color_sig,cmap = cmap, alpha = alpha,
                          vmin = vmin, vmax = vmax, edgecolors=edgecolors)
       elif isinstance(colors,str):
           h = ax.scatter(dyn[0,:], dyn[1,:], dyn[2,:], marker =marker, s = marker_size,c= colors, alpha = alpha, edgecolors=edgecolors)
       else:
           h = ax.scatter(dyn[0,:], dyn[1,:], dyn[2,:], marker =marker, s = marker_size,c= color_sig, alpha = alpha, edgecolors=edgecolors)
   else:
       dyn = np.array(dyn)
       
       if len(colors) == 0:
           h = ax.scatter(dyn[0,:], dyn[1,:],  marker = marker, s = marker_size,c= color_sig,cmap = cmap, edgecolors=edgecolors, alpha = alpha,
                          vmin = vmin, vmax = vmax)
       elif isinstance(colors,str):
           h = ax.scatter(dyn[0,:], dyn[1,:], marker =marker, s = marker_size,c= colors, edgecolors=edgecolors, alpha = alpha)
       else:
           h = ax.scatter(dyn[0,:], dyn[1,:],  marker = marker, s = marker_size,c= color_sig, alpha = alpha, edgecolors=edgecolors)
  
           params_plot['zlabel'] = None
   if len(params_plot) > 0:
     if dyn.shape[0] == 3:
         if 'xlabel' in params_plot.keys():
           add_labels(ax, xlabel=params_plot.get('xlabel'), ylabel=params_plot.get('ylabel'), zlabel=params_plot.get('zlabel'), title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
         elif 'zlabel' in params_plot.keys():
               add_labels(ax,  zlabel=params_plot.get('zlabel'), title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
         else:
           add_labels(ax,   title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
     else:
         if 'xlabel' in params_plot.keys():
           add_labels(ax, xlabel=params_plot.get('xlabel'), ylabel=params_plot.get('ylabel'), zlabel=None, title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =None)
         elif 'zlabel' in params_plot.keys():
               add_labels(ax,  zlabel=None, title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =None)
         else:
           add_labels(ax,   title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =None,zlabel = None);
   if len(text_points) > 0:
       
       if dyn.shape[0] == 3:
           [ax.text(dyn[0,t]+delta_text,dyn[1,t]+delta_text,dyn[2,t]+delta_text, '%s = %s'%(add_text, str(t)),  fontsize =fontsize_times, fontweight = 'bold') for t in text_points]
       else:
           [ax.text(dyn[0,t]+delta_text,dyn[1,t]+delta_text, '%s = %s'%(add_text, str(t)),  fontsize =fontsize_times, fontweight = 'bold') for t in text_points]
   if remove_back:
       remove_edges(ax)
       ax.set_axis_off()
   if colorbar:
       fig.colorbar(h)
   if not checkEmptyList(view_init):
       ax.view_init(view_init[0], view_init[1])
   if return_mappable:
       return h
       
def find_stretch_and_rot(reco_dyn, real_dyn, to_rot = True, to_strecth = True, sens = 0.01)   :
    
    if to_rot:
        rots = [create_rotation_mat(theta = phi, dims = 2) for phi in np.arange(0,np.pi, sens)]
        errors = [np.sum((rot @ reco_dyn - real_dyn)**2) for rot in rots]
        loc_min = np.argmin(errors)
        reco_dyn = rots[loc_min] @ reco_dyn
    if to_strecth:
        
        for dim in range(real_dyn.shape[0]):
            stre_opts = np.arange(0.01, 100, sens)
            errors = [np.sum((reco_dyn[dim]*stre - real_dyn[dim])**2)  for stre in stre_opts]
            loc_min = np.argmin(errors)
            reco_dyn[dim] =  reco_dyn[dim] * stre_opts[loc_min]
    return reco_dyn
        
        




def check_z_increase(reco, to_run= {}, ax = [], fig = [], ax_c = [], fig_c = [],coefficients = [], 
                     colors = ['black','gray','lightgray','darkgreen'],
                     lw = 5, lss = ['-','--','dashdot','dotted'],
                     colors_lab = ['brown','darkgreen','darkblue','dimgray','darkorange','crimson','C'], 
                     fsize = 16, lab = 0):        
    to_run = {**{'dz':True,'z':True,'dz_z':True, 'z_z':True}, **to_run}
    fig, ax = create_ax(ax, return_fig = True, nums = (1,1), size = (1,15))
    if len(coefficients) > 0:
        fig_c, ax_c = create_ax(ax_c, return_fig = True, nums = (1,1), size = (1,15))
    else:
        print('You should provide coeffs to check_z_increase')
        

    if to_run.get('dz'): ax.plot(np.diff(reco[2,:]), color = colors[0], lw = lw, label = '$\Delta(z_t)$', ls = lss[0])
 
    if to_run.get('z'):  ax.plot(reco[2,:], color = colors[1], lw = lw, label = '$z_t$', ls = lss[1])
    
    if to_run.get('dz_z'): ax.plot(np.diff(reco[2,reco[2,:] != 0])/reco[2,reco[2,:] != 0][:-1], color = colors[2], ls= lss[2],
                                   lw = lw, label = r'$\frac{\Delta(z_t)}{z_t}$')
    
    if to_run.get('z_z'): ax.plot(reco[2,reco[2,:] != 0][1:]/reco[2,reco[2,:] != 0][:-1], 
                                  color = colors[3], ls = lss[3],
                                  lw = lw, label = r'$\frac{z_{t+1}}{z_t}$')
    remove_edges(ax, include_ticks=True, bottom = True, left = True)
    ax.legend(prop = {'size':16})
    add_labels(ax, xlabel = 'Time',ylabel = None, zlabel = None, title = 'change in z, for $c_%d$'%lab, xlabel_params = {'fontsize':fsize},
     title_params = {'fontsize':fsize}  )
    
    if len(coefficients) > 0:
      
        ax_c.plot(coefficients.flatten(), colors_lab[lab], lw = lw, ls = lss[0], 
                  label = '$c_%d$'%(lab+1), alpha = 0.4)
        ax_c.plot(reco[2,reco[2,:] != 0][1:]/reco[2,reco[2,:] != 0][:-1], 
                                     color = colors_lab[lab], ls = lss[0],
                                     lw = lw, label =  r'$\frac{z_{t+1}}{z_t}$')
        figx, axx = plt.subplots()
        axx.plot(coefficients.flatten())

        add_labels(ax, xlabel = 'Time',ylabel = None, zlabel = None, title = 'change in z', xlabel_params = {'fontsize':fsize},
         title_params = {'fontsize':fsize}  )
        ax_c.legend(prop = {'size':16})
        

def check_subs_effect(latent_dyn,F,coefficients, ax = [], dict_store = {}, pre_name ='without', to_plot = True , min_time = 0, params_plot = {}, 
                      update_coeffs = True,  return_map = True,
                      color_sig_type = 'mse', fig = [], title_fig = '', include_colorbar = False,cmap = 'cool', random_colors = True,store_data = True,
                      plot_percent = False,range_close = [],ax_percent = [], plot_backward = True, plot_forward = True, figsize = (15,10), 
                      colors = [], colors_sim = [],  remove_back = False, fig_z_increase = [], ax_z_increase = [],to_plot_z_change =True,
                      include_identity = False):
  """
  Check the effect of each sub-dynamics by exploring the gain in error when removing it, and the gain of error when using only it. 
  """    
  if latent_dyn.shape[0] < 3: to_plot_z_change = False
      
  if len(range_close) == 0: range_close = np.linspace(10**-8, 10,30)
  num_subdyns = len(F)
  withouts = [list(itertools.combinations(np.arange(num_subdyns),k)) for k in range(num_subdyns)]  
  if isinstance(colors, list) and len(colors) == 0: 
      colors = ['r','g','b','gray','orange','m','cyan']
  if isinstance(colors_sim, list) and len(colors_sim) == 0: 
      colors = ['brown','darkgreen','darkblue','dimgray','darkorange','crimson','C']
  if store_data: 
      
     
      stored_contri = {'Increase in error WITH only solo sub-dyn':pd.DataFrame(np.zeros((coefficients.shape[0],2)),
                                                                               index = ['f%g'%(i+1) for i in np.arange(coefficients.shape[0])] ,
                                                                               columns= ['Error','% Wrong']),
                       'Increase in error WITHOUT a sub-dyn':pd.DataFrame(np.zeros((coefficients.shape[0],2)), 
                                                                          index = ['f%g'%(i+1) for i in np.arange(coefficients.shape[0])] , columns = ['Error','% wrong'])}
      
  if to_plot_z_change:
     fig_z_increase, ax_z_increase = create_ax(ax_z_increase, fig = fig_z_increase, return_fig= True, size = (5, 5*len(withouts[-1])), nums = (1,len(withouts[-1])))
     fig_c, ax_c = create_ax([], return_fig= True, size = (5*len(F), 5), nums = (1,len(F)), sharey = False)
    

  if plot_percent:
      if isinstance(ax_percent,list):
        if len(ax_percent) == 0:
            fig_percent, ax_percent = plt.subplots(2,2, figsize =figsize)      
  if to_plot:
    if isinstance(ax, list):
      if len(ax) == 0:
          max_len_without = np.max([len(without_spec) for without_spec in withouts])
          if latent_dyn.shape[0] == 3:         fig, ax = plt.subplots(len(withouts),max_len_without,figsize = (max_len_without*8, len(withouts)*6), subplot_kw={'projection':'3d'})  
          else:                                fig, ax = plt.subplots(len(withouts),max_len_without,figsize = (max_len_without*8, len(withouts)*6))  
         
    if not isinstance(ax,np.ndarray):        ax = np.array([[ax]])
    if len(ax.shape) == 1: ax = ax.reshape((-1,1))
  for group_num, without_group in enumerate(withouts):
    for without_num, without in enumerate(without_group):
      with_subs = list(set(np.arange(num_subdyns)) - set(without)) 
      with_subs_color = np.array(colors)[[i for i in range(len(colors)) if i not in without]]
      if update_coeffs:        
        coeffs_run = update_c(np.array(F)[with_subs].tolist(),latent_dyn[:,min_time:],{},include_identity = include_identity )        
      else:
        if len(with_subs) == 1: coeffs_run = coefficients[np.array(with_subs),min_time:].reshape((1,-1))          
        else: 
            coeffs_run = coefficients[np.array(with_subs),min_time:]            
      F_run = [f_i for i,f_i in enumerate(F) if i in with_subs]
      reco = create_reco(latent_dyn[:,min_time:],coeffs_run, F_run)
      name_store = '_'.join(['without'] + [str(num_without) for num_without in without])
      dict_store[name_store] = reco
      mse_without = np.sqrt(np.mean((reco-latent_dyn[:,min_time:])**2))
      ## Store data      
      if store_data:
          if  len(without)  == 0 and plot_forward:
              plot_dots_close(reco,latent_dyn[:,min_time:], range_close =range_close, conf_int = 0.05, ax =ax_percent[1,0], color ='black' , label = 'reference')
              plot_dots_close(reco,latent_dyn[:,min_time:], range_close =range_close, conf_int = 0.05, ax =ax_percent[1,1], color ='black' , label = 'reference')
             
              stored_contri['Increase in error WITHOUT a sub-dyn'].iloc[-1,:] =  calcul_contribution(reco, latent_dyn[:,min_time:], 
                                                                                      direction = 'backward')
              stored_contri['Increase in error WITH only solo sub-dyn'].iloc[-1,:] =  calcul_contribution(reco, latent_dyn[:,min_time:], 
                                                                                     direction = 'forward')

          elif len(with_subs) == 1:
              calcul_contribution(reco, latent_dyn[:,min_time:], direction = 'forward')
              stored_contri['Increase in error WITH only solo sub-dyn'].iloc[with_subs[0],:] =  calcul_contribution(reco, latent_dyn[:,min_time:], direction = 'forward')

              if plot_forward:
                  plot_dots_close(reco,latent_dyn[:,min_time:], range_close =range_close, conf_int = 0.05, ax =ax_percent[1,0],
                                  color =colors[with_subs[0]] , label = '$f_%s$'%str(with_subs[0]+1))
              if to_plot_z_change and reco.shape[0] == 3:
     
                  check_z_increase(reco, ax = ax_z_increase[with_subs[0]], fig = fig_z_increase, ax_c = ax_c[with_subs[0]], fig_c = fig_c, colors_lab = colors,
                                   coefficients = coeffs_run, lab = with_subs[0])#
              
          elif len(without) == 1:

              stored_contri['Increase in error WITHOUT a sub-dyn'].iloc[without[0],:] =  calcul_contribution(reco, latent_dyn[:,min_time:], direction = 'backward')
              if plot_backward:
                  plot_dots_close(reco,latent_dyn[:,min_time:], range_close =range_close, conf_int = 0.25, ax =ax_percent[1,1], 
                                  color =colors[without[0]] , label = '$f_%s$'%str(without[0]+1)  )        
      ## Plot
      if to_plot:
        if color_sig_type == 'mse': 
            color_sig = np.mean(np.abs(reco-latent_dyn[:,min_time:]),0)
            color_by_dominant = False
        elif color_sig_type == 'coeffs':
              color_sig =with_subs_color
              
              color_by_dominant = True
        else:
            color_sig =[]
            color_by_dominant = False
        if to_plot:
            h = visualize_dyn(reco, ax[group_num, without_num], params_plot, color_sig= color_sig, remove_back =  remove_back ,
                          return_fig = True, colors_dyns = [], color_by_dominant = color_by_dominant, 
                          coefficients =coeffs_run  ,cmap = cmap, colors = [], vmin = 0, vmax = coefficients.shape[0])

        if len(params_plot) > 0:
          add_labels(ax[group_num, without_num], xlabel=params_plot.get('xlabel'), ylabel=params_plot.get('ylabel'), zlabel=params_plot.get('zlabel'), title=params_plot.get('title'),
            xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
        else:
          if len(name_store)   > 0:
              name_store_new =name_store.split('_')
              name_store_new = name_store_new[0] + ' ' + ' & '.join(['$f_%s$'%str(int(i)+1) for i in name_store_new[1:]])
          else:
              name_store_new = 'With all sub-dynamics'
          if latent_dyn.shape[0] == 3:
              if group_num == 0 and without_num ==0:
                  ax[group_num, without_num].set_title('With All Operators' + '\n MSE: '+'{:.2e}'.format(mse_without ) )
              else:
                  add_labels(ax[group_num, without_num], title = name_store_new + '\n MSE: '+'{:.2e}'.format(mse_without ) )

              
          else:
              add_labels(ax[group_num, without_num], title = name_store_new + '\n MSE: '+'{:.2e}'.format(mse_without ),
                           zlabel = None)
         
            
    if to_plot:
        [ax_spec.axis('off') for ax_num,ax_spec in enumerate(ax[group_num, :]) if ax_num > without_num]
        [remove_edges(ax_spec) for ax_spec in ax.flatten()]
  if to_plot:
      fig.suptitle(title_fig);
  if include_colorbar:
      fig.subplots_adjust(right=0.7)
      cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
      fig.colorbar(h, cax=cbar_ax)
  if to_plot_z_change:
    [ax_c_spec.legend(prop = {'size':15}) for ax_c_spec in ax_c]
    [ax_c_spec.set_xlabel('Time (AU)', fontsize = 15) for ax_c_spec in ax_c]
    [ax_c_spec.set_title('$f_%d$'%(i+1), fontsize = 15) for i,ax_c_spec in enumerate(ax_c)]
    [remove_edges(ax_c_spec, include_ticks=True, bottom = True, left = True) for ax_c_spec in ax_c]      
  if store_data: 
      if plot_percent:
          ax_percent[1,1].legend(loc = 'lower right', prop = {'size':16})
          ax_percent[1,0].set_title('% points within a distance,\n for reconstruction WITH solo sub-dyn')
          ax_percent[1,0].set_ylim([0,1])
          
          ax_percent[1,1].set_title('% points within a distance, \n for reconstruction WITHOUT a sub-dyn')
          ax_percent[1,1].set_ylim([0,1])

          if return_map:
              return stored_contri, dict_store, h, ax_percent,fig_percent
          else:
              return stored_contri, dict_store, ax_percent,fig_percent
      return stored_contri, dict_store, h

    
  return dict_store, h

def check_eigenspaces(F, colors = [],figsize = (15,8), ax = [], title2 = 'Eigenspaces of different sub-dynamics',title1= 'Eigenvalues of different sub-dynamics'):
  fig = plt.figure(figsize = figsize)
  ax1 = fig.add_subplot(121)
  if np.shape(F[0])[0] == 3:
      ax2 = fig.add_subplot(122, projection='3d')
  else:
      ax2 = fig.add_subplot(122)
  if len(colors) == 0:
    colors = np.random.rand(3,len(F))
  evals_list = []
  evecs_list = []
  for f_num, f_i in enumerate(F):
    if isinstance(colors,np.ndarray):
      cur_color =  [list(colors[:,f_num])]
    else:
      cur_color = colors[f_num]
    eigenvalues, eigenvectors =  linalg.eig(f_i)
    evals_list.append(eigenvalues)
    evecs_list.append(eigenvectors)
    ax1.scatter(np.real(eigenvalues),np.imag(eigenvalues),marker = 'o', c =cur_color, label = 'G%g'%f_num)
    eigenvectors_real = np.real(eigenvectors)

    if eigenvectors_real.shape[0] == 3:
        ax2.scatter( eigenvectors_real[0,:],eigenvectors_real[1,:], eigenvectors_real[2,:],marker = 'o', c = cur_color, label = 'G%g'%f_num)
    elif eigenvectors_real.shape[0] == 2:
        ax2.scatter( eigenvectors_real[0,:],eigenvectors_real[1,:],marker = 'o', c = cur_color, label = 'G%g'%f_num)

    # 1. create vertices from points
    if eigenvectors_real.shape[0] == 3:
        verts = [list(zip(eigenvectors_real[0,:],eigenvectors_real[1,:],eigenvectors_real[2,:]))]
    if eigenvectors_real.shape[0] == 2:
        verts = [list(zip(eigenvectors_real[0,:],eigenvectors_real[1,:]))]
    srf = Poly3DCollection(verts, alpha=.25, facecolor= cur_color)

    ax2.add_collection3d(srf)
  add_labels(ax2, title=title2)
  add_labels(ax1, xlabel='Real', ylabel = 'Img',zlabel =None,  title=title1)

  return evecs_list,evals_list


def add_arrow(ax, start, end,arrowprops = {'facecolor' : 'black', 'width':1.8, 'alpha' :0.5} ):
    arrowprops = {**{'facecolor' : 'black', 'width':1.8, 'alpha' :0.5, 'edgecolor':'none'}, **arrowprops}
    ax.annotate('',ha = 'center', va = 'bottom',  xytext = start,xy =end,
                arrowprops = arrowprops)

    
    
def plot_sub_effect(sub_dyn, rec_rad_all = 5, colors = ['r','g','b','m'], alpha = 0.8, ax = [], 
                    n_points = 100, figsize = (10,10), params_labels = {'title':'sub-dyn effect'}, lw = 4 , projection = [0,1]):
    params_labels = {**{'zlabel':None}, **params_labels}
    
    if sub_dyn.shape[0] > 2: 
        f_proj = sub_dyn[:,projection]
        sub_dyn = f_proj[projection,:]
        
        
    if isinstance(ax,list) and len(ax) == 0:
        fig, ax = plt.subplots(figsize = figsize)
    if len(colors) == 1: colors = [colors]*4
    if not isinstance(rec_rad_all,list): rec_rad_all = [rec_rad_all]
    ax.axhline(0, alpha = 0.1, color = 'black', ls = 'dotted')
    ax.axvline(0, alpha = 0.1, color = 'black', ls = 'dotted')
    for rec_rad in rec_rad_all:        
        ax.plot([-rec_rad, rec_rad],[rec_rad,rec_rad],alpha = alpha**2, color = colors[0], ls ='--',lw=lw)
        ax.plot([-rec_rad, rec_rad],[-rec_rad,-rec_rad],alpha = alpha**2, color = colors[1], ls = '--',lw=lw)
        ax.plot([rec_rad,  rec_rad],[-rec_rad,rec_rad],alpha = alpha**2, color = colors[2], ls = '--',lw=lw)
        ax.plot([-rec_rad,-rec_rad], [ -rec_rad,  rec_rad],alpha = alpha**2, color = colors[3], ls = '--',lw=lw)
    

        if not (sub_dyn == 0).all():
            sub_dyn = norm_mat(sub_dyn, type_norm = 'evals')
        effect_up = sub_dyn @ np.vstack([np.linspace(-rec_rad, rec_rad, n_points), [rec_rad]*n_points])
        effect_down = sub_dyn @ np.vstack([np.linspace(-rec_rad, rec_rad, n_points), [-rec_rad]*n_points])
        effect_right = sub_dyn @ np.vstack([[rec_rad]*n_points,np.linspace(-rec_rad, rec_rad, n_points)])
        effect_left = sub_dyn @ np.vstack([[-rec_rad]*n_points,np.linspace(-rec_rad, rec_rad, n_points)])
        ax.plot(effect_up[0,:],effect_up[1,:],alpha = alpha, color = colors[0],lw=lw)
        ax.plot(effect_down[0,:],effect_down[1,:],alpha = alpha, color = colors[1],lw=lw)
        ax.plot(effect_right[0,:],effect_right[1,:],alpha = alpha, color = colors[2],lw=lw)
        ax.plot(effect_left[0,:],effect_left[1,:],alpha = alpha, color = colors[3],lw=lw)
        # Up
        add_arrow(ax, [0,rec_rad], [np.mean(effect_up[0,:]),np.mean(effect_up[1,:])],arrowprops = {'facecolor' :colors[0]})
        add_arrow(ax, [0,-rec_rad], [np.mean(effect_down[0,:]),np.mean(effect_down[1,:])],arrowprops = {'facecolor' :colors[1]})
        add_arrow(ax, [rec_rad,0], [np.mean(effect_right[0,:]),np.mean(effect_right[1,:])],arrowprops = {'facecolor' :colors[2]})
        add_arrow(ax, [-rec_rad,0], [np.mean(effect_left[0,:]),np.mean(effect_left[1,:])],arrowprops = {'facecolor' :colors[3]})
    add_labels(ax, **params_labels)


    

def plot_evals_evecs(ax , sub_dyn, colors =['r','g','b','m'] , alpha = 0.7, title ='$\lambda$',
                     d3 = False, t = 0, cmap = 'viridis',
                     tcolor = 0, markers = ['_','|'], to_draw = True):
    eigenvalues, eigenvectors =  linalg.eig(sub_dyn)
    if to_draw:
        for eval_num, eigenval in enumerate(eigenvalues):
            
            if not d3:
                ax.scatter( np.real(eigenval),np.imag(eigenval), alpha = alpha, color = colors, s = 300)
            else:
            
                if np.real(eigenval) > 0:
                    ax.scatter(xs = t, ys = np.real(eigenval),zs = np.imag(eigenval), alpha = alpha, c = np.array(tcolor[:-1]).reshape((1,-1)), s = 30, marker = markers[1])#, cmap = cmap )
                else:
                    ax.scatter(xs = t, ys = np.real(eigenval),zs = np.imag(eigenval), alpha = alpha, c = np.array(tcolor[:-1]).reshape((1,-1)), s = 30, marker = markers[0])#, cmap = cmap )
    
        if not d3:
            ax.set_xlabel('Re($\lambda$)')
            ax.set_ylabel('Im($\lambda$)')
            ax.axhline(0, alpha = 0.3, color = 'black', ls = 'dotted')
            ax.axvline(0, alpha = 0.3, color = 'black', ls = 'dotted')
            ax.set_title('evals')
    else:
        return np.real(eigenvalues), np.imag(eigenvalues)

def plot_3d_color_scatter(latent_dyn,coefficients, ax = [], figsize = (15,10), delta = 0.4, colors = []):
    
    if latent_dyn.shape[0] != 3:
        print('Dynamics is not 3d')
        pass
    else:
        if len(colors) == 0:
            colors = ['r','g','b']
        if isinstance(ax,list) and len(ax) == 0:
            fig, ax = plt.subplots(figsize = figsize, subplot_kw={'projection':'3d'})  
        for row in range(coefficients.shape[0]):
            coefficients_row = coefficients[row]
            coefficients_row[coefficients_row == 0]  = 0.01
            
            ax.scatter(latent_dyn[0,:]+delta*row,latent_dyn[1,:]+delta*row,latent_dyn[2,:]+delta, s = coefficients_row**0.3, c = colors[row])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)

def compare_coeffs_to_discrete_coeffs(coefficients_n, axs = [],figsize = (10,15), colors = [[1, 0, 0], [0, 1, 0], [0, 0,1]],
                                      n_rep =350,type_plot = 'heatmap',
                                      titles = ['Dynamics \n Decomposition','Discrete Dynamics \n (flexible weights)','Discrete Dynamics \n (fixed weights)'],
                                      include_inter = True):
    """
    Plot the normalized coefficients of our model vs the discrete versions
    type_plot:  can be 'heatmap' or 'plot'
    """
    if isinstance(axs,list) and len(axs) == 0:
        if include_inter:        
            fig, axs = plt.subplots(3,1, sharex = True, figsize = figsize)
        else:
            fig, axs = plt.subplots(2,1, sharex = True, figsize = figsize)
    colors = np.array(colors[:coefficients_n.shape[0]])        
    
    averaged_cols = np.array(colors).T @ coefficients_n
    dstack_res = np.dstack([averaged_cols[color_num,:] for color_num in range(len(colors[0]))])
    
    max_ind = np.argmax(coefficients_n, axis = 0)
    
    # Coefficients weight fixed
    coefficients_n_zeroed = np.zeros(coefficients_n.shape)
    for max_ind_num, max_ind_spec in enumerate(max_ind): 
        coefficients_n_zeroed[max_ind_spec,max_ind_num]  = coefficients_n[max_ind_spec,max_ind_num]
    averaged_cols_zeroed = np.array(colors).T @ coefficients_n_zeroed
    dstack_res_zeroed = np.dstack([averaged_cols_zeroed[color_num,:] for color_num in range(len(colors[0]))])
    if type_plot == 'heatmap':
        dstack_res_zeroed = dstack_res_zeroed / np.max(dstack_res_zeroed, 1)
        axs[0].imshow(np.repeat(dstack_res[:,:],n_rep, axis = 0), alpha = 0.5)
        if include_inter:
            axs[1].imshow(np.repeat(dstack_res_zeroed[:,:],350, axis = 0), alpha = 0.5)
    elif type_plot == 'plot':
        
        [axs[0].plot(coefficients_n[i,:], color = colors[i]) for i in range(coefficients_n.shape[0])];    
        if include_inter:
            [axs[1].plot(coefficients_n_zeroed[i,:], color = colors[i]) for i in range(coefficients_n_zeroed.shape[0])];
    else:
        raise NameError('Type plot value is invalid. Should be "heatmap" or "plot"')
    # Coefficients weight fixed
    coefficients_n_zeroed_2 = np.zeros(coefficients_n.shape)
    for max_ind_num, max_ind_spec in enumerate(max_ind): 
        coefficients_n_zeroed_2[max_ind_spec,max_ind_num]  = 1
    averaged_cols_zeroed = np.array(colors).T @ coefficients_n_zeroed_2
    dstack_res_zeroed = np.dstack([averaged_cols_zeroed[color_num,:] for color_num in range(len(colors[0]))])
    dstack_res_zeroed = dstack_res_zeroed / np.sum(dstack_res_zeroed,2).reshape((1,-1,1))
    
    if include_inter: num_plot = 2
    else: num_plot = 1
    if type_plot == 'heatmap':

        axs[num_plot].imshow(np.repeat(dstack_res_zeroed[:,:],350, axis = 0), alpha = 0.5)
    elif type_plot == 'plot':
        [axs[num_plot].plot(coefficients_n_zeroed_2[i,:], color = colors[i]) for i in range(coefficients_n_zeroed_2.shape[0])];
    axs[-1].set_xlabel('Time')
    if type_plot == 'heatmap': [ax.set_title(titles[i]) for i, ax in enumerate(axs)]
    elif type_plot == 'plot':  [ax.set_ylabel(titles[i]) for i, ax in enumerate(axs)]
    [ax.spines['top'].set_visible(False) for ax in axs]
    [ax.spines['right'].set_visible(False) for ax in axs]
    [ax.spines['bottom'].set_visible(False) for ax in axs]
    [ax.spines['left'].set_visible(False)    for ax in axs]
    [ax.get_xaxis().set_ticks([]) for ax in axs]
    [ax.get_yaxis().set_ticks([]) for ax in axs]
    if type_plot == 'heatmap': fig.subplots_adjust(wspace=0.02,hspace=0.02)
    elif type_plot == 'plot': fig.subplots_adjust(wspace=0.3,hspace=0.3)
    
def show_spines(ax):
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    return ax
def add_dummy_sub_legend(ax, colors,lenf, label_base = 'f'):
    dummy_lines = []
    for i,color in enumerate(colors[:lenf]):
        dummy_lines.append(ax.plot([],[],c = color, label = '%s %s'%(label_base, str(i)))[0])
    ax.set_title('Dynamics colored by mix of colors of the dominant dynamics')
    legend = ax.legend([dummy_lines[i] for i in range(len(dummy_lines))], ['f %s'%str(i) for i in range(len(colors))], loc = 'upper left')
    ax.legend()
        
def plot_subs_effects_2d(F, colors =[['r','maroon','darkred','coral'],['forestgreen','limegreen','darkgreen','springgreen']] , alpha = 0.7 , 
                         rec_rad_all = 5, to_legend = False,
                         n_points = 100,  params_labels = {'title':'sub-dyn effect'}, lw = 4, evec_colors = ['r','g'],
                         include_dyn = False, loc_leg = 'upper left', dx = 'dx',dy = 'dy',
                         axs = [], fig = [],to_revmoe_edges = True):

    if include_dyn:
        fig, axs = plt.subplots(len(F), 3, figsize = (35,8*len(F)),sharey='col', sharex = 'col')
    else:
        if isinstance(axs,list) and len(axs) == 0:
            fig, axs = plt.subplots(len(F), 2, figsize = (30,8*len(F)),sharey='col', sharex = 'col')
            
    if isinstance(colors[0], list):
        [plot_sub_effect(f_i, rec_rad_all , colors[i] , alpha, axs[i,1], n_points, params_labels = {'title':'$f_%s$ effect'%str(i+1)},
                         lw = lw) for i,f_i in enumerate(F)]

    else:
        [plot_sub_effect(f_i, rec_rad_all , colors , alpha, axs[i,1], n_points, params_labels = {'title':'$f_%s$ effect'%str(i+1)}, 
                         lw = lw) for i,f_i in enumerate(F)]
        
    [plot_evals_evecs(axs[i,0], f_i, evec_colors[i] , alpha) for i,f_i in enumerate(F)]
    dummy_lines = []
    dummy_lines.append(axs[0,0].plot([],[], c="black", ls = '--', lw = lw)[0])
    dummy_lines.append(axs[0,0].plot([],[], c="black", ls = '-', lw = lw)[0])
    
    if to_legend:
        legend = axs[0,1].legend([dummy_lines[i] for i in [0,1]], ['Original', 'after sub-dynamic transform'], loc = loc_leg )
        [add_labels(axs_spec, xlabel = dx,ylabel = dy, zlabel = None) for axs_spec in axs[:,1:].flatten()]
        axs[0,1].add_artist(legend)
    if include_dyn:
        [quiver_plot(sub_dyn = f, ax = axs[i,2], chosen_color = evec_colors[i], type_plot='streamplot',cons_color =True,
                     xlabel = dx,ylabel = dy) for i,f in enumerate(F)]
        [axs[i,2].set_title('$f_%s$'%str(i+1), fontsize = 18) for i in range(len(F))]
    if to_revmoe_edges:
        [remove_edges(ax_spec) for ax_spec in axs.flatten()]
    fig.subplots_adjust(wspace = 0.4, hspace = 0.4)
    
    
def auto_compare_synth_simple_to_ground_truth(ground_truth_dict, model_dict): 

    if 'F' in ground_truth_dict and 'F' in model_dict:
        corrs, matches = match_F_compare(ground_truth_dict['F'], 
                                         ground_truth_dict['F'], matches = {}, corrs_mat = [], corrs = [])
        pd.DataFrame(corrs).plot.bar()
        can_adj_c = True    
    else:
        can_adj_c = True    
        print('F not in comparison!')
        print('c cannot be compared due to F')
        
    # reorganize c:
    diff_matches = list(set(np.arange(len(ground_truth_dict['F']))) - set(np.unique(matches.keys()) ))         
    if len(diff_matches) > 0:
       raise ValueError('you must match all F1!')
       
    if 'cs' in ground_truth_dict and 'coefficients' in model_dict and can_adj_c:
       ground_truth_c = ground_truth_dict['cs']
       model_c = model_dict['coefficients']
       reorient = np.arange([matches[key] for key in np.arange(len(ground_truth_dict['F']))])
       if isinstance(model_c, dict):
           model_c = {key:c_i[reorient,:] for key, c_i in model_c.items()}
       elif isinstance(model_c, list):
           model_c = [c_i[reorient,:] for c_i in model_c]
       else:
           raise ValueError('c type is unrecognized')
        



        
def match_F_compare(F1, F2, matches = {}, corrs_mat = [], corrs = {}):
    """
    Compare two lists of features (F1 and F2) and find the best matches between them.
    
    Parameters:
    - F1 (list): List of features representing the ground truth.
    - F2 (list): List of features to be matched against the ground truth.
    - matches (dict, optional): Dictionary specifying initial matches between F1 and F2.
                                Keys represent indices from F1, and values represent indices from F2.
                                Defaults to an empty dictionary.
    - corrs_mat (numpy.ndarray, optional): Precomputed correlation matrix between F1 and F2.
                                           Should be provided as an empty array initially.
                                           Defaults to an empty numpy array.
    - corrs (list, optional): List to store correlation values of matched features.
                             Defaults to an empty list.
    
    Returns:
    - tuple: A tuple containing two elements:
             1. Dictionary containing matched indices (F2 index as key, F1 index as value).
             2. List of correlation values corresponding to the matched features.
    """
    # these are lists
    # F1 is the ground truth and thould be equal in len or shhorter than F2
    # matches  refer to how how should organize F2. e.g. matches = {0:4,1:2,2:1} where the key is F1 index and value F2 index
    if np.sum(corrs_mat) == 0:
        return corrs, matches
    else:
        if checkEmptyList(corrs):
            # mat of len F2 X len F1
            corrs_mat = np.vstack([[np.abs(spec_corr(F1_i, F2_j)) for F1_i in F1] for F2_j in F2])
        argmax_row, argmax_col = np.unravel_index(np.argmax(corrs_mat))
        corrs = {**corrs, **{argmax_col:{argmax_row:corrs_mat[argmax_row, argmax_col] }} }# append(corrs_mat[argmax_row, argmax_col])
        corrs_mat[:,argmax_col ] = 0
        corrs_mat[argmax_row, :] = 0
        matches = {**matches, **{argmax_col:argmax_row}}
        return match_F_compare(F1, F2, matches = matches, corrs_mat = corrs_mat , corrs = corrs)
    

def plot_mid(file_save, name_file, f, error_reco_array_med, coefficients, x, D,  dynamics_type, 
             regions = [], info_keep_order = [],
              sparse_cur_list = [],       data_reco_error_list = [] , latent_dim_per_region = 3, cs_ground_truth = [], max_plots = 5
             ):

    if not os.path.exists(file_save + os.sep + 'coeffs'):
        os.makedirs(file_save + os.sep + 'coeffs')
    if not checkEmptyList(cs_ground_truth):
        
        if not os.path.exists(file_save + os.sep + 'coeffs_tog'):
            os.makedirs(file_save + os.sep + 'coeffs_tog')
    if not os.path.exists(file_save + os.sep + 'coeffs_mat'):
        os.makedirs(file_save + os.sep + 'coeffs_mat')
    if not os.path.exists(file_save + os.sep + 'error'):
        os.makedirs(file_save + os.sep + 'error')      
    if not os.path.exists(file_save + os.sep + 'fs'):
        os.makedirs(file_save + os.sep + 'fs')   
    if not os.path.exists(file_save + os.sep + 'x_mat'):
        os.makedirs(file_save + os.sep + 'x_mat')   
    if not os.path.exists(file_save + os.sep + 'D'):
        os.makedirs(file_save + os.sep + 'D') 
        
        
        
    """
    save D
    """   
    if not checkEmptyList(D):
        
        if isinstance(D, list):
            fig, axs = plt.subplots(1, np.min([ max_plots, len(D)]), figsize = (20,8))
            if len(D) == 1:
                axs = np.array([axs])
            for i, D_i in enumerate(D):  
                if i < max_plots:
                    ax = axs[i]
                
                    heatmap = sns.heatmap(D_i, ax = ax)   
                    if 'multi_reg' in dynamics_type or 'synth_multi_' in dynamics_type:
                        if isinstance(latent_dim_per_region, (list, tuple, np.ndarray)):
                            latent_dim_per_region = latent_dim_per_region[0]
                        
                        if isinstance(info_keep_order, dict):
                            info_keep_order = info_keep_order[list(info_keep_order.keys())[0]]
                        un, count = np.unique(info_keep_order, return_counts=True)#regions

           
                    
                    add_labels(ax, title = 'D', xlabel = 'ensembles (p)', ylabel  = 'Neurons (N)', zlabel = '')
                

            fig.tight_layout()
            fig.savefig(file_save+ os.sep + 'D' + os.sep+ name_file + '_D.png')  
            plt.close()
        else:
            fig, ax = plt.subplots()
            heatmap = sns.heatmap(D, ax = ax)   
            if 'multi_reg' in dynamics_type :
                if isinstance(latent_dim_per_region, (list, tuple, np.ndarray)):
                    latent_dim_per_region = latent_dim_per_region[0]
                
                un, count = np.unique(regions, return_counts=True)
                yticklabels = np.cumsum(count) - count/2
                yticklines = np.cumsum(count)[:-1]
                
                [ax.axvline(i, color = 'pink', alpha = 0.5) for i in range(latent_dim_per_region ,D.shape[1], latent_dim_per_region)]
                [ax.axhline(i, color = 'pink', alpha = 0.5) for i in yticklines]    
                ax.set_yticks([0.5] +  list(yticklabels) + [ D.shape[0]-0.5])
                ax.set_yticklabels([1] + list(info_keep_order) + [D.shape[0]], fontsize = 20)
            # Get the current colorbar
            cbar = heatmap.collections[0].colorbar
            
            # Set the font size, the number of ticks, and their format
            cbar.ax.tick_params(labelsize=14)
            cbar.set_ticks([cbar.vmin, cbar.vmax])
            cbar.set_ticklabels(['{:.1f}'.format(cbar.vmin), '{:.1f}'.format(cbar.vmax)])
            
            add_labels(ax, title = 'D', xlabel = 'ensembles (p)', ylabel  = 'Neurons (N)', zlabel = '')
        
            ax.set_xticks([0.5, D.shape[1]-0.5])
            ax.set_xticklabels([1, D.shape[1]], fontsize = 20)     
            fig.savefig(file_save+ os.sep + 'D' + os.sep+ name_file + '_D.png')    
            plt.close()
                
    """
    save c
    """
    if isinstance(coefficients, list):
        if len(coefficients) == 1:
            fig ,ax = plt.subplots() 
            ax = np.array([ax])
            num_plots = 1
        else:
            num_plots = np.min([max_plots,np.ceil(len(coefficients)/2)])
            fig ,ax = plt.subplots(2, int(num_plots), figsize = (20,7))
            ax = ax.flatten()
        [sns.heatmap(coefficients_i, ax = ax[i])  for i,coefficients_i in enumerate(coefficients) if i < num_plots]
        [add_labels(ax_i, xlabel = 'time', ylabel = 'coefficients %d'%i, title = 'coefficients', zlabel = '')
         for i, ax_i in enumerate(ax)]
        fig.savefig(file_save+ os.sep + 'coeffs_mat' + os.sep+ name_file + '_coeffs_heat.png')
        plt.close()
    else:
        fig ,ax = plt.subplots()
        sns.heatmap(coefficients, ax = ax)
        add_labels(ax, xlabel = 'time', ylabel = 'coefficients', title = 'coefficients heatmap', zlabel = '')
        fig.savefig(file_save+ os.sep + 'coeffs_mat' + os.sep+ name_file + '_coeffs_heat.png')        
        plt.close()
    if not checkEmptyList(cs_ground_truth):    
        if isinstance(coefficients, list) and len(coefficients) != 1:
            fig ,axs = plt.subplots(2, np.min([max_plots,len(coefficients)]),  figsize = (20,10))
            ax = axs[0]
            [sns.heatmap(coefficients_i, ax = ax[i])  for i,coefficients_i in enumerate(coefficients) if i < num_plots]
            [sns.heatmap(cs_ground_truth_i, ax = axs[1][i])  for i,cs_ground_truth_i in enumerate(cs_ground_truth) if i < num_plots]
            [add_labels(ax_i, xlabel = 'time', ylabel = 'coefficients %d'%i, title = 'coefficients', zlabel = '')
             for i, ax_i in enumerate(axs[1])]
            fig.savefig(file_save+ os.sep + 'coeffs_tog' + os.sep+ name_file + '_coeffs_vs_ground.png')
            plt.close()
        else:
            if isinstance(coefficients, list) and len(coefficients) == 1:
                coefficients = coefficients.copy()[0]
            if isinstance(cs_ground_truth, list) and len(cs_ground_truth) == 1:
                cs_ground_truth = cs_ground_truth.copy()[0]
                
            fig ,ax = plt.subplots(2,1)
            sns.heatmap(coefficients, ax = ax[0])
            sns.heatmap(cs_ground_truth, ax = ax[1])
            
            [add_labels(ax_i, xlabel = 'time', ylabel = 'coefficients', title = 'coefficients heatmap', zlabel = '') for ax_i in ax]
            fig.savefig(file_save+ os.sep + 'coeffs_mat' + os.sep+ name_file + '_coeffs_heat.png')        
            plt.close()

    """
    f
    """
    fig, axs = plt.subplots(1, len(f), figsize = (30,5))
    min_f = np.min([-np.max(np.abs(f_i)) for f_i in f])
    max_f = -min_f
    [sns.heatmap(f[i], square=True, robust = True, ax = axs[i], vmin = min_f, vmax = max_f, annot = f[0].shape[0] < 5) for i in range(len(f))]
    [remove_edges(ax) for ax in axs]
    [add_labels(ax, xlabel = 'net', ylabel = 'net', title = 'fs', zlabel = '') for ax in axs     ]

    fig.tight_layout()
    fig.savefig(file_save+ os.sep + 'fs' + os.sep+ name_file+ '_fs.png')
    plt.close()

    """
    error
    """
    if len(error_reco_array_med) > 1:
        fig ,ax = plt.subplots(1,3, figsize = (20,5))
              
        ax[0].plot(error_reco_array_med[1:])
        ax[2].plot(sparse_cur_list)
        ax[1].plot( data_reco_error_list[1:] )
        names = ['error_dynamics', 'error_obs', 'sparsity mean']
        [add_labels(ax_i, ylabel = names[i],
                   xlabel = 'iterations',
                   title = 'error over time', zlabel = '') for i, ax_i in enumerate(ax)]
    
        [remove_edges(ax_i, left = True, bottom = True, include_ticks = True)
         for i, ax_i in enumerate(ax)]
        fig.tight_layout()
        fig.savefig(file_save + os.sep + 'error' + os.sep+ name_file+ '_error.png')
        plt.close()
    
    
    """
    coeffs
    """
    if isinstance(coefficients, list) and len(coefficients) != 1:
        num_plots = np.min([max_plots,   int(np.ceil(len(coefficients)/4)) ])
        fig ,ax = plt.subplots(4, num_plots, figsize = (30,7))
        ax = ax.flatten()
        [ax[i].plot(coefficients_i.T)  for i,coefficients_i in enumerate(coefficients)  if i < num_plots]
        [add_labels(ax_i, xlabel = 'time', title = 'coefficients %d'%i, ylabel = '', zlabel = '') 
         for i, ax_i in enumerate(ax)]
        fig.tight_layout()
        fig.savefig(file_save+ os.sep + 'coeffs' + os.sep+ name_file + '_coeffs_heat.png')
    else:
        fig ,ax = plt.subplots()
        if isinstance(coefficients, list) and len(coefficients) == 1:
            coefficients = coefficients.copy()[0]
        ax.plot(coefficients.T)
        add_labels(ax, xlabel = 'time', ylabel = 'coefficients', title = 'coefficients', zlabel = '')
        fig.savefig(file_save+os.sep + 'coeffs'+ os.sep+ name_file +'_coeffs.png')
        plt.close()
 
    if isinstance(x, list) and len(x) != 1:
        num_plots = np.min([max_plots,int(np.ceil(len(coefficients)/3))])
        fig ,ax = plt.subplots(3, num_plots , figsize = (30,7))
        ax = ax.flatten()
        [sns.heatmap(x_i, ax = ax[i])  for i,x_i in enumerate(x)  if i < num_plots*3]
        [add_labels(ax_i, xlabel = 'time', title = 'coefficients %d'%i, ylabel = '', zlabel = '')
         for i, ax_i in enumerate(ax)]
        fig.savefig(file_save+ os.sep + 'x_mat' + os.sep+ name_file + '_x_heat.png')
        plt.close()
        
        if x[0].shape[0] == 3:
            fig, axs = plt.subplots( np.min([max_plots,len(x)]), 1, sharex = True, sharey = True, figsize = (8,17), 
                                    subplot_kw={'projection': '3d'})
            [plot_3d(x_i,  ax = axs[i])             for i, x_i in enumerate(x)  if i < num_plots]
            shapes = [x_i.shape for i, x_i in enumerate(x)]
            max_len = np.max([shapes[i][1] for i in range(len(shapes))])
           
            [add_labels(ax, ylabel = 'sub-dyn', xlabel = '', zlabel = '', title = 'session %d'%(i+1)) 
             for i,ax in enumerate(axs)]
            add_labels(axs[-1], ylabel = '', zlabel = '',  xlabel = 'Time', title = '') 
    
            plt.suptitle('x', fontsize = 30)
            fig.tight_layout()
            plt.savefig(file_save+ os.sep + 'x_mat' + os.sep+ name_file + '_x_3d.png')
            plt.savefig(file_save+ os.sep + 'x_mat' + os.sep+ name_file + '_x_3d.png')    
            plt.close()
        
        
    else:
        fig ,ax = plt.subplots()
        if isinstance(x, list) and len(x) == 1:
            x = x.copy()[0]
        sns.heatmap(x, ax = ax)
        add_labels(ax, xlabel = 'time', ylabel = 'x', title = 'x heatmap', zlabel = '')
        fig.savefig(file_save+ os.sep + 'x_mat' + os.sep+ name_file + '_x_heat.png')
        plt.close()
    

from sklearn.decomposition import PCA


def apply_PCA_per_region(y, p, indices_regs_session = [] , PCA_type = 'local'):
    if not isinstance(p, (list, np.ndarray)):
        p = [p]*len(indices_regs_session)
    if PCA_type.lower() == 'local':
  
        if len(indices_regs_session) == 0:
            indices_regs = [np.arange(y.shape[0])]
        

        if y.shape[1] < np.sum(p):
            raise ValueError('y is already low dim')

        new_D = np.zeros((y.shape[0], np.sum(p)))
 
        transformed_region = []     
        if y.ndim != 2:
            raise ValueError("Each input array must be 2-dimensional")
    
            
        p_sum = 0
        row_sum = 0
        for counter, area_indices in enumerate(indices_regs_session): 
    

            begin_col = p_sum
            end_col = begin_col + p[counter]

            if len(area_indices) > 0:
                pca = PCA(n_components = np.min([p[counter],len(area_indices) ]))

                
                p_sum += p[counter]
                y[area_indices,: ].T
                comps = pca.fit(y[area_indices,: ].T).components_.T

           
                new_D[indices_regs_session[counter],begin_col:begin_col + comps.shape[1]] = comps
                #transformed_region.append(y_pca) # now I have components for each region
    elif PCA_type.lower() == 'global':
        print('pay attention that you must update the latent dyn and remove the d_mask!')
        p_new = np.sum([p_i  if len(el) > 0 else 0 for p_i, el in zip(p, indices_regs_session)])
        pca = PCA(n_components = p_new)
        print(y.shape)
        new_D = pca.fit(y.T).components_.T
    else:
        raise ValueError('undefined PCA_type. Must be either global or local but PCA_type = %s'%PCA_type)
    return new_D


    

def infer_x_with_sibblings_only(y, D, x_former = [], params_infer_x_no_prior = {'lambda_frob': 0.1 , 'lambda_smooth_iters': 0 , 'lambda_smooth_time': 0.1 , 'lambda_decor': 0.1}):
    # Check if y, D are lists or a single session
    if params_infer_x_no_prior.get('lambda_smooth_iters',0) != 0 and checkEmptyList(x_former):
        raise ValueError('you must provide x_former if smooth over iterations. ')
    
    if isinstance(y, list) or isinstance(D, list):
        if not isinstance(x_former, list):
            raise ValueError('x_former must be a list in this case!')
            
        x = []
        for i, (y_i, D_i)  in enumerate(zip(y, D)):    
            if len(x_former) > 0:
                x_hat = infer_x_with_sibblings_only(y_i, D_i, x_former = x_former[i])
            else: 
                x_hat = infer_x_with_sibblings_only(y_i, D_i, x_former = x_former)
                
            x.append(x_hat)
    else:        
        if y.shape[0] != D.shape[0]:
            raise ValueError('unmatched shapes. y shape is %s while D shape is %s'%(str(y.shape), str(D.shape)))
        # Infer x for a single trial
        # solve w1* \| y - Dx \|_F^2 + w2*
        
        p = D.shape[1]
        T = y.shape[1]
        
        #left (i.e. the y side)
        left = []
        
        # right (i.e. the matrix factorization side)
        right = []
        
        # data fidelity
        w1 =  1
        cost1_left = w1*y
        cost1_right = w1*D
        
        left.append(cost1_left)
        right.append(cost1_right)
        
        # frobenious
        w2 = params_infer_x_no_prior.get('lambda_frob',0)
        if w2 > 0:
            cost2_left = w2*np.zeros((p,T))
            cost2_right = w2*np.eye(p)
            left.append(cost2_left)
            right.append(cost2_right)
        
        
        
        # lambda_smooth_iters
        w3 = params_infer_x_no_prior.get('lambda_smooth_iters',0)

        if w3 > 0:        
            cost3_left = w3*x_former
            cost3_right =  w3*np.eye(p)       
            left.append(cost2_left)
            right.append(cost2_right)
        
        
        #lambda_smooth_time
        w4 = params_infer_x_no_prior.get('lambda_decor',0)
        if w4 > 0:
            
            
            if not checkEmptyList(x_former):
                nonself_corr = np.sqrt(np.sum(x_former**2,1).reshape((1,-1)))

            else:
                nonself_corr = np.ones((1,p))
            self_corr = np.diag(nonself_corr.flatten())

            corrs = np.vstack([self_corr, nonself_corr])
            
            cost4_left =  np.zeros((1 + p, T))
            cost4_right = w4*corrs
            left.append(cost4_left)
            right.append(cost4_right)
        
        
        
    
        left = np.vstack(left)

        right = np.vstack(right)
        if params_infer_x_no_prior.get('lambda_smooth_time',0) == 0:
            #lambda_smooth_time

            x = np.linalg.pinv(right) @ left
        
            
        else:
            w5 = params_infer_x_no_prior.get('lambda_smooth_time',0)
            

            x = []
            for t in range(T):
                if t == 0:
                    xt = np.linalg.pinv(right) @ left[:,t].reshape((-1,1))
                    x.append(xt.reshape((-1,1)))
                else:
                    right_loc = np.vstack([right, w5*np.eye(p) ])
                    left_loc = np.vstack([left[:,t].reshape((-1,1)), w5*x[-1] ])
                    
                    xt = np.linalg.pinv(right_loc) @ left_loc
                    x.append(xt.reshape((-1,1)))
                    


            x = np.hstack(x)
        
    return x
    
    
    
    
        
    
def plot_subs(F, axs = [],params_F_plot = {'cmap':'PiYG'}, include_sup = True,annot = True):
  """
  This function plots heatmaps of the sub-dynamics
  """
  params_F_plot = {**{'cmap':'PiYG'},**params_F_plot}
  if isinstance(axs,list):
    if len(axs) == 0:
      fig, axs = plt.subplots(1,len(F), sharex = True,sharey = True)

  [sns.heatmap(f_i, ax = axs[i],annot=annot, **params_F_plot) for i,f_i in enumerate(F)]
  [ax.set_title('f#%g'%i) for i,ax in enumerate(axs)]
  if include_sup: plt.suptitle('Sub-Dynamics')
  plt.subplots_adjust(hspace = 0.5,wspace = 0.5)



def spec_corr(v1,v2, to_abs = True):
  """
  absolute value of correlation
  """
  corr = np.corrcoef(v1[:],v2[:])
  if to_abs:
      return np.abs(corr[0,1])
  return corr[0,1]



def update_D(former_D, step_D , x, y, reg1 = 0, reg_f= 0, bias_out_val = [], enable_D_inverse = False, 
             try_different_sizes = True, ratio_min = 1/100, ratio_max = 50, num_steps = 10, D_min = 0.0001) :
  """
  Update the matrix D by applying GD. Relevant just in case where D != I
  data_reco_error_list
  """
  if step_D < D_min:
      step_D = D_min
  if len(bias_out_val) == 0: bias_out_val = np.zeros((former_D.shape[0], 1))
  if isinstance(bias_out_val[0], np.ndarray): 
      bias_out_val = bias_out_val[0]
  if reg1 == 0 and reg_f ==0 and  enable_D_inverse:
    D = y @ linalg.pinv(x)
    print('D solved with inverse.')
  else:

    basic_error = -2*(y - former_D @ x ) @ x.T 
    or_error = np.mean((y - former_D @ x )**2) #
    basic_error_or = basic_error 
    if reg1 != 0:      reg1_error = np.sum(np.sign(former_D))
    else: reg1_error = 0      
    if reg_f != 0:      reg_f_error = 2*former_D
    reg_f_error = 0

    D = former_D - step_D *(basic_error + reg1*reg1_error + reg_f* reg_f_error)
    error = np.mean((y - D @ x )**2) 
    
    if error > or_error :
        print('error D increased')
        new_Ds = []
        errors = []
        optional_steps = np.linspace( ratio_min*step_D, ratio_max*step_D , num_steps)
        
        for step_D_i in optional_steps:
            #basic_error = -2*(y -_D @ x ) @ x.T
            D_i = former_D - step_D_i *(basic_error + reg1*reg1_error + reg_f* reg_f_error)
            error = np.mean((y - D_i @ x )**2) 
            errors.append(error)
            new_Ds.append(D_i)
            
        errors_min = np.argmin(errors)
 
        D = new_Ds[errors_min]
        step_D = optional_steps[errors_min]

  return D, step_D

def update_X(D, data, latent_dyn = [], type_x_infer = 'nls', lambda_x = 100, F = [], coefficients = [], 
             random_state = 0, params_x = {}, 
             counter = 0, direction_update = 1, x0 = [], 
             use_new_est = True, use_both_obs_and_latent = 'latent' , w_f = 0.1):  

  """
  Update the latent dynamics. Relevant just in case where D != I
  """
  if use_both_obs_and_latent == 'alternate' :
      if np.mod(counter, 2) == 0:
          use_both_obs_and_latent = 'obs'
      else:
          use_both_obs_and_latent == 'latent'
  params_x  = {**{'weight_observations':6}, **params_x }
  if counter != 0 and (checkEmptyList(F) or checkEmptyList(coefficients) ):
      raise ValueError('you must provide Fc or latent_dyn if counter is not 0')
                       

  next_mat = data
  mul_mat = D
 

  if not checkEmptyList(x0) and  not checkEmptyList(latent_dyn):
    latent_dyn[:,0] = x0
      
  if type_x_infer not in ['nls', 'inv'] and use_both_obs_and_latent == 'obs':
      raise ValueError('TODO!')
      
  if ((lambda_x == 0 or type_x_infer == 'inv') and counter != 0) or  use_both_obs_and_latent == 'obs':
    p = mul_mat.shape[1]
    T = next_mat.shape[1]
    mul_mat_new = np.vstack([ mul_mat, w_f*np.eye(p)  ])  
    next_mat_new = np.vstack([  next_mat, np.zeros((p, T  )) ])
    x = linalg.pinv(mul_mat_new) @ next_mat_new
    if checkEmptyList(x0):
      x0 = x[:,0]
    else:
      x[:,0] = x0
      
      
  else:

    Fc_list = [np.sum(np.dstack(
        [F[i]*coefficients[i,t]  for i in range(len(F))]
        ),2)
        for t in range(coefficients.shape[1])            
        ]


    
    """
    take the last element of x and construct from the left
    """
    if direction_update == -1:
        x_former = solve_lasso_problem(data[:,-1], D, lambda_x,  type_x_infer, random_state = 0, 
                            params_update_c  = params_x).reshape((-1,1)) 
        x_former_former = x_former.copy()
        for t in np.arange(data.shape[1]-1)[::-1]:
            if use_new_est:
                vector_x = x_former_former 
            else:
                vector_x = latent_dyn[:,t+1].reshape((-1,1))
            x_former_former = solve_lasso_problem( np.vstack([data[:,t].reshape((-1,1)) , 
                                                              vector_x ]).flatten(), 
                                np.vstack([D, Fc_list[t]]), lambda_x,  type_x_infer, random_state = 0, 
                                params_update_c  = params_x).reshape((-1,1)) 
            
            x_former = np.hstack([x_former_former , x_former])
            
            
            
        x = x_former
      
        if checkEmptyList(x0):
          x0 = x[:,0]
        else:
          x[:,0] = x0
          
    elif direction_update == 1:
        if checkEmptyList(x0):
          x_hat = linalg.pinv(D) @ data[:,0].reshape((-1,1))

        else:
          x_hat = x0.reshape((-1,1))  
          
        x = x_hat.reshape((-1,1))  
        for t in range(1, data.shape[1] ):
            if use_new_est:
                vector_x = x_hat
            else:
                vector_x =latent_dyn[:,t - 1].reshape((-1,1))
            if use_both_obs_and_latent == 'both': 

                
                x_hat = solve_lasso_problem( np.vstack([data[:,t].reshape((-1,1)) , 
                                                                  vector_x ]).flatten(), 
                                    np.vstack([D, np.linalg.pinv(Fc_list[t-1] ) ]), lambda_x,  type_x_infer, random_state = 0, 
                                    params_update_c  = params_x).reshape((-1,1)) 
                
            elif use_both_obs_and_latent == 'obs': 
               
                raise ValueError('should not arrive here. sanity check')
                
            elif use_both_obs_and_latent == 'latent': 
                x_hat = Fc_list[t-1] @ vector_x.reshape((-1,1))
            else:
                raise ValueError('use_both_obs_and_latent is not valid! %s'%use_both_obs_and_latent )
                
        
            x = np.hstack([x, x_hat])
    else:
        raise ValueError('direction_update must be 1 or -1')
  return x    
        
            
            



def check_F_dist_init(F, max_corr = 0.1):
    """
    This function aims to validate that the matrices in F are far enough from each other
    """
    combs = list(itertools.combinations(np.arange(len(F)),2))
    corr_bool = np.array([spec_corr(F[comb_s[0]],F[comb_s[1]]) > max_corr for comb_s in combs])
    counter = 100
    while (corr_bool == False).any():
        counter +=1
        for comb_num,comb in enumerate(combs):
            if spec_corr(F[comb[0]],F[comb[1]])  > max_corr:
                fi_new = init_mat(np.shape(F[0]),dist_type = 'norm',r_seed = counter)
                F[comb[0]] = fi_new
    return F
        
def create_ax(ax, nums = (1,1), size = (10,10), proj = 'd2',return_fig = False,sharey = False, sharex = False, fig = []):
  
    if isinstance(ax, list) and len(ax) == 0:
        
        if proj == 'd2':
            fig,ax = plt.subplots(nums[0], nums[1], figsize = size, sharey = sharey, sharex = sharex)
        elif proj == 'd3':
            fig,ax = plt.subplots(nums[0], nums[1], figsize = size,subplot_kw={'projection':'3d'}, sharey = sharey, sharex = sharex)
        else:
            raise NameError('Invalid proj input')
        if return_fig:
            return fig, ax

    if  return_fig :
        return fig, ax
    return ax


def perc_to_nulify_in_block_mat(block_mat, perc_or = 80, num_ens_per_region = 2, num_regions = 3):
    if not isinstance(num_ens_per_region, (list, tuple, np.ndarray)):
        num_ens_per_region  = np.array([num_ens_per_region ]*num_regions)
    if block_mat.shape[1] != np.sum(num_ens_per_region):
        raise ValueError('error in mat dim')
   
    mean_reg = np.mean(num_ens_per_region)
    non_block_size = block_mat.shape[1] - mean_reg
   
    return (non_block_size/block_mat.shape[1] + perc_or/100*mean_reg/block_mat.shape[1])*100
    

def nullify_part(f,axis = 'both', percent0 = 80):
    if not isinstance(axis, str): axis = str(axis)
    if axis == 'both':
        f[f < np.percentile(np.abs(f), percent0)] = 0
    elif axis == '0':
        perc = np.percentile(np.abs(f), percent0, axis = 0)
        for col in range(f.shape[1]):
            f[f[:,col] < perc[col],col] =0
    elif axis == '1':
        perc = np.percentile(np.abs(f), percent0, axis = 1)
        for row in range(f.shape[0]):
            f[row,f[row,:] < perc[row]] =0
    return f

def create_orth_F(num_subdyns, num_neurons, evals = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], seed_f = 0 , 
                   dist_type = 'random' ):
    if num_neurons > len(evals): evals = evals + [0]*(num_neurons - len(evals))
    np.random.seed(seed_f)
    orth_mats = [np.linalg.qr(np.random.rand(num_neurons,num_neurons))[0]
                 for num_subdyn in range(num_neurons)]
    evecs = [np.hstack([orth_mat[:,i].reshape((-1,1)) 
                        for orth_mat in orth_mats]) for i in range(num_neurons)]
    F = [evec @ np.diag(evals[:evec.shape[0]]) @ np.linalg.pinv(evec) for i, evec in enumerate(evecs)]
    np.random.seed(seed_f)
    if len(F)< num_subdyns:
        print('Only %d sud-dyns are  orthogonal')
        if dist_type == 'random' :
            F2 = [np.random.randn(num_neurons,num_neurons) for f_num in range(num_subdyns - len(F) ) ]
        else:
            raise ValueError('Unknown dist type')
        F =  F + F2
    return F[:num_subdyns]
    
def update_c_full(F, cur_reco, params, latent_dyn,  seed,
                        other_params_c, include_identity, one_dyn, same_c, acumulated_error, add_avg, wind_avg,
                        coefficients): #add here previous coeffs 
    if one_dyn:
      if acumulated_error:
        coefficients = update_c(F, cur_reco, params, clear_dyn= latent_dyn, direction = 'n2c',random_state=seed,
                                other_params=other_params_c, include_identity =include_identity , coefficients=coefficients)
      else:
        coefficients = update_c(F,latent_dyn, params,random_state=seed,other_params=other_params_c,
                                include_identity =include_identity, coefficients=coefficients )
      if add_avg:
          coefficients = np.hstack([np.mean(coefficients[:, np.max([0,t - wind_avg ]): np.min([coefficients.shape[1], t + wind_avg ])], 1).reshape((-1,1))
                                    for t in np.arange(coefficients.shape[1])
                                    ]) 
      
    else:
      if same_c:
        if acumulated_error:
          coefficients = np.mean(np.dstack([update_c(F, cur_reco[:,:,i], params, clear_dyn= latent_dyn, 
                                                     direction = 'n2c',other_params=other_params_c, include_identity = include_identity,coefficients=coefficients) for i in range(cur_reco.shape[2])]),2)
        else:
          coefficients = np.mean(np.dstack([update_c(F,latent_dyn[i], params,other_params=other_params_c,
                                                     include_identity = include_identity,coefficients=coefficients ) 
                                            for i in range(cur_reco.shape[2])]),2)
      else:
        if acumulated_error:
          coefficients = [update_c(F, cur_reco[i], params, clear_dyn= latent_dyn, direction = 'n2c',other_params=other_params_c,
                                   include_identity = include_identity,coefficients=coefficients[i]) for i in range(len(cur_reco))] # future - fix latent dyn here
        else:
          try:
              coefficients = [update_c(F,latent_dyn[i], params,other_params=other_params_c, 
                                       include_identity = include_identity,
                                       coefficients=coefficients[i])
                              for i in range(len(cur_reco))]
          except:
              F = check_values(F, max_val = 10**3, min_val = -10**3, rep_nan = True, rep_0 = True, noise_0 = 0.2)
              F = [f + np.random.rand(*f.shape)*0.1 for f in F]
              
              latent_dyn = check_values(latent_dyn, max_val = 10**3, min_val = -10**3, rep_nan = True, rep_0 = True, noise_0 = 0.2)
              latent_dyn = [latent_dyn_i + np.random.rand(*latent_dyn_i.shape)*0.1 for latent_dyn_i in latent_dyn]
              
              
              coefficients = check_values( coefficients, max_val = 10**3, min_val = -10**3, rep_nan = True, rep_0 = True, noise_0 = 0.2)
              coefficients = [coefficients_i + np.random.rand(*coefficients_i.shape)*0.1 for coefficients_i in coefficients]
              
              
  
              coefficients = [update_c(F,latent_dyn[i], params,other_params=other_params_c, 
                                       include_identity = include_identity,
                                       coefficients=coefficients[i])
                              for i in range(len(cur_reco))]
          
              
              
             
    return coefficients

def update_f_with_mat(latent_dyn, coefficients,  dur_update = 3):
    
    print(' update_f_with_mat')
    # M is the number of subdyns
    M = coefficients.shape[0]
    #print(latent_dyn.shape)
    max_T_update = latent_dyn.shape[1] - dur_update    

    rand_start = np.random.randint(0, max_T_update - 1)
    rand_end = rand_start  + dur_update
    
    x = latent_dyn[:,rand_start:rand_end]
    coefficients = coefficients[:,rand_start:rand_end- 1]
    x_plus = x[:,1:]
    x_minus = x[:, :-1]
    dim = latent_dyn.shape[0]
    x_minus_rep = np.tile(x_minus, (M, 1))

    cs_reps = np.repeat(coefficients, dim, 0)

    x_cs = x_minus_rep * cs_reps

    hstacked_F = x_plus @ np.linalg.pinv(x_cs)
    edges = np.linspace(0,hstacked_F.shape[1], M + 1).astype(int)
  
    F = [hstacked_F[:,edges[j]:edges[j+1]] for j in range(M)]
    return F
    
                             
    
def update_f_apr():
    pass
    
    
    
    
def update_f(latent_dyn,F,coefficients,step_f, acumulated_error, 
               error_order, action_along_time, weights_orders,                
              normalize_eig ,  bias_val,  include_identity,  type_norm ,
                min_step_f, count_gd_step , one_dyn, decay_decay =  False, GD_decay= 0.99, counter = 0, 
                num_subdyns = 0, same_c = True, size_batch = 0.5, latent_dyns = [], 
                ratio_min = 1/20, ratio_max = 20, wise_step = True, num_steps = 5,dur_update = 3, only_dec_f = True, update_by_mat = False):
    print(' update_f')
    if step_f < min_step_f:  
        step_f = min_step_f
    if  acumulated_error:
        type_reco ='lookahead'
    else:
        type_reco = 'step'
        
    if update_by_mat:
        count_gd_step += 1
        if one_dyn:
            x_stack = latent_dyn
            c_stack = coefficients
            F = update_f_with_mat( x_stack, c_stack,  dur_update = dur_update)
            return F, step_f, count_gd_step
        else:
            if size_batch < 1: 
                size_batch = np.max([1, int(np.floor(len(latent_dyns)*size_batch))])
            
                samples_take = np.random.choice(len(latent_dyns), size_batch, replace=False)
            F_list = []
            for samp in  samples_take:
                x_stack = latent_dyns[samp]
                c_stack = coefficients[samp]

                F_list.append(update_f_with_mat( x_stack, c_stack,  dur_update = dur_update))
            F = [np.mean(np.dstack([ F[j]  for F in F_list]),2) for j in range(len(F_list[0]))]
            return  F, step_f, count_gd_step
        
        
        
        
    elif one_dyn:        
        F = update_f_all(latent_dyn,F,coefficients,step_f,normalize=False, acumulated_error = acumulated_error, 
                       error_order = error_order-1, action_along_time= action_along_time, 
                       weights = weights_orders, 
                       normalize_eig = normalize_eig , bias_val = bias_val, 
                       include_identity = include_identity,  type_norm =  type_norm, dur_update = dur_update )
    elif wise_step:

        np.random.seed( int(str(datetime2.now()).split('.')[-1]))
        if checkEmptyList(latent_dyns):
            raise ValueError('latent_dyns is not defined!')
        if size_batch < 1: 
            size_batch = np.max([1, int(np.floor(len(latent_dyns)*size_batch))])
        
            samples_take = np.random.choice(len(latent_dyns), size_batch, replace=False)
            
            
        optional_steps = np.linspace( ratio_min*step_f, ratio_max*step_f , num_steps)    
    
        latent_dyns = latent_dyn.copy()  
        errors = []
        F_or = F.copy()
        F_store = {}
        step_f_or = step_f
       
        error_F = np.mean([np.median((latent_dyn  - create_reco_new(latent_dyn, coefficients[i], F_or, type_reco))**2)
                   for i, latent_dyn in enumerate(latent_dyns) if i in samples_take])
      
        for count_step, step_f in enumerate(optional_steps):
            F = F_or.copy()
            
            if same_c:
                
                
                for i in samples_take:
                    F = update_f_all(latent_dyns[i],F,coefficients,step_f,normalize=False, acumulated_error = acumulated_error, 
                                            error_order = error_order-1, action_along_time= action_along_time, weights = weights_orders,
                                            bias_val = bias_val, include_identity =  include_identity,   type_norm =  type_norm ,dur_update = dur_update)
                    
                        

            else: 

              for i in samples_take:
              
                  F = update_f_all(latent_dyns[i],F,coefficients[i],step_f,normalize=False, acumulated_error = acumulated_error,
                                    error_order = error_order-1, action_along_time= action_along_time, weights = weights_orders, 
                                    bias_val =[],   include_identity =  include_identity,   type_norm =  type_norm,dur_update = dur_update) 
            F_store[count_step]      = F
            error_F = np.mean([np.nanmedian((latent_dyn  - create_reco_new(latent_dyn, coefficients[i], F, type_reco))**2)
                       for i, latent_dyn in enumerate(latent_dyns) if i in samples_take])
                       
                       
            errors.append(error_F)

       
        print('-----------===============')
        print(np.array(errors)*100 )
        print(np.array(errors)*100 - error_F*100)        

        print('errors above')
        print('-----------===============')

        if (not only_dec_f ) or ((np.array(errors)- error_F) < 0).any(): 
            best_F_arg = np.argmin(errors)

            F = F_store[best_F_arg]

            step_f = optional_steps[best_F_arg]
        else:
            F = [f + np.random.rand(*f.shape)*0.1 for f in F]
            print('did not update!!')
        
    else:
        np.random.seed( int(str(datetime2.now()).split('.')[-1]))
        if checkEmptyList(latent_dyns):
            raise ValueError('latent_dyns is not defined!')
        if size_batch < 1: 
            size_batch = int(np.floor(len(latent_dyns)*size_batch))
            samples_take = np.random.choice(len(latent_dyns), size_batch, replace=False)
        # NMNM - put seed here in the future     
        latent_dyns = latent_dyn.copy()  

        if same_c:
            
            
            for i in samples_take:
                F = update_f_all(latent_dyns[i],F,coefficients,step_f,normalize=False, acumulated_error = acumulated_error, 
                                        error_order = error_order-1, action_along_time= action_along_time, weights = weights_orders,
                                        bias_val = bias_val, include_identity =  include_identity,   type_norm =  type_norm)
                
                    

        else: 
          for i in samples_take:
              F = update_f_all(latent_dyns[i],F,coefficients[i],step_f,normalize=False, acumulated_error = acumulated_error,
                                error_order = error_order-1, action_along_time= action_along_time, weights = weights_orders, 
                                bias_val = bias_val[i],  include_identity =  include_identity,   type_norm =  type_norm) 
 
    if step_f > min_step_f:    
        if decay_decay and counter !=0:
            step_f = step_f*(GD_decay**(1/np.sqrt(counter)))
        else:
            step_f *= GD_decay
                
    count_gd_step += 1
    
    if one_dyn:
        return F, step_f, count_gd_step
    else:
        return F, step_f, count_gd_step
    
    
def update_f_depracated(latent_dyn,F,coefficients,step_f, acumulated_error, 
               error_order, action_along_time, weights_orders,                
              normalize_eig ,  bias_val,  include_identity,  type_norm ,
                min_step_f, count_gd_step , one_dyn, decay_decay =  False, GD_decay= 0.99, counter = 0, num_subdyns = 0, same_c = True):

    if one_dyn:
        
        F = update_f_all(latent_dyn,F,coefficients,step_f,normalize=False, acumulated_error = acumulated_error, 
                       error_order = error_order-1, action_along_time= action_along_time, 
                       weights = weights_orders, 
                       normalize_eig = normalize_eig , bias_val = bias_val, 
                       include_identity = include_identity,  type_norm =  type_norm  )
    
    else:
        latent_dyns = latent_dyn.copy()  
        if same_c:
          F_lists = [update_f_all(latent_dyns[i],F,coefficients,step_f,normalize=False, acumulated_error = acumulated_error, 
                                  error_order = error_order-1, action_along_time= action_along_time, weights = weights_orders,
                                  bias_val = bias_val, include_identity =  include_identity,   type_norm =  type_norm)
                     for i in range(len(latent_dyns))]
          store_F = np.zeros((latent_dyns[0].shape[0], latent_dyns[0].shape[0],num_subdyns))
          for F_list in F_lists:
            store_F = store_F + np.dstack(F_list)
          store_F = store_F / len(F_lists)
          F = list(store_F.T)
        else: 
          F_lists = [update_f_all(latent_dyns[i],F,coefficients[i],step_f,normalize=False, acumulated_error = acumulated_error,
                                  error_order = error_order-1, action_along_time= action_along_time, weights = weights_orders, 
                                  bias_val = bias_val[i],  include_identity =  include_identity,   type_norm =  type_norm) 
                     for i in range(len(latent_dyns))]
          store_F = np.zeros((latent_dyns[0].shape[0], latent_dyns[0].shape[0],num_subdyns))
          for F_list in F_lists:
            store_F = store_F + np.dstack(F_list)
          store_F = store_F / len(F_lists)
          F = list(store_F.T)    
    if step_f > min_step_f:    
        if decay_decay and counter !=0:
            step_f = step_f*(GD_decay**(1/np.sqrt(counter)))
        else:
            step_f *= GD_decay
                
    count_gd_step += 1
    
    if one_dyn:
        return F, step_f, count_gd_step
    else:
        return F_lists, step_f, count_gd_step
    
def keep_max_vec(vec,k, return_indices = False):
    """
    Keep the k largest values in the vector and set the rest to zero.
    
    Args:
        vec (ndarray): The input vector.
        k (int or float): The number of largest values to keep. If float, it represents a ratio of the vector length.
    
    Returns:
        ndarray: The vector with the k largest values preserved and the rest set to zero.
    """
    vec = vec.flatten()
    if k < 1:
        k = int(k*len(vec))
    lp = np.sort(np.abs(vec))[-k]
    vec[np.abs(vec) < lp] = 0
    if return_indices:
        return vec, np.where(vec != 0)[0]
    return vec

def keep_only_first_last_ticklabels(ax, xticklabls = [], yticklabels = [],  xticks = [], yticks = [],
                                    fontsize = 14, apply_to_x = True, apply_to_y = True):
    """
    Modify tick labels and tick positions on a matplotlib axis to show only the first and last ticks.
    
    Parameters:
    ax (matplotlib.axes._subplots.AxesSubplot): The matplotlib axis to be modified.
    xticklabels (list, optional): Custom tick labels for the x-axis. If empty, default tick labels will be used.
    yticklabels (list, optional): Custom tick labels for the y-axis. If empty, default tick labels will be used.
    xticks (list, optional): Custom tick positions for the x-axis. If empty, default tick positions will be used.
    yticks (list, optional): Custom tick positions for the y-axis. If empty, default tick positions will be used.
    fontsize (int, optional): Font size for the tick labels. Default is 14.
    apply_to_x (bool, optional): If True, modify tick labels and positions on the x-axis. Default is True.
    apply_to_y (bool, optional): If True, modify tick labels and positions on the y-axis. Default is True.
    
    Returns:
    matplotlib.axes._subplots.AxesSubplot: The modified matplotlib axis.
    """
    if checkEmptyList(xticks):
        xticks = ax.get_xticks()
    if checkEmptyList(yticks):
        yticks = ax.get_yticks()
    if checkEmptyList(xticklabls):
        xticklabels = ax.get_xticks()
        xticklabels[1:-1] = ''
    if checkEmptyList(yticklabels):
        yticklabels = ax.get_yticks()
        yticklabels[1:-1] = ''
    if apply_to_x:
        ax.set_xticks([xticks[0], xticks[1]])
        ax.set_xticklabels(xticklabels)
    if apply_to_y:
        ax.set_yticks([yticks[0], yticks[1]])
        ax.set_yticklabels(yticklabels)
        
    
    return ax
        
    
    
    
    
    

    

def apply_hard_thres(mat, axis = 0, k = 0.5):
    """
    Apply hard thresholding to a matrix along a specified axis.

    Args:
        mat (ndarray): The input matrix.
        axis (int, optional): The axis along which to apply the thresholding. Default is 0.
        k (int or float, optional): The number of largest values to keep. If float, it represents a ratio of the matrix length.

    Returns:
        ndarray: The matrix with the hard thresholding applied.

    Raises:
        ValueError: If axis is not 0, 1,2 or -1.
    """
    if np.max(mat.shape) == len(mat.flatten()):
        return keep_max_vec.reshape(mat.shape)
        
    if axis == 0:
        if k > mat.shape[0]:
            k = mat.shape[0] - 1
        if k < 1:
            k =int( k*mat.shape[0])
        return np.hstack(
            [keep_max_vec(mat[:,i], k ).reshape((-1,1)) for i in range(mat.shape[1])]
            )
    elif axis == -1:
        lp = np.sort(np.abs(mat).flatten())[-k]
        mat[np.abs(mat) < lp] = 0
        return mat
    elif axis == 2:
        
        return np.transpose(np.dstack([apply_hard_thres(mat[i,:,:].T, axis = 0, k = k)
                             for i in range(mat.shape[1])]), [2,1,0])
    elif axis == 1:
        return apply_hard_thres(mat.T, axis = 0, k = k).T
    else:
        raise ValueError('axis must be 1, or 0, or -1')
    
def check_1d(mat):        
    return np.max(mat.shape) == len(mat.flatten())
    
def check_complete_zero(coeffs, params_zero_window = {}):
    """
    Check and replace wide windows with no coefficients in the given array.
    
    Args:
        coeffs (numpy.ndarray): Array of coefficients to check.
        params_zero_window (dict, optional): Parameters for zero window checking. Defaults to {}.
    
    Returns:
        numpy.ndarray: Array with replaced zero windows.
    
    """
    # check that there are not wide windows with no coeffs
    params_zero_window = {**{'width': 20, 'freq_fill':30, 'overlapp':5, 'zero_thres':1e-7}, **params_zero_window}
    winds_initials = np.arange(0, coeffs.shape[1] - params_zero_window['width'], params_zero_window['overlapp'])
    winds_ends = winds_initials +  params_zero_window['width']
    winds = [coeffs[:,wind_initial: winds_ends[i]]  for i, wind_initial in enumerate(winds_initials)     ]
    for count_wind, wind in enumerate(winds):
        if check_zero_wind(wind, params_zero_window['zero_thres']):
            init1 = winds_initials[count_wind]
            end1 = winds_ends[count_wind]
            wind = wind + np.vstack([np.sin(np.arange(wind.shape[1])*params_zero_window['freq_fill'] + np.random.rand()).reshape((1,-1))
                                            for _ in range(wind.shape[0])])
            coeffs[:,init1 : end1] = wind
            print('disnullified part!')
    return coeffs
    
    
def check_zero_wind(wind, thres):
    """
    Check if the average absolute value of coefficients in a window is below a threshold.
    
    Args:
        wind (numpy.ndarray): Window of coefficients.
        thres (float): Threshold for zero window detection.
    
    Returns:
        bool: True if all coefficients in the window are below the threshold, False otherwise.
    
    """
    return (np.mean(np.abs(wind),1) < thres).all()
        
        
        
        
def z_normalize_neurons(matrix):
    """
    Perform Z-normalization on the neurons' activity matrix.
    
    Args:
        matrix (ndarray): The matrix of neurons' activity.
    
    Returns:
        ndarray: The normalized matrix.
    """
    # Calculate the mean and standard deviation along the time axis (axis=1)
    mean = np.mean(matrix, axis=1, keepdims=True)
    std = np.std(matrix, axis=1, keepdims=True)

    # Perform Z normalization
    normalized_matrix = (matrix - mean) / std

    return normalized_matrix        
        

def create_data_name(data_name = '', xmin = '0', xmax = 'n',ymin = '0',ymax = 'n', type_name = 'data'):
    """
    This function creates a string with a specified format for data file names.
    
    Parameters:
    data_name (str, optional): The name of the data. Default value is an empty string.
    xmin (str, optional): The lower limit of the x axis. Default value is '0'.
    xmax (str, optional): The upper limit of the x axis. Default value is 'n'.
    ymin (str, optional): The lower limit of the y axis. Default value is '0'.
    ymax (str, optional): The upper limit of the y axis. Default value is 'n'.
    type_name (str, optional): The type of data. Default value is 'data'.
    
    Returns:
    str: The generated string in the format 'type_name_data_name_xmin_xmax_ymin_ymax.npy'.
    
    Example:
    >>> create_data_name('data_sample', '-5', '5', '-10', '10', 'experiment')
    'experiment_data_sample_xmin_-5_xmax_5_ymin_-10_ymax_10.npy'
    """    
    return '%s_%s_xmin_%s_xmax_%s_ymin_%s_ymax_%s.npy'%(type_name, data_name,str(xmin), str(xmax), str(ymin), str(ymax))
    

def cut_gauss(gaussian, t, wind, left, right):
    """
    Cuts a Gaussian array to fit within specified left and right boundaries.
    
    Parameters:
        gaussian (numpy.ndarray): The 1D Gaussian array to be cut.
        t (int): The center index around which the Gaussian array is considered.
        wind (int): The half-size of the window around the center index 't'.
        left (int): The left boundary index of the desired region.
        right (int): The right boundary index of the desired region.
    
    Returns:
        numpy.ndarray: The trimmed Gaussian array that fits within the specified boundaries.
    """
    if t + wind > right:
        diff = t + wind - right
        g = gaussian[:-diff]
    elif t - wind < left:
        diff = left - (t - wind)
        g = gaussian[diff:]
    else:
        g = gaussian.copy()
    g = g/g.sum()
    return g

def gaussian_convolve(mat, wind = 10, direction = 1, sigma = 1, norm_sum = True, plot_gaussian = False):
    """
    Convolve a 2D matrix with a Gaussian kernel along the specified direction.
    
    Parameters:
        mat (numpy.ndarray): The 2D input matrix to be convolved with the Gaussian kernel.
        wind (int, optional): The half-size of the Gaussian kernel window. Default is 10.
        direction (int, optional): The direction of convolution. 
            1 for horizontal (along columns), 0 for vertical (along rows). Default is 1.
        sigma (float, optional): The standard deviation of the Gaussian kernel. Default is 1.
    
    Returns:
        numpy.ndarray: The convolved 2D matrix with the same shape as the input 'mat'.
        
    Raises:
        ValueError: If 'direction' is not 0 or 1.
    """
    if direction == 1:
        gaussian = gaussian_array(2*wind,sigma)
        if norm_sum:
            gaussian = gaussian / np.sum(gaussian)

        mat_shape = mat.shape[1]
        return np.vstack([ [np.sum(mat[row, np.max([t - wind,0]): np.min([t + wind, mat_shape])]*cut_gauss(gaussian, t, wind, left = 0, right = mat_shape)) 
                      for t in range(mat.shape[1])] 
                    for row in range(mat.shape[0])])
    elif direction == 0:
        return gaussian_convolve(mat.T, wind, direction = 1, sigma = sigma).T
    else:
        raise ValueError('invalid direction')
    
def pad_mat(mat, pad_val, size_each = 1, axis = 1):
    if axis == 1:
        each_pad = np.ones((mat.shape[0], size_each))*pad_val
        mat = np.hstack([each_pad, mat, each_pad])
    else:
        each_pad = np.ones((size_each, mat.shape[1]))*pad_val
        mat = np.vstack([each_pad, mat, each_pad])        
    return mat
    
def mov_avg(c, axis = 1, wind = 5):
    if len(c.shape) == 2 and axis == 1:
        return np.hstack([np.mean( c[:,np.max([i-wind, 1]):np.min([i+wind, c.shape[1]])],1).reshape((-1,1))
              for i in range(c.shape[1])])
    elif len(c.shape) == 2 and axis == 0:
        return mov_avg(c.T, axis = 1).T
    elif len(c.shape) == 3: # and axis == 0:
        return np.dstack([mov_avg(c[:,:,t], axis = axis) for t in range(c.shape[2])  ])
    else:
        raise ValueError('how did you arrive here? data dim is %s'%str(c.shape))
    
    

    

def gaussian_pdf(x, mu, sigma):
    """
    Calculate Gaussian Probability Density Function (PDF) values.

    Parameters:
    - x: array-like, values at which to evaluate the PDF
    - mu: mean of the distribution
    - sigma: standard deviation of the distribution

    Returns:
    - y: array, Gaussian PDF values corresponding to the input x values
    """
    y = norm.pdf(x, loc=mu, scale=sigma)
    return y
    
def return_gaussian_weighted_times_apikes(spike_times, start_wind, end_wind,  sigma = 1):
    """
    Calculate the sum of Gaussian-weighted spike times within a given window.
    
    Parameters:
        spike_times (numpy.ndarray): Array of spike times.
        start_wind (float): The starting time of the window.
        end_wind (float): The ending time of the window.
        sigma (float, optional): The standard deviation parameter for the Gaussian distribution. Defaults to 1.
    
    Returns:
        float: The sum of spike times, weighted by their Gaussian weights within the specified window.
        
    Example:
        spike_times = np.array([0.3, 0.6, 0.8, 1.2, 1.7])
        start_wind = 0.5
        end_wind = 1.5
        sigma = 0.5
        result = return_gaussian_weighted_times_apikes(spike_times, start_wind, end_wind, sigma)
        print(result)  # Output will vary based on the given spike_times array.
    """
    center = 0.5*(end_wind - start_wind  ) + start_wind
    relevant_vals = spike_times[(spike_times < end_wind) & (spike_times > start_wind)] - center
    if len(relevant_vals) > 0:
        gaussian = np.exp(-(relevant_vals ** 2) / (2 * sigma ** 2))
        
        sum_vals = np.sum(gaussian)
    else:
        sum_vals = 0
    return sum_vals
                 


def compute_eigenvector_similarity(matrix1, matrix2):
    """
    Compute the similarity between the largest eigenvectors of two matrices.
    
    Args:
        matrix1 (ndarray): The first matrix.
        matrix2 (ndarray): The second matrix.
    
    Returns:
        float: The similarity score between the largest eigenvectors.
    """
    # Compute eigenvalues and eigenvectors of matrix1
    eigenvalues1, eigenvectors1 = np.linalg.eig(matrix1)
    
    # Sort the eigenvalues and eigenvectors by magnitude
    sorted_indices1 = np.argsort(np.abs(eigenvalues1))[::-1]
    eigenvalues1 = eigenvalues1[sorted_indices1]
    eigenvectors1 = eigenvectors1[:, sorted_indices1]
    
    # Compute eigenvalues and eigenvectors of matrix2
    eigenvalues2, eigenvectors2 = np.linalg.eig(matrix2)
    
    # Sort the eigenvalues and eigenvectors by magnitude
    sorted_indices2 = np.argsort(np.abs(eigenvalues2))[::-1]
    eigenvalues2 = eigenvalues2[sorted_indices2]
    eigenvectors2 = eigenvectors2[:, sorted_indices2]
    
    # Compute the cosine similarity between the largest eigenvectors
    similarity = np.abs(np.dot(eigenvectors1[:, 0], eigenvectors2[:, 0]))
    
    return similarity

def is_similar_to_existing(matrix, matrices, similarity_threshold):
    """
    Check if a given matrix is similar to any of the matrices in a list based on the similarity threshold.

    Parameters:
        matrix (numpy.ndarray): The matrix to be compared with existing matrices.
        matrices (list of numpy.ndarray): A list of matrices for comparison.
        similarity_threshold (float): The minimum similarity value required to consider two matrices similar.

    Returns:
        bool: True if a similar matrix is found in the list; False otherwise.

    Example:
        matrix = np.array([[1, 2], [3, 4]])
        matrices = [np.array([[0, 1], [2, 3]]), np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        similarity_threshold = 0.9
        result = is_similar_to_existing(matrix, matrices, similarity_threshold)
        print(result)  # Output will be True, as the given matrix matches the second matrix in the list.
    """    
    for existing_matrix in matrices:
        similarity = compute_eigenvector_similarity(matrix, existing_matrix)
        if similarity > similarity_threshold:
            return True
    return False


def deparacated_generate_random_different_fs(p, num_subdyns, seed = 0, similarity_threshold = 0.8):
    
    """
    Generate p random matrices with a spectral radius of 1, ensuring dissimilarity.
    
    Args:
        p (int): The number of random matrices to generate.
        similarity_threshold (float): The maximum allowed similarity between matrices.
    
    Returns:
        list: The list of generated matrices.
    """
    matrices = []      
    for i in range(num_subdyns):
        while True:
            # Step 2: Generate random matrix Mi
            Mi = np.random.rand(p, p)
            Mi = norm_mat(Mi, type_norm = 'evals', to_norm = True)
            # Step 4b: Compute spectral radius of Mi
            sr = np.max(np.abs(np.linalg.eigvals(Mi)))
            
            # Step 4c: Check if spectral radius is 1
            if not is_similar_to_existing(Mi, matrices, similarity_threshold):
                matrices.append(Mi)
                print('made %d matrices!'%len(matrices))
                break  # Move to the next matrix
                
    return matrices



def make_kernel_2d(firing_rate, with_kNN = True, with_norm = True, k = 30):
    """
    Create a kernel matrix based on firing rate data.
    
    Args:
        firing_rate (ndarray): The firing rate data matrix, where each row represents the firing rates of a neuron.
        with_kNN (bool, optional): Whether to apply kNN-based hard thresholding. Default is True.
        with_norm (bool, optional): Whether to normalize the kernel matrix. Default is True.
        k (int, optional): The number of nearest neighbors to consider when applying kNN thresholding. Default is 30.
    
    Returns:
        ndarray: The kernel matrix.
    
    Raises:
        ValueError: If with_kNN or with_norm is not a boolean value.
    """
        
    H = np.zeros((firing_rate.shape[0],firing_rate.shape[0]))
    for neuron in range(firing_rate.shape[0]):
        cur_activity = firing_rate[neuron,:].reshape((1,-1))
        dists = ((firing_rate - cur_activity)**2).sum(1)
        H[neuron] = dists
    if with_kNN:
        H = apply_hard_thres(H, axis = 1, k = k)
    if with_norm:
        H = H / np.sqrt((H**2).sum(1)).reshape((-1,1))
    return H

def make_kernel_3d(firing_rate, indices_regs, with_kNN = True, with_norm = True, k = 30):
    """
    Create a 3D kernel matrix based on firing rates and specified indices.
    the different dimensions are for different regions. i.e. each kay : val refer to different area
    Args:
        firing_rate (numpy.ndarray): The firing rate matrix.
        indices_regs (list): A list of indices specifying regions.
        with_kNN (bool, optional): Whether to include k-nearest neighbor calculation. Defaults to True.
        with_norm (bool, optional): Whether to include normalization. Defaults to True.
        k (int, optional): The number of nearest neighbors to consider. Defaults to 30.
    
    Returns:
        list: A list of 2D kernel matrices.
    
    """
    return [make_kernel_2d(firing_rate[indices_reg,:], with_kNN, with_norm, k)
     for indices_reg in indices_regs]
    
def update_D_lasso(lambda_D,D, data, latent_dyn, update_type_D, lambdas = [], D_graph_driven = True, 
                   params = {}):
    """
    Update the D matrix using Lasso regularization based on the given data and parameters.
    
    Args:
        lambda_D (float): Regularization term for Lasso.
        D (numpy.ndarray): The D matrix to be updated.
        data (numpy.ndarray): The input data for one trial.
        latent_dyn (numpy.ndarray): The latent dynamics matrix.
        lambdas (numpy.ndarray, optional): The lambdas matrix. Defaults to an empty list.
        D_graph_driven (bool, optional): Whether the D matrix is graph-driven. Defaults to True.
        params (dict, optional): A dictionary containing additional parameters. Defaults to an empty dictionary.
    
    Returns:
        numpy.ndarray: The updated D matrix.
    
    Raises:
        ValueError: If D is graph-driven but lambdas is empty.
    """
    D_update = D.copy()
    params = {**{'update_c_type':'OMP','reg_term':0}, **params} 
    # H is the graph
    #  I assume here that data is for one trial - i.e. it is a numpy array
    # size of lambdas is num_neurons over p
    # I assume that if it is block then only the block is given. 
    if D_graph_driven and checkEmptyList(lambdas):
        raise ValueError('if graph driven you must provide graph')

    for neuron in range(D.shape[0]):

        D_neuron = solve_lasso_problem(data[neuron,:], latent_dyn.T, 
                                       reg =  lambda_D, update_type = update_type_D, 
                        random_state = 0, params_update_c  = params)
        if D_graph_driven:
            lambdas_vec = lambdas[neuron, :]
            D_neuron = D_neuron.flatten()/lambdas_vec.flatten();
        D_update[neuron,:] = D_neuron
    # data must 
    return D_update
    
def list_of_array_reorder(list_of_arrays, argsort_vals):
    """
    list of array reorder
    """
    return_list = []
    for val in np.sort(argsort_vals):
        return_list.append(list_of_arrays[val])
    return return_list
        
    
def list_of_array_reorder(list_of_arrays, argsort_vals):
    """
    list of array reorder
    """
    return_list = []
    for val in np.sort(argsort_vals):
        return_list.append(list_of_arrays[val])
    return return_list
        

def order_list_function(indices_regs, argsort_vals_0):
    """
    Convert a list of indices to corresponding elements from another list.

    Parameters:
    - indices_regs (list): List containing elements to be mapped to.
    - argsort_vals_0 (list): List of indices to be converted to elements from indices_regs.

    Returns:
    - list: A new list containing elements from indices_regs corresponding to the indices
            specified in argsort_vals_0.
    """   
    

    new_list = []
    for el in argsort_vals_0:
        new_list.append(indices_regs[el])
    return new_list
    

    

def from_regions_to_indices(regions_list):
    print('pay attention from_regions_to_indices not suitable for missing regions!')
    regions_list = np.array(regions_list)
    indices_regs = [np.where(regions_list == reg)[0] for reg in np.unique(regions_list)]

    vals_0 = np.array([indices_reg[0] for indices_reg in indices_regs if len(indices_reg) > 0  ])

    argsort_vals_0 = np.argsort(vals_0)

    try:
        return np.array(indices_regs)[argsort_vals_0 ]
    except:
        return order_list_function(indices_regs, argsort_vals_0)

    
    

def updateLambdasMat(D, H, params_graph, params ):
    """
    Update the lambdas matrix based on the given D block and H matrix.
    
    Args:
        D (numpy.ndarray): The D block matrix.
        H (numpy.ndarray): The H matrix.
        params_graph (dict): A dictionary containing the parameters for the graph.
    
    Returns:
        numpy.ndarray: The updated lambdas matrix.
    
    Raises:
        ValueError: If the weight projection matrix has a different number of rows than the neurons.
    """
    # this one is D block
    p = D.shape[1]
    n_neurons = D.shape[0]
    params_graph = {**{'epsilon':1,  'mask':[], 'beta':1, 'zeta':10}, **params_graph}
    beta = params_graph['beta']

    print(H.shape)
    print(n_neurons)
    if H.shape[0] ==  n_neurons:                                 #    - If the weight projection matrix has the same number of rows as pixels, update based on a matrix multiplication                     
    
        lambdas = params_graph['epsilon']/(beta +  D + params_graph['zeta']*H @ D);                #      - Calculate the wright updates tau/(beta + |s_i| + [P*S]_i)
    else:
        raise ValueError('This case is not defined yet') #future


    return lambdas    
    
def list2countsdict(list_areas):
    un, counts = np.unique(list_areas, return_counts = True)
    return {un_i:counts[i] for i,un_i in enumerate(un)}    
         
import time     

def check_values(list_of_arrays, max_val = 10**3, min_val = -10**3, rep_nan = True, rep_0 = True, noise_0 = 0.2):
    ll_new = []
    
    for el in list_of_arrays:
        el = el.copy()
        el[el >= max_val] = np.min([np.nanmedian(el), max_val])
        el[el <= min_val] = np.max([np.nanmedian(el), min_val])
        if rep_nan:
            if np.isnan(el).all():
                el = np.random.rand(*el.shape)
            else:
                el[np.isnan(el)] = np.nanmean(el)
        if rep_0:
            if np.sum(np.abs(el)) < 10**-9:
                el += np.random.rand(*el.shape)*noise_0
        ll_new.append(el)    
    return ll_new
        
    
    
    
#%% Main Model Training
def train_model_include_D(max_time = 500, dt = 0.1, dynamics_type = 'lorenz',num_subdyns = 3, 
                          error_reco = np.inf, error_order_max  = 1, error_order = 1, data = [], same_c = False,step_f = 30, 
                          GD_decay = 0.99, weights_orders = [],clean_dyn = [],max_error = 1e-7,grad_vec_min_max = [], 
                          max_iter = 3000, F = [], coefficients = [], params= {'update_c_type':'inv','reg_term':0,'smooth_term':0}, 
                          epsilon_error_change = 10**(-5), D = [],
                          x_former =[], latent_dim = None, include_D  = True,step_D = 30, reg1=0,reg_f =0 , 
                          max_data_reco = 1e-3, acumulated_error = False, sigma_mix_f = 0.1, error_step_add = 120, 
                          action_along_time = 'median', error_order_max_display = 2, to_print = True, seed = 0, seed_f = 0, 
                          return_evolution = False,  normalize_eig  = True, 
                          params_ex = {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2,'theta':0, 'orientation_ax':'x'}, 
                          start_sparse_c = False,
                          max_corr = 0.1,
                          decaying_reg = 1, 
                          center_dynamics = False, 
                          bias_term = False, 
                          bias_out = False,
                          other_params_c = {}, include_last_up = False,
                          min_step_f = 1e-3,weights_orders_style = 'update',
                          initial_weight_order_add = 1e-8, redefine_step_f = False, 
                          decay_decay = False, num_no_change = 5,
                          mix_f_method = 'all', sparse_f = False, 
                          sparse_f_params = {},
                          add_bias_3 = False, 
                          warm_start = True, 
                          warm_start_path = '',
                          order_type = 'gradient', 
                          include_identity = False, 
                          error_while = 'median',
                          bias_val = 0, bias_out_val = 0, 
                          include_patch = False, dist_type_f = 'random',
                          patch_size = 200, repeat_num = 3, num_patch = 5, 
                          init_orth = True, num_gradient_steps = 1, 
                          to_save_mid = False,
                          path_save = '.', type_norm = 'evals', latent_dim_per_region = [],  
                          save_freq = 10, add_avg = False,
                          wind_avg = 20, sparsity_on_f_max = 70, increase_in_sparsity_f = 0.5,
                          take_multiple_gd  = False, D_graph_driven = False, combine_session = True, 
                          saving_graphs = True, saving_graph_freq = 5,
                          infer_x_c_together = True, 
                          lambda_x = 0, 
                          with_batces  = False, 
                          include_mask =False,
                          data_min = 0, 
                          data_max = np.inf, multiply_data = 1, nullify_big_winds = True, 
                          addition = '', update_D_based_on_one_trial = 1, norm_D_cols = True,
                          decorrelate_D = False, decorrelate_F = False, update_block_D = False,
                          weight_observation_eq= 5, indices_regs = [],
                          D_graph_params = {'with_kNN' : True, 'with_norm':True, 'k': 15}, params_graph = {},
                          lambda_D = 0.3, H = [], update_type_D = 'nls',  
                          type_x_infer ='nls', 
                          latent_dyns_initialization = 'random', 
                          multiple_D = False, null_D = True,
                          D_with_lasso = True, step_D_decay = 0.9999,
                          fix_D = True, # if to fix D
                          fix_f = True, 
                          fix_c = True, 
                          fix_x0 = True, 
                          fix_x = False,
                          x0 = [], 
                          use_x0 = True, # whether to use these for init. 
                          use_new_est = True, 
                          use_both_obs_and_latent  = 'latent', # HOW TO UPDATE X IF UPDATING ONLY X
                          params_update_x = {},
                          to_mix_F = False,
                          to_hard_thres_c = True,
                          k_hard_thres_c = 2,
                          hard_thres_c_freq = 2,
                          addi_save = {},
                          parameters_f_wise_step = {}, 
                          save_comparison_to_ground_truth = True,
                          normalize_F = False, 
                          noise_level = 0.1, l1_D = [], params_D = {},
                          single_session = -1, 
                          dynamics_prior = True,
                          params_infer_x_no_prior = {'lambda_frob': 0.1 , 
                                                     'lambda_smooth_iters': 0,
                                                     'lambda_smooth_time': 0.1,
                                                     'lambda_decor': 0.1},
                          PCA_type  = 'local',
                          D_with_PCA = False, 
                          all_regs_together = False,
                          nullify_D = False
                          ):
    
  """
  This is the main function to train the model! 
  Inputs:
      max_time      = Number of time points for the dynamics. Relevant only if data is empty;
      dt            =  time interval for the dynamics
      dynamics_type = type of the dynamics. Can be 'cyl', 'lorenz', 'multi_cyl', 'torus', 'circ2d', 'spiral'
      num_subdyns   = number of sub-dynamics
      error_reco    = intial error for the reconstruction (do not touch)
      error_order_max= the step of the weights given to errors from different orders
      error_order   = error of the order
      data          = if one wants to use a pre define groud-truth dynamics. If not empty - it overwrites max_time, dt, and dynamics_type
      same_c        = if there is more than one sample for the dynamics (for instance - noisy case), than whether to find a shared coefficients representation to all samples (irrelevant if only one sample) 
      step_f        = initial step size for GD on the sub-dynamics
      GD_decay      = Gradient descent decay rate
      weights_orders= only use if you have a pre-defined set of weights for the different orders
      clean_dyn     = use if the dynamics in data is not clean (e.g. noisy scenario). Otherwise - keep empty.
      max_error     = Threshold for the model error. If the model arrives at a lower reconstruction error - the training ends.
      grad_vec_min_max    = the amount by which the curve in 'weights_orders' will change towards higher orders
      max_iter      = # of max. iterations for training the model
      F             = pre-defined sub-dynamics. Keep empty if random.
      coefficients  = pre-defined coefficients. Keep empty if random.
      params        = dictionary that includes info about the regularization and coefficients solver. e.g. {'update_c_type':'inv','reg_term':0,'smooth_term':0}
      epsilon_error_change = check if the sub-dynamics do not change by at least epsilon_error_change, for at least 5 last iterations. Otherwise - add noise to f
      D             = pre-defined D matrix (keep empty if D = I)
      x_former      = IGNORE; NEED TO ERASE! (NM&&&&)
      latent_dim    =  If D != I, it is the pre-defined latent dynamics.
      include_D     = If True -> D !=I; If False -> D = I
      step_D        = GD step for updating D, only if include_D is true
      reg1          = if include_D is true -> L1 regularization on D
      reg_f         = if include_D is true ->  Frobenius norm regularization on D
      max_data_reco = if include_D is true -> threshold for the error on the reconstruction of the data (continue training if the error (y - Dx)^2 > max_data_reco)
      acumulated_error       = whether to check a k_th order error or the acumulated error (True = accumulated, False = ordered error)
      sigma_mix_f            = std of noise added to mix f
      error_step_add         = consider a new order only after passing error_step_add  iterations. Do not touch. 
      action_along_time      = the function to take on the error over time. Can be 'median' or 'mean'
      error_order_max_display = error order to print when training (int > 0)
      to_print               = to print error value while training? (boolean)
      seed                   = random seed
      seed_f                 = random seed for initializing f
      return_evolution       = store the evolution of the training (does not change the model, but can be very heavy so recommneded False unless the evolution is needed)
      normalize_eig          = whether to normalize each sub-dynamic by dividing by the highest abs eval
      params_ex              = parameters related to the creation of the ground truth dynamics. e.g. {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}
      start_sparse_c         = If true - start with sparse c and then infer F. If False - start with random F and infer c (not necessarily sparse)
      init_distant_F         = when initializing F -> make sure that the correlation between each pair of {f}_i does not exeed a threshold
      max_corr               = max correlation between each pair of initial sub-dyns (relevant only if init_distant_F is True)
      decaying_reg           = decaying factor for the l1 regularization on the coefficients. If 1 - there is no decay. (should be a scalar in (0,1])
      center_dynamics        = whether to shift the dynamics to be centered around (0,0). (boolean)                                                                                                                       
      bias_term              = whether to add a bias term to the model, in the form of x_(t+1) = \sum(f_i c_i)* x_t + bias (boolean)
      bias_out               = in cases where D!=I, y = Dx + bias_out
      weights_orders_style   = can be 'update' or 'renew'
      initial_weight_order_add = the weight of the additional order
      redefine_step_f        = whether to re-define the step size of f's gradient-descent
      num_no_change          = how many unchanged iterations have passed till mixing of f
      mix_f_method           = how to mix f
      latent_dyns_initialization = 'random' or 'lass'
      multiple_D             = If False - creates 1 D for all trials; if True - create a D for each trials. 
      
      
     future: change update_D_based_on_one_trial to # of choice
     
     
     latent_dim_per_region = num of ensembles for a region
  """  

  if D_with_PCA:
      print('PAY ATTENTION! D IS JUST PCA. i.e. this is NOT CREIMBO. ok?')
      input('ok PCA?!')
     
  parameters_f_wise_step = {**{
      "size_batch": 0.5,
      "ratio_min": 1/20,
      "ratio_max": 20,
      "wise_step": True,
      "num_steps": 5
  },**parameters_f_wise_step}
  if not use_x0:
      x0 = []
  x_together = infer_x_c_together
  if  decorrelate_D or decorrelate_F or update_block_D :
      raise ValueError('need to implement!')
      
  cur_reco = []
  sparse_f_params = {**{'axis':'1', 'percent0':50},**sparse_f_params}
  params_graph = {**{'epsilon':1,  'mask':[], 'beta':1, 'zeta':10},**params_graph}
  """
  create path save
  """

  if path_save == '.':
      path_save = os.getcwd() + os.sep + today
  if not os.path.exists(path_save):
      os.makedirs(path_save)     
      
          
  """
  raise Error
  """
  if acumulated_error  and include_D: 
      raise ValueError('When including D, the error should not be cumulative (you should set the acumulated_error input to False')
  if np.isnan(error_order): 
    error_order           = 1  
  res_intermediate = {}
  if error_order_max > 1 and include_D: 
    print('Error step was reduced to 1 since D is updated')
    error_order_max = 1
    error_order_max_display = 1
  if len(weights_orders) == 0:
      if error_order_max == 1:
          weights_orders = [1]
      else:
          weights_orders = np.linspace(1,2**error_order_max,error_order)[::-1]
      weights_orders = weights_orders/np.sum(weights_orders)
      
  
  if include_D and bias_term and bias_out:
      print('Disabling internal bias term since D ~=I and bias_out is true')
      bias_term = False
  if not include_D and bias_out: # disable bias out if D = I
      bias_out = False
      

  if return_evolution:
      store_iter_restuls = {'coefficients':[], 'F':[],'L1':[]}
  step_f_original = step_f
  
  """
  Define data and number of dyns
  """  
  if len(data) == 0 :
    print(dynamics_type)
    if dynamics_type == 'epi_amir_short' :
        data  , names_files          = create_dynamics(type_dyn = dynamics_type,
                                          max_time = max_time, dt = dt, params_ex = params_ex, addition = addition)

    else:

        data            = create_dynamics(type_dyn = dynamics_type, max_time = max_time, dt = dt, params_ex = params_ex, addition = addition,
                                          single_session = single_session, 
                                          all_regs_together = all_regs_together)
        if 'synth' in dynamics_type:
            data_load = data.copy()


  if isinstance(data, np.ndarray): 
      if not include_D: 
          latent_dyn = data
      one_dyn = True
      
  else:       
      if len(data) == 1: 
          one_dyn = True
          data = data[0]
          if not include_D: 
              latent_dyn = data 
          
      else: 
        one_dyn = False
        if not include_D: 
          latent_dyns = data
        

  if 'multi_reg' in dynamics_type :
    if  'meso' not in dynamics_type and 'human' not in dynamics_type:
        if all_regs_together:
            raise ValueError('TODO!')
        regions = data['labels']
        info_keep_order = []
        for reg in regions:
            if reg not in info_keep_order:
                info_keep_order.append(reg)
            else:
                pass
        
    
        indices_regs = from_regions_to_indices(regions)  
        info_keep_order = []
        for reg in regions:
            if reg not in info_keep_order:
                info_keep_order.append(reg)
            else:
                pass

        counts = {count:len(indices_reg) for count, indices_reg in enumerate(indices_regs)}

        unique_regions = info_keep_order 
        
        num_regions = len(np.unique(regions))
        graphs = data['H_dict']
        graphs = [graphs[i] for i in np.arange(len(graphs))]
        
                                                                                                                         
    elif  'meso' in dynamics_type or 'human' in dynamics_type :  
          data_load = data.copy()

          regions_full = data['labels']
          ys = data['data_active']
          data_new = []
          counts = {}
          unique_regions = {}
          num_regions = {}
          if not all_regs_together:
              indices_regs_full = data_load['indices_regs'] # SINCE CURRENTLY DOES NOT WORK WITH MISSING REGIONS
          else:
              indices_regs_full = {ses: [np.arange(y_i.shape[0])] for ses, y_i in ys.items()}
              
          indices_regs = {}
          info_keep_order = {}
          graphs = {}
          keys_to_num = data_load['keys_to_num']

          
          for session_full, regions in regions_full.items():
                session = keys_to_num[session_full]
                data_new.append(ys[session_full])

                info_keep_order[session] = []
                for reg in regions:
                    if reg not in info_keep_order[session]:
                        info_keep_order[session].append(reg)
                    else:
                        pass
                indices_regs[session] = indices_regs_full[session_full] 

                counts[session] = np.array([len(indices_reg) for indices_reg in indices_regs[session]])

                unique_regions[session] = info_keep_order[session]
          
                num_regions[session] = len(np.unique(regions))
                if D_graph_driven:   
                    graphs[session] = data['H_dict'][session_full]

    if dynamics_type == 'multi_reg_neuron_per_trial' or  'multi_reg_meso' in dynamics_type or 'human' in dynamics_type :

        data = data['data_active']  
        if 'multi_reg_meso' in dynamics_type or 'human' in dynamics_type :
            keys = data_load['keys_to_num']
            data = [data[key] for key in keys]
        else:
            data = [data[key] for key in list(data.keys())]

    else:

        data = data['data'] 
     
        
  elif  'synth_multi_' in dynamics_type : 
    regions_full = data['labels'] 
    info_keep_order = {}
    unique_regions = {}
    num_regions = {}
    indices_regs = {}
    counts = {}
    regions_full_new = {}
    
    ys = data['ys']
    cs = data['cs']
    xs = data['xs']
    F_ground_truth  = data['F']
    Ds = data['Ds']
    if 'Ds_masks' in data:
        Ds_masks_or = data['Ds_masks']
    else:
        Ds_masks_or = data['Ds']
    
    Ds_ground_truth = []
    xs_ground_truth = []
    cs_ground_truth = []
    Ds_masks = []
    data_new = []

    for session_i, (session, regions) in enumerate(regions_full.items()):
          data_new.append(ys[session])
          Ds_masks.append(Ds_masks_or[session])

          Ds_ground_truth.append(Ds[session])
          xs_ground_truth.append(xs[session])
          cs_ground_truth.append(cs[session])
          
          
          info_keep_order[session] = []
          for reg in regions:
              if reg not in info_keep_order[session]:
                  info_keep_order[session].append(reg)
              else:
                  pass
          indices_regs[session_i] = from_regions_to_indices(regions)  
          counts[session_i] = np.array([len(indices_reg) for indices_reg in indices_regs[session_i]])
          unique_regions[session_i] = info_keep_order[session]
          regions_full_new[session_i] = regions_full[session]
          num_regions[session_i] = len(np.unique(regions))
          


          
          if D_graph_driven:   
              if isinstance(data['H_dict'] , dict):
                  graphs[session_i] = data['H_dict']
                  graphs[session_i] = [graphs[i] for i in np.arange(len(graphs))]
              else:
                  graphs[session_i] = data['H_dict'][session]
                 
              
          if use_x0:    
              x0 = [xs_ground_truth[i][:,0] for i in range(len(xs_ground_truth))]
    

    data = data_new.copy()
    if fix_c:
        coefficients = cs_ground_truth
    else:
        coefficients = [cs_i + np.random.rand(*cs_i.shape)*noise_level for cs_i in  cs_ground_truth]
    if fix_f:
        F = F_ground_truth
    else:
        F = [f_i + np.random.rand(*f_i.shape)*noise_level for f_i in  F_ground_truth] 
        if num_subdyns > len(F_ground_truth):
            addi_F_num  =  num_subdyns - len(F_ground_truth)
            addi_F = [np.random.rand(*F[0].shape)*noise_level for f_i in  range(len(F))] 
            F = F + addi_F

            
    if fix_D:
        D = Ds_ground_truth
    else:
        D = [D_i + np.random.rand(*D_i.shape)*noise_level for D_i in  Ds_ground_truth]
        
 
  if data_min != 0 or  data_max != np.inf:
      if one_dyn    :
          data = multiply_data*data[:, data_min : data_max]
          
      else:
          data = [multiply_data*data_i[:, data_min : data_max] for data_i in data]
          
  if not include_D and ((len(data) > 1 and isinstance(data, np.ndarray)) or (len(data)==1 and isinstance(data,list))):
      latent_dyn = data
      

      data = (data - np.percentile(data,5))/(np.percentile(data,95) - np.percentile(data,5))
  if one_dyn:
      n_times = data.shape[1]
  if ('multi_reg' in dynamics_type or  'synth_multi_' in dynamics_type)  and not isinstance(latent_dim_per_region, (list, np.ndarray, tuple)):

      if isinstance( num_regions, dict):
          if 'meso' in dynamics_type or 'human' in dynamics_type :
              if all_regs_together:
                  num_regions = 1
              else:   
                  num_regions = len(data_load['unique_regions'])
              if num_regions != len(list(indices_regs.values())[0]):
                  raise ValueError('num regions must match the number of indices per session')
      
          elif  (np.array(list(num_regions.values())[0] ) == list(num_regions.values())[0] ).all(): # i.e. number of regions for different recordings
              num_regions = list(num_regions.values())[0]
          else:
              raise ValueError('need to implement!') 
              
      latent_dim_per_region = [latent_dim_per_region] * num_regions
      
  elif 'multi_reg' in dynamics_type  and len(latent_dim_per_region) != num_regions   :
      raise ValueError('The number of per region dim must be equal to the number of regions ')
      
  if not checkEmptyList(latent_dim_per_region) and ('multi_reg' in dynamics_type or 'synth_multi_' in dynamics_type):

      latent_dim = np.sum(latent_dim_per_region) 
 
  if 'multi_reg' in dynamics_type or 'synth_multi_'  in dynamics_type:
      

      if not multiple_D:         
        
    
          D_mask = block_diag(*[np.ones(( counts[i], 
              latent_dim_per_region[i] ))*(i+1) for i in range(num_regions)]) 
          cols_blocks =  np.split(np.arange(D_mask.shape[1]), np.cumsum(latent_dim_per_region)[:-1])

    
          if saving_graphs:
              fig, ax = plt.subplots(figsize = (25,25))
              D_mask_nan = D_mask.copy()
              D_mask_nan[D_mask == 0] = np.nan

              sns.heatmap(pd.DataFrame(D_mask_nan),ax = ax, square = False) 

              ax.set_xticks(np.arange(D_mask_nan.shape[1]))
              ax.set_xticklabels(np.repeat(unique_regions,latent_dim_per_region), fontsize = 6)

              fig.tight_layout()
              
              plt.savefig(path_save + os.sep +  'D.png')
              plt.close()
      else:
          if 'meso' in dynamics_type or 'human' in dynamics_type :
              

              D_masks_or = data_load['D_masks']
              
              if all_regs_together:
                  D_masks_or = {ses: np.ones((D_mask.shape[0], latent_dim)) for ses, D_mask in D_masks_or.items()} 
                  

                  
              keys2nums = data_load['keys_to_num']
              D_masks = {keys2nums[key]: D_mask for key, D_mask in D_masks_or.items()}
              if all_regs_together:
                  unique_regions_all = np.array(['all'])
              else:   
                  unique_regions_all = data_load['unique_regions']

              if not all_regs_together:
                  D_masks = {key:np.repeat(mask,latent_dim_per_region[0], axis = 1) for i, (key, mask) in  enumerate(D_masks.items())}
              D_mask = list(D_masks.values())[0]
              cols_blocks =  np.split(np.arange(D_mask.shape[1]), np.cumsum(latent_dim_per_region)[:-1])

              if saving_graphs:
                  max_Ds_show = 5
                  Ds_show = np.min([max_Ds_show, len(D_masks)])
                  fig, axs = plt.subplots(1, Ds_show, figsize = (Ds_show*6, 10))
                  if Ds_show == 1:
                      axs = [axs]
                      
                 
                  if Ds_show == 1:
                      axs = np.array([axs])
     
                  for c, D_mask in enumerate(list(D_masks.values())):
                      if c < Ds_show:
                          ax = axs[c]
                          D_mask_nan = D_mask.copy()

                          
                          sns.heatmap(pd.DataFrame(D_mask_nan), ax = ax, square = False) 

                  fig.tight_layout()
                  print(path_save + os.sep +  'D_%d.png'%c)    
                  fig.savefig(path_save + os.sep +  'D_%d.png'%c)
                  plt.show()
 
                  
          elif 'synth' in dynamics_type:
              D_masks = Ds_masks.copy() 
              D_masks = {j:D_mask for j , D_mask in enumerate(D_masks)}
              cols_blocks =  np.split(np.arange(D_masks[0].shape[1]), np.cumsum(latent_dim_per_region)[:-1])

                  
              
                
              if saving_graphs:
                  max_Ds_show = 5
                  Ds_show = np.min([max_Ds_show, len(counts)])
                
                  fig, axs = plt.subplots(1, Ds_show, figsize = (Ds_show*6, 10))
                  if Ds_show == 1:
                      axs = [axs]
                  for c, session in enumerate(counts.keys() ):
                      if c < Ds_show:
                        ax = axs[c]  

                        D_mask = D_masks[session]
                        D_mask_nan = D_mask.copy()
                        D_mask_nan[D_mask == 0] = np.nan

                        regions = regions_full_new[session]
                        sns.heatmap(pd.DataFrame(D_mask_nan, index = regions, 
                                                 columns = np.repeat(unique_regions[session],latent_dim_per_region)), 
                                    ax = ax, square = False)

                    
                  fig.tight_layout()           
                  plt.show()
                  fig.savefig(path_save + os.sep +  'D_%s.png'%session)
            
                  
                  plt.close()                
      
              
          else:
              D_masks = {}
              
              raise ValueError('TODO - need to adjust keys starting from 0 for new data as done for the human and synth')

                  
                  
              for session in counts.keys() :


                  D_mask = block_diag(*[np.ones(( counts[session][i], 
                      latent_dim_per_region[i] ))*(i+1) for i in range(num_regions)]) 

                  cols_blocks =  np.split(np.arange(D_mask.shape[1]), np.cumsum(latent_dim_per_region)[:-1])
         
                  regions = regions_full[session]  

                      
                  D_masks[session] = D_mask
                  if saving_graphs:
        
                    fig, ax = plt.subplots(figsize = (25,25))
                    D_mask_nan = D_mask.copy()
                    D_mask_nan[D_mask == 0] = np.nan

                    sns.heatmap(pd.DataFrame(D_mask_nan, index = regions, 
                                             columns = np.repeat(unique_regions[session],latent_dim_per_region)), 
                                ax = ax, square = False)
                    ax.set_xticks(np.arange(D_mask_nan.shape[1]))
                    ax.set_xticklabels(np.repeat(unique_regions[session],latent_dim_per_region), fontsize = 6)
                    ax.set_yticks(np.arange(D_mask_nan.shape[0]))
                    ax.set_yticklabels(regions, fontsize = 6)
                    fig.tight_layout()
                    
                    fig.savefig(path_save + os.sep +  'D.png')
                    plt.close()       
  if 'multi_reg' not in dynamics_type and latent_dim == 0 and include_D and 'synth_multi_' not in dynamics_type:
    """
    CREATION OF D
    """

    if isinstance(data, list):
      latent_dim = int(np.max([data[0].shape[0] / 5,3])); n_times = data[0].shape[1]
    else:
      latent_dim = int(np.max([data.shape[0] / 5,3])); n_times = data.shape[1]
          
  if not include_D:
    if isinstance(data, list):
      latent_dim = data[0].shape[0] ; n_times = data[0].shape[1]
    else:
      latent_dim = data.shape[0]; n_times = data.shape[1]
      

  if D_with_PCA and not  include_D:
      raise ValueError('if D is with PCA you must include D')

  
  if include_D : # Namely - model need to study D


    if fix_D:
        if 'Ds_ground_truth' not in locals():
            raise ValueError('Ds_ground_truth not in locals')
        D = Ds_ground_truth
   
        
          
    else:
        if one_dyn:
          if   D_with_PCA: 
            
            D = apply_PCA_per_region(data, latent_dim, indices_regs_session = indices_regs.values()[0] , PCA_type = PCA_type)
            fix_D = True    
          elif len(D) == 0: 
              D = init_mat(size_mat = (data.shape[0], latent_dim) , dist_type ='uni', init_params={'k':4})
              if norm_D_cols:
                  D =   D/((np.sum(D**2, 0)**0.5).reshape((1,-1)) + 1e-19).reshape((1,-1))
          if D.shape[0] != data.shape[0]: raise ValueError('# of rows in D should be = # rows in the data ')
          
          
          
        elif not  multiple_D:
            if   D_with_PCA:
                raise ValueError('TODO pca one D multi dyn')
            else:
                D = init_mat(size_mat = (data[0].shape[0], latent_dim) , dist_type ='uni')
                if norm_D_cols:
                    D =   D/((np.sum(D**2, 0)**0.5).reshape((1,-1)) + 1e-9)
        else:

            if   D_with_PCA:

              D = [apply_PCA_per_region(data[i], latent_dim_per_region, indices_regs_session = indices_regs[i], PCA_type = PCA_type)
                   for i in range(len(data))]
              
              fix_D = True    
            else:
                D = [init_mat(size_mat = (data[i].shape[0], latent_dim) , dist_type ='uni') for i in range(len(data))]
                if norm_D_cols:
                    D = [ cur_block/((np.sum(cur_block**2, 0).reshape((1,-1)) + 1e-19)**0.5) for cur_block in D]
                
        if not multiple_D and not D_with_PCA:  
            if ('multi_reg' in dynamics_type or 'synth_multi_' in dynamics_type) and include_mask and include_D:
                D[D_mask == 0] = 0
        
            if norm_D_cols:
                D = D/((np.sum(D**2, 0).reshape((1,-1)) + 1e-19)**0.5)
            if null_D:
                D = nullify_part(D, percent0 = 30, axis = '1')
        else: # IE MULTIPLE Ds
        

      
            for i, D_i in enumerate(D):
             
                if ('multi_reg' in dynamics_type or 'synth_multi_' in dynamics_type) and include_mask and include_D:

                    D_i[D_masks[i] == 0] = 0

                if norm_D_cols:
                    zero_cols = np.sum(D_masks[i],0) == 0
                    D_i = D_i/((np.sum(D_i**2, 0)**0.5).reshape((1,-1)) + 1e-19)
                    D_i[:,zero_cols] = 0
                if null_D:
                    D_i = nullify_part(D_i, percent0 = 30, axis = '1')    
                    
                D[i]  = D_i
 

    if fix_x:
       if one_dyn:
           latent_dyn = xs
       else:
           latent_dyns = xs
              
       
    elif one_dyn:
        if latent_dyns_initialization == 'random':
            latent_dyn = np.random.rand(latent_dim, data.shape[1])
        else:
            latent_dyn = np.hstack([solve_lasso_problem(data[:,t], D, reg = lambda_x, update_type = params['update_c_type'], 
                                random_state = 0).reshape((-1,1)) for t in range(data.shape[1])])

        if  fix_x0:
            latent_dyn[:,0] = x0
            
       
          
    else: 
        if latent_dyns_initialization == 'random':
            latent_dyns = [np.random.rand(latent_dim, data_i.shape[1])
                           for i,data_i in enumerate(data)]
        else:
            latent_dyns = [np.hstack([solve_lasso_problem(data_i[:,t], D, reg = lambda_x,
                                                          update_type = params['update_c_type'], 
                                random_state = i).reshape((-1,1)) for t in range(data_i.shape[1])])
                           for i,data_i in enumerate(data)]
            
        if 'multi_reg' in dynamics_type and 'meso' not in dynamics_type and 'human' not in dynamics_type: 
            fig, ax = plt.subplots(figsize = (25,25))
            D_mask_nan = D_mask.copy()
            D_mask_nan[D_mask == 0] = np.nan
            sns.heatmap(pd.DataFrame(D_mask_nan, index = regions, 
                                     columns = np.repeat(unique_regions,latent_dim_per_region)), 
                        ax = ax)
            ax.set_xticks(np.arange(D_mask_nan.shape[1]))
            ax.set_xticklabels(np.repeat(unique_regions,latent_dim_per_region), fontsize = 6)
            ax.set_yticks(np.arange(D_mask_nan.shape[0]))
            ax.set_yticklabels(regions, fontsize = 6)
           
            
            plt.savefig(path_save + os.sep +  'D.png')
            plt.close()

        if  fix_x0:

            for i in range(len(latent_dyns)):
                latent_dyns[i][:,0] = x0[i]

      
      

  """
  CREATE INITIALIZE F
  """
  if len(F) == 0:  
      print(init_orth)      
      if init_orth:
          F = create_orth_F(num_subdyns,latent_dim, seed_f = seed_f , dist_type = dist_type_f)

      else:
          F              = [init_mat( size_mat = (latent_dim,latent_dim), dist_type = dist_type_f, normalize=True, r_seed = seed_f+i) 
                            for i in range(num_subdyns)]

          
  if len(clean_dyn) == 0 and len(data) > 1 and isinstance(data,list) and same_c :
      clean_dyn = np.mean(np.dstack(data),2)
      
         
              
  """
  Initialize Coeffs
  """
  if len(coefficients) == 0 and not fix_c: 
      if one_dyn or same_c:
          if start_sparse_c:
              coefficients   = init_mat((num_subdyns,n_times-1),dist_type = 'regional') * np.median(data)
          else:
              coefficients   = init_mat((num_subdyns,n_times-1)) * np.median(data)
      else:
          
          coefficients = [ init_mat((num_subdyns,data_s.shape[1]-1)) 
                          for i,data_s in enumerate(data)]
      if nullify_big_winds:
          if one_dyn:
              coefficients = check_complete_zero(coefficients)
          else:
              coefficients = [check_complete_zero(coefficients_s)
                              for coefficients_s in coefficients]
  if len(params) == 0:       params         = {'update_c_type':'inv','reg_term':0,'smooth_term':0}
  counter = 1
  
  error_reco_all            = np.inf*np.ones((1,error_order_max))
  error_reco_all_med        = np.inf*np.ones((1,error_order_max))
  error_reco_all_obs            = np.inf*np.ones((1,error_order_max))
  error_reco_all_med_obs        = np.inf*np.ones((1,error_order_max))
  error_reco_array      = np.inf*np.ones((1,max(error_order_max_display,error_order_max)))
  error_reco_array_med  = np.inf*np.ones((1,max(error_order_max_display,error_order_max)))
  
  if not include_D:
    if one_dyn:

      if latent_dyn.shape[1] == coefficients.shape[1] + 1:
          coefficients = np.hstack([coefficients, np.ones(coefficients.shape[0]).reshape((-1,1))])
      elif latent_dyn.shape[1] == coefficients.shape[1] + 1:
          coefficients = coefficients
      else:
          raise ValueError('not possible')
  
      cur_reco              = create_reco(latent_dyn=latent_dyn, coefficients= coefficients, F=F, 
                                          accumulation = acumulated_error)
    else:
      if same_c:
        cur_reco              = np.dstack([create_reco(latent_dyn=latent_dyn_i, coefficients= coefficients, F=F, accumulation = acumulated_error) for latent_dyn_i in latent_dyns])
      else:
          
        cur_reco              = [create_reco(latent_dyn=latent_dyn_i, coefficients= coefficients[samp_num], F=F,
                                             accumulation = acumulated_error)
                                 for samp_num, latent_dyn_i in enumerate(latent_dyns)]
 


                                                                                 
  counter_change_order  = 1
  if include_D:
    data_reco_error  = np.inf
  else:
    data_reco_error = -np.inf
  data_reco_error_list = [data_reco_error]
  if len(grad_vec_min_max) == 0: grad_vec_min_max = [0.99, 1.01]
  """
  Center dynamics
  """
  if center_dynamics:
      if one_dyn:  to_center_vals = -np.mean(data)
      else:        to_center_vals = [-np.mean(dyn) for dyn in data]
      if not  include_D:
          if one_dyn:  data = data + to_center_vals
          else:        
              data = [data_spec + to_center_vals[i] for i, data_spec in enumerate(data)]
      else:          
          if one_dyn:  
              latent_dyn = latent_dyn + to_center_vals
          else:        
              latent_dyns = [latent_dyn + to_center_vals[i] for i, latent_dyn in enumerate(latent_dyns)]
      
  else:
    to_center_vals = 0  
    
  grad_vec = np.linspace(grad_vec_min_max[0],grad_vec_min_max[1],error_order) 

     
  one_dyn_original = one_dyn   
  if 'multi_reg' in dynamics_type or 'synth_multi_' in dynamics_type:
      if not multiple_D:
          D_mask_nan = D_mask.copy()
          D_mask_nan[D_mask == 0] = np.nan
      else:
          D_masks_nan = D_masks.copy()
          for key,val in D_masks_nan.items():
            
              if 'meso' not in dynamics_type and 'human' not in dynamics_type:
                  val[D_masks[key] == 0] = np.nan

          
      
  if not one_dyn:
      fig, ax = plt.subplots(figsize = (25,25))
    
      sns.heatmap(latent_dyns[0], 
                  ax = ax, square = False, robust = True)
    
      fig.tight_layout()
      
      plt.savefig(path_save + os.sep +  'x.png')
      plt.close()
  

  


  sparse_cur_list = []
  """
  The loop !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  """    

  if error_while=='median': error_while_ar = error_reco_all_med
  if error_while=='mean': error_while_ar = error_reco_all
     
  i_patch = 0  
  save_locals = True


  while ((error_reco_all_med[0,error_order-1] > max_error) or (data_reco_error > max_data_reco) or error_order < error_order_max) and (counter < max_iter) :
      

    print('counter')  
    
    print(counter)
    print('er med')
    print(error_reco_all_med)

    

            
            
    if saving_graphs:

        if np.mod(counter, saving_graph_freq) == 0 : 

            if 'multi_reg' in dynamics_type or 'synth_multi_' in dynamics_type:              
                plot_mid(path_save , 'iter%d_'%counter, F, error_reco_array_med, coefficients, x = latent_dyns,
                             D = D, dynamics_type = dynamics_type, regions = regions,info_keep_order =  info_keep_order, sparse_cur_list = sparse_cur_list, 
                             data_reco_error_list = data_reco_error_list, latent_dim_per_region = latent_dim_per_region, 
                             cs_ground_truth = cs_ground_truth if 'cs_ground_truth' in locals() else [])
                
                
            else:
                plot_mid(path_save , 'iter%d_'%counter, F, error_reco_array_med, coefficients, x = latent_dyn,
                             D = D, dynamics_type = dynamics_type, cs_ground_truth = cs_ground_truth if 'cs_ground_truth' in locals() else [])                

            print('figs saved in ' + path_save)
          
    else:
       
        pass

    
    ### Store Iteration Results
    if return_evolution:
        store_iter_restuls['F'].append(F)
        store_iter_restuls['coefficients'].append(coefficients)
        store_iter_restuls['L1'].append(np.sum(np.abs(np.array(coefficients)),1))
    
    if error_while_ar[0,error_order-1] < max_error and  error_order < error_order_max and error_order_max != 1:
      error_order += 1
      counter_change_order = 1
      if redefine_step_f:
          step_f = step_f_original*(GD_decay**(error_order**2))   
      grad_vec = np.linspace(grad_vec_min_max[0],grad_vec_min_max[1],error_order) 
      if weights_orders_style == 'renew':
          weights_orders = np.linspace(1,1.2**error_order_max,error_order)[::-1]
      else:
          weights_orders = np.hstack([weights_orders, np.array([initial_weight_order_add])])
      weights_orders = weights_orders/np.sum(weights_orders)
          
      # Store restuls so far
      res_intermediate[counter - 1] = {'F': F,'coefficients':coefficients,
                                           'error_reco_array':error_reco_array, 'error_reco_array_med':error_reco_array_med, 
                                           'counter':counter}
      
    else:
      counter_change_order += 1
    

    
    """
    Patch (before infer x)
    """
    if include_patch and counter > 1:
          np.random.seed(counter)
          if one_dyn:
              if patch_size > latent_dyn.shape[1]: include_patch = False
              else:
                  latent_dyn_copy = latent_dyn.copy()
                  last_vals = latent_dyn.shape[1]-patch_size
                  latent_dyns_copy = latent_dyn.copy()
              
          else:
              if not np.array([patch_size < latent_dyn.shape[1] for latent_dyn in latent_dyns]).all():
                  include_patch = False
                  print('patch is too big')
              else:
                  shape_min = np.min([latent_dyn.shape[1] for latent_dyn in latent_dyns])
                  latent_dyns_copy = [latent_dyn.copy() for latent_dyn in latent_dyns]
                  last_vals = [latent_dyn.shape[1]-patch_size for latent_dyn in latent_dyns]
                

          if one_dyn:
              initial_vals = np.random.choice(np.arange(last_vals), num_patch)
              if num_patch == 1:
                  latent_dyns = latent_dyn_copy[:,initial_vals[0]:initial_vals[0]+patch_size] 
               
              
              else:
                  latent_dyns = [latent_dyn_copy[:,initial_val:initial_val+patch_size] for initial_val in initial_vals]
                  
                  one_dyn = False
                        
      
          else:      
              initial_valslists2list = lists2list(initial_vals)
              initial_vals = [np.random.choice(np.arange(last_val), num_patch) for last_val in last_vals]
              order_patches = lists2list([ [i
                                          for initial_val in initial_vals[i]]
                             for i,latent_dyn_copy in enumerate(latent_dyns_copy)])  
              latent_dyns = lists2list([ [latent_dyn_copy[:,initial_val:initial_val+patch_size] 
                                          for initial_val in initial_vals[i]]
                             for i,latent_dyn_copy in enumerate(latent_dyns_copy)])  
              raise ValueError('need to add back')
    

 

    """
    Update x / c
    """

    if infer_x_c_together and  include_D and (not fix_c) and (not fix_x) and dynamics_prior: # namely D != I

        if one_dyn:
            coefficients, latent_dyn =  update_c(F,[], params,random_state=seed,other_params=other_params_c,    
                                        include_identity =include_identity,coefficients=coefficients ,
                                    x_together = x_together, data_i = data, D = D, lambda_x = lambda_x, weight_observation_eq = weight_observation_eq)
            
        else:
            if one_dyn_original: # meaning that only that latent dyns is with pathes 
                for count_val, initial_val in enumerate(initial_vals):
                    coefficients_local, latent_dyn_local =  update_c(F,[], params,random_state=seed,other_params=other_params_c,    
                                        include_identity =include_identity,coefficients=coefficients[:,initial_val:initial_val+patch_size] ,
                                    x_together = x_together, data_i = data[:,initial_val:initial_val+patch_size], D = D, lambda_x = lambda_x, 
                                    weight_observation_eq = weight_observation_eq)
                    latent_dyns[count_val] = latent_dyn_local
                    coefficients[:,initial_val:initial_val+patch_size-1] = coefficients_local
                    
            else:

                for i, coefficients_spec in enumerate(coefficients):
                    
                    coefficients_local, latent_dyn_local =  update_c(F,[], params,random_state=seed,other_params=other_params_c,  
                                                                     include_identity =include_identity,coefficients=coefficients_spec ,
                                                                     x_together = x_together, data_i = data[i], D = D[i], lambda_x = lambda_x, weight_observation_eq = weight_observation_eq)
                    coefficients[i] = coefficients_local
                    latent_dyns[i] = latent_dyn_local 
  
 
            
                
    

    else: # namely - do not infer these together!
        # here is the update of x! to change if there is no dynamics prior
        if include_D and not fix_x:
            if not dynamics_prior:
                if one_dyn:
                    dyn_give = latent_dyn
                else:
                    dyn_give = latent_dyns
                    
                latent_dyn = infer_x_with_sibblings_only( data, D,x_former = dyn_give, 
                                            params_infer_x_no_prior = params_infer_x_no_prior)
 
                if not one_dyn:
                    latent_dyns = latent_dyn.copy()
            else:
            
                """
                infer x
                """
    
                if one_dyn: 
             
                    latent_dyn = update_X(D, data, latent_dyn, type_x_infer = type_x_infer ,
                                          lambda_x = lambda_x, F = F, coefficients = coefficients, 
                                 random_state = counter, params_x = params_update_x , 
                                 counter = counter, x0 = x0, use_new_est = use_new_est)
                    
                    
        
                elif one_dyn_original:
                    
                    latent_dyns = [
                        update_X(D, data[:,initial_val:initial_val+patch_size], latent_dyn = latent_dyns[count_val], 
                                 type_x_infer = type_x_infer ,
                                             lambda_x = lambda_x, F = F, coefficients = coefficients[:,initial_val:initial_val+patch_size], 
                                    random_state = counter, params_x = params_update_x , 
                                    counter = counter,  x0 = x0[count_val], use_new_est = use_new_est)
                        for count_val, initial_val in enumerate(initial_vals)]
                        
                        
                      
                        
                else: 
                    
                    if not multiple_D:
                        if checkEmptyList(x0):
                            x0 = [l[:,0] for l in latent_dyns]
                        latent_dyns = [
                            update_X(D, data_i, latent_dyn = latent_dyns[count_val], 
                                     type_x_infer = type_x_infer ,
                                                 lambda_x = lambda_x, F = F, coefficients = coefficients[count_val], 
                                        random_state = count_val, params_x = params_update_x , 
                                        counter = counter,  x0 = x0[count_val], use_new_est = use_new_est)
                            for count_val, data_i in enumerate(data)]
                    else:

                        if checkEmptyList(x0):
                            x0 = [np.random.rand(D[0].shape[1]) for _ in range(len(data))]
                        latent_dyns = [
                            update_X(D[count_val], data_i, latent_dyn = latent_dyns[count_val], 
                                     type_x_infer = type_x_infer ,
                                                 lambda_x = lambda_x, F = F, coefficients = coefficients[count_val], 
                                        random_state = count_val, params_x = params_update_x , 
                                        counter = counter,  x0 = x0[count_val], use_new_est = use_new_est)
                            for count_val, data_i in enumerate(data)]                    
    
                        

                    
                   
              
        
            #else:
            """
            Update coefficients with LASSO
            """
            if not fix_c: 

                F  = check_values(F, max_val = 10**3, min_val = -10**3, rep_nan = True, rep_0 = True, noise_0 = 0.2)
                coefficients  = check_values(coefficients, max_val = 10**3, min_val = -10**3, rep_nan = True, rep_0 = True, noise_0 = 0.2)
                D  = check_values(D, max_val = 10**3, min_val = -10**3, rep_nan = True, rep_0 = True, noise_0 = 0.2)
                if not one_dyn:
                    latent_dyns  = check_values(latent_dyns, max_val = 10**3, min_val = -10**3, rep_nan = True, rep_0 = True, noise_0 = 0.2)
                    
                    
                if one_dyn or not  one_dyn_original:
                    if one_dyn:
                        coefficients =  update_c_full(F, latent_dyn, params, latent_dyn,  seed,
                                            other_params_c, include_identity, one_dyn, same_c, acumulated_error,
                                            add_avg, wind_avg, coefficients)
                    else:
                        coefficients =  update_c_full(F, latent_dyns, params, latent_dyns,  seed,
                                            other_params_c, include_identity, one_dyn, same_c, acumulated_error,
                                            add_avg, wind_avg, coefficients)
                elif one_dyn_original:
                    coefficients_list =  [coefficients[:,initial_val:initial_val+patch_size] 
                                               for count_val, initial_val in enumerate(initial_vals)]
                    
                    coefficients_list_updated =  update_c_full(F, latent_dyns, params, latent_dyns,  seed,
                                            other_params_c, include_identity, one_dyn, same_c, acumulated_error,
                                            add_avg, wind_avg, coefficients_list) 
                    for count_val, initial_val in enumerate(initial_vals):
                        coeff_to_add = coefficients_list_updated[count_val]
                        if np.mean(np.abs(coeff_to_add)) < 1e-7:
                            coeff_to_add += np.random.rand(*coeff_to_add.shape)
                        coefficients[:, initial_val:  initial_val + patch_size -1] =  coeff_to_add
                        
                    
     
                
                    
    if add_avg and counter > 1 and  not fix_c: # to_average coefficients
        if one_dyn or one_dyn_original:
            coefficients = np.hstack([np.mean(coefficients[:, np.max([0,t - wind_avg ]): np.min([coefficients.shape[1], t + wind_avg ])], 1).reshape((-1,1))
                                  for t in np.arange(coefficients.shape[1])
                                  
                                  ]) 
        else:
    
            coefficients = [np.hstack([np.mean(coefficients_s[:, np.max([0,t - wind_avg ]): np.min([coefficients_s.shape[1], t + wind_avg ])], 1).reshape((-1,1))
                                  for t in np.arange(coefficients_s.shape[1])]) 
                            for coefficients_s in coefficients]
    
    F  = check_values(F, max_val = 10**3, min_val = -10**3, rep_nan = True, rep_0 = True, noise_0 = 0.2)
    coefficients  = check_values(coefficients, max_val = 10**3, min_val = -10**3, rep_nan = True, rep_0 = True, noise_0 = 0.2)
    D  = check_values(D, max_val = 10**3, min_val = -10**3, rep_nan = True, rep_0 = True, noise_0 = 0.2)
    if not one_dyn:
        latent_dyns  = check_values(latent_dyns, max_val = 10**3, min_val = -10**3, rep_nan = True, rep_0 = True, noise_0 = 0.2)
        
    """
    heree add saving
    """
    if to_save_mid and np.mod(counter, save_freq) == 0:

        if one_dyn:
            latent_dyns_save  = latent_dyn
        else:
            latent_dyns_save  = latent_dyns
        if save_comparison_to_ground_truth :
            if single_session <= -1:
                c_comp = ((cs_ground_truth[1][:,1] - coefficients[1][:,1] )**2).sum()
                D_comp = ((Ds_ground_truth[1][:,1] - D[1][:,1] )**2).sum()                
                F_comp = ((F_ground_truth[1][:,1] - F[1][:,1] )**2).sum()
                compare_g_truth = {'c_comp':c_comp, 'F_comp':F_comp, 'D_comp':D_comp}
            else:
                c_comp = ((cs_ground_truth[0][:,1] - coefficients[0][:,1] )**2).sum()
                D_comp = ((Ds_ground_truth[0][:,1] - D[0][:,1] )**2).sum()                
                F_comp = ((F_ground_truth[0][:,1] - F[0][:,1] )**2).sum()
                compare_g_truth = {'c_comp':c_comp, 'F_comp':F_comp, 'D_comp':D_comp}
                
        else:
            compare_g_truth = {}
        print(path_save + os.sep + 'iter%d.npy'%counter)        

        reg_type_on_c = str(params.get('update_c_type'))
        if single_session > -1:
            notes =  'if single session - the keys2nums is just 0! and nums2keys also!'
    
            or_key_new_key = data_load.get('or_key_new_key', 'no key')
        
            chosen_session ='all'
        else:
            notes = 'multi_session'
            or_key_new_key = 'identical'
            
            chosen_session = data_load.get('chosen_session','NA')
        np.save(path_save + os.sep + 'iter%d_reg_type_%s.npy'%(counter, reg_type_on_c),{'F': F,'coefficients':coefficients,
                                             'error_reco_array':error_reco_array, 'error_reco_array_med':error_reco_array_med, 
                                             'data_reco_error':data_reco_error_list, 'sparse_cur_list':sparse_cur_list, 
                                             'single_session': single_session, 'chosen_session': chosen_session,
                                             'counter':counter, 'latent_dyns_save': latent_dyns_save, 'D':D, 
                                             'D_with_lasso':D_with_lasso, 'D_graph_driven':D_graph_driven,
                                            'notes':notes, 'or_key_new_key':  or_key_new_key,
                                             'data_shape':data[0].shape, 'addi_save':addi_save, 'compare_g_truth': compare_g_truth})
        print('saved in %s'%(path_save + os.sep + 'iter%d.npy'%counter))
       
        
        if save_locals :
            input_args = {k: v for k, v in locals().items() if k != 'func' and not callable(v)}
            input_args = {k:v for k,v in input_args.items() if k != 'axs'}
            input_args = {key:val for key,val in input_args.items() 
                          if not isinstance(val,matplotlib.axes._axes.Axes)
                          and not isinstance(val,matplotlib.figure.Figure )}
            np.save(path_save + os.sep + 'params.npy', input_args)
   
            
            save_locals = False
            
            
            
            
    
    """
    Decay reg 
    """
    if params['update_c_type'] == 'lasso':
        params['reg_term'] = params['reg_term']*decaying_reg 
        
    """
    hard thres c
    """    
    if to_hard_thres_c and np.mod(counter, hard_thres_c_freq) == 0 and not fix_c:
        if one_dyn:
            raise ValueError('future')
        else:
            for i, c_i in enumerate(coefficients):
                for t in range(c_i.shape[1]):
                    Fx = np.hstack([(f_j @ latent_dyns[i][:,t].reshape((-1,1))).reshape((-1,1)) for j, f_j in enumerate(F)])
                    print(latent_dyns[i].shape)
                    if latent_dyns[i][:,t+1].sum() == 0:

                        latent_dyns[i][:,t+1] = np.linalg.pinv(D[i]) @ data[i][:,t+1].reshape((-1,1))
                        
                    if np.sum(c_i[:,t]) == 0:
                        #raise ValueError('c_i is 0 at %d'%t)
                        c_i[:,t] = c_i[:,t] + np.random.rand(*c_i[:,t].shape)
                    else:
                        c_t = infer_under_mask( latent_dyns[i][:,t+1], Fx, x_mask = [], k = k_hard_thres_c ,
                                           params_update_c = params, indices_mask = [], vec = c_i[:,t])
                        coefficients[i][:,t] = c_t
        

    """
    Update bias_out
    """

    if one_dyn:
        bias_out_val = np.zeros((data.shape[0], 1))
    else:
        bias_out_val = [np.zeros((data_spec.shape[0], 1)) for i,data_spec in enumerate(data)]
    
    """
    preparations for updating D
    """
    if D_graph_driven:
        if checkEmptyList(indices_regs) and include_mask:
            raise ValueError('you must provide indices_regs if D_graph_driven and include_maks')
        if include_mask:
            if checkEmptyList(H) :

                if isinstance(data, list): # multiple trials
                    if data[0].shape[0] != len(lists2list(indices_regs)):
                        raise ValueError('data must be the same len as regions_list but len(data) is %d and len(regions) is %d'%(data.shape[0], len(lists2list(indices_regs))))
    
                    H = make_kernel_3d(np.hstack(data),indices_regs, **D_graph_params)
                else: # meaning mask but not multi trial
                    if data.shape[0] != len(lists2list(indices_regs)):
                        raise ValueError('data must be the same len as regions_list but len(data) is %d and len(regions) is %d'%(data.shape[0], len(lists2list(indices_regs))))
                    H = make_kernel_3d(data, indices_regs, **D_graph_params)
    
            lambdas = [updateLambdasMat(D[indices_reg.min():indices_reg.max()+1,cols_blocks[i].min():cols_blocks[i].max()], H[i], params_graph,  params)
                       for i,indices_reg in enumerate(indices_regs)]
            
            
                
            
                
            
        else:
            if checkEmptyList(H): # namely no mask
                if isinstance(data, list):
                    H = make_kernel_2d(np.hstack(data), **D_graph_params)
                else:
                    H = make_kernel_2d(data, **D_graph_params)
            lambdas = updateLambdasMat(D, H, params_graph,params)
    else:
        if include_mask:
        
            lambdas = [l1_D]*num_regions 


        else:
            lambdas = []

   
    """
    Update D with gradient descent
    """    

   
    if include_D and not fix_D:
        if ( 'multi_reg' not in dynamics_type and 'synth_multi_' not in dynamics_type) or not include_mask :

          
          
            if one_dyn: 
                
                D = update_D_lasso(lambda_D,D, data, latent_dyn,  update_type_D, lambdas = lambdas, D_graph_driven = D_graph_driven, 
                                  params = params_D)
                if norm_D_cols and include_D:
                   D = D/((np.sum(D**2, 0)**0.5).reshape((1,-1)) + 1e-9)
        
            elif one_dyn_original:
                data_stack = np.hstack([data[:,initial_val:initial_val+patch_size] 
                                        for count_val, initial_val in enumerate(initial_vals)])
                latent_dyn_stack = np.hstack(latent_dyns)
                D = update_D_lasso(lambda_D,D, data_stack, latent_dyn_stack, update_type_D, lambdas = lambdas, D_graph_driven = D_graph_driven, 
                                  params = params_D)
                if norm_D_cols and include_D:
                   D = D/((np.sum(D**2, 0)**0.5).reshape((1,-1)) + 1e-9)
                
                               
            else:
                if update_D_based_on_one_trial > 0:
                    np.random.seed(counter)
                    choices = np.random.randint(len(data), size = update_D_based_on_one_trial)
                    for choice in choices:
                        D = update_D_lasso(lambda_D,D, data[choice], latent_dyns[choice],   update_type_D, 
                                           lambdas = lambdas, D_graph_driven = D_graph_driven, 
                                          params = params_D)
                        if norm_D_cols and include_D:
                           D = D/((np.sum(D**2, 0)**0.5).reshape((1,-1)) + 1e-9)
 
                
                else:
                    choices = np.arange(len(data))
                    for choice in choices:
                        D = update_D_lasso(lambda_D,D, data[choice], latent_dyns[choice], update_type_D, 
                                           lambdas = lambdas, D_graph_driven = D_graph_driven, 
                                          params = params_D)
                        if norm_D_cols and include_D:
                           D = D/((np.sum(D**2, 0)**0.5).reshape((1,-1)) + 1e-9)
                    #D = np.median(np.dstack([update_D(D, step_D, latent_dyns[i], data[i], reg1,reg_f, bias_out_val)                                              for i in range(len(data))]), 2)

        else: # NAMELY - WITH MASK

                
            if not multiple_D:
                for block_count, block_cols in enumerate(cols_blocks):
                    rows_blocks = indices_regs[block_count]
                    row_min = rows_blocks.min()
                    row_max = rows_blocks.max()
                    col_min = block_cols.min()
                    col_max = block_cols.max()
                    
                    cur_block = D[row_min: row_max, col_min:col_max]
                    
                    
                    """
                    now apply it only on the block
                    """
               
                    if one_dyn: 
                        cur_block = update_D_lasso(lambda_D,cur_block, data[row_min: row_max,:], latent_dyn[col_min:col_max,:], 
                                                   update_type_D,  lambdas = lambdas[block_count], D_graph_driven = D_graph_driven, 
                                          params = params_D)
                        if norm_D_cols and include_D:
                            cur_block = cur_block/((np.sum(cur_block**2, 0)**0.5).reshape((1,-1)) + 1e-9)
           
                    elif one_dyn_original:
                        data_stack = np.hstack([data[row_min: row_max,initial_val:initial_val+patch_size] 
                                                for count_val, initial_val in enumerate(initial_vals)])
                        latent_dyn_stack = np.hstack(latent_dyns)[col_min:col_max,:]
                        cur_block = update_D_lasso(lambda_D,cur_block, data_stack, latent_dyn_stack, update_type_D,  lambdas = lambdas[block_count], D_graph_driven = D_graph_driven, 
                                          params = params_D)
                        if norm_D_cols and include_D:
                           cur_block = cur_block/((np.sum(cur_block**2, 0)**0.5).reshape((1,-1)) + 1e-9)
                    
                                       
                    else:
                        if update_D_based_on_one_trial > 0:
                            np.random.seed(counter)
                            choices = np.random.randint(len(data), size = update_D_based_on_one_trial)
                            for choice in choices:
    
                                cur_block = update_D_lasso(lambda_D,cur_block, data[choice][row_min: row_max,:], 
                                                           latent_dyns[choice][col_min:col_max,:], 
                                                   update_type_D,  lambdas = lambdas[block_count], 
                                                   D_graph_driven = D_graph_driven, 
                                                  params = params_D)
                                if norm_D_cols and include_D:
                                   cur_block = cur_block/((np.sum(cur_block**2, 0)**0.5).reshape((1,-1)) + 1e-9)
     
                            
    
                        else:
                            choices = np.arange(len(data))
                            for choice in choices:
                                cur_block = update_D_lasso(lambda_D,cur_block, data[choice][row_min: row_max,:], latent_dyns[choice][col_min:col_max,:], 
                                                    update_type_D, lambdas = lambdas[block_count], D_graph_driven = D_graph_driven, 
                                                  params = params_D)
                                if norm_D_cols and include_D:
                                   cur_block = cur_block/((np.sum(cur_block**2, 0)**0.5).reshape((1,-1)) + 1e-9)
            else:    
         
                D_or = D.copy()
                indices_regs_or = indices_regs.copy()

                
                for i,D in enumerate(D_or):     

                    indices_regs = indices_regs_or[i]

                    for block_count, block_cols in enumerate(cols_blocks):
                        try:
                            rows_blocks = indices_regs[block_count]
                        except:
                            print('??')


                        if len(rows_blocks) > 0:
                            row_min = rows_blocks.min()
                        
                            row_max = rows_blocks.max() + 1
                            
                            if len(block_cols) > 1:
                                col_min = block_cols.min()
                                col_max = block_cols.max()
                                
                                cur_block = D[row_min: row_max, col_min:col_max]
                            else:
                                col_min = block_cols[0].min()
                                col_max = block_cols[0].max() + 1
                                cur_block = D[row_min: row_max, block_cols[0]].reshape((-1,1))

                            """
                            now apply it only on the block
                            """
                       
                            if one_dyn: 
                                if  D_with_lasso:
                                    cur_block = update_D_lasso(lambda_D, cur_block, data[row_min: row_max,:], latent_dyn[col_min:col_max,:], 
                                                               update_type_D,  lambdas = lambdas[block_count], D_graph_driven = D_graph_driven, 
                                                      params = params_D)
                                else:
                                    cur_block, step_D = update_D(cur_block, step_D, latent_dyns[col_min:col_max,:], data[row_min: row_max,:], reg1,reg_f, bias_out_val) 
                                    

                                if norm_D_cols and include_D:
                                    cur_block = cur_block/((np.sum(cur_block**2, 0)**0.5).reshape((1,-1)) + 1e-9)
                   
                            elif one_dyn_original:
                                data_stack = np.hstack([data[row_min: row_max,initial_val:initial_val+patch_size] 
                                                        for count_val, initial_val in enumerate(initial_vals)])
                                latent_dyn_stack = np.hstack(latent_dyns)[col_min:col_max,:]
                                
                                if D_with_lasso:
                                    cur_block = update_D_lasso(lambda_D,cur_block, data_stack, latent_dyn_stack, update_type_D,  
                                                               lambdas = lambdas[block_count], D_graph_driven = D_graph_driven, 
                                                  params = params_D)
                                else:
                                    cur_block, step_D  = update_D(cur_block, step_D,latent_dyn_stack, data_stack, reg1,reg_f, bias_out_val) 
                                if norm_D_cols and include_D:
                                   cur_block = cur_block/((np.sum(cur_block**2, 0)**0.5).reshape((1,-1)) + 1e-9)
                                #pdate_D(D, step_D, latent_dyn_stack, data_stack, reg1,reg_f, bias_out_val) 
                                               
                            else: # I.E. MULTIPLE DS. MEANING A D FOR EACH. SO THE DATA IS ACCORDINGLY
                                if D_with_lasso:
                                    if update_D_based_on_one_trial > 0:
                                        np.random.seed(counter)
                                        # fix here future
                                        
                                        choices = np.random.randint(len(data), size = update_D_based_on_one_trial)
                                        #for choice in choices:
                                        choice = i

                                        cur_block = update_D_lasso(lambda_D,cur_block, data[choice][row_min: row_max,:], 
                                                                   latent_dyns[choice][col_min:col_max,:], 
                                                           update_type_D,  lambdas = lambdas[block_count], 
                                                           D_graph_driven = D_graph_driven, 
                                                          params = params_D)
                                        if norm_D_cols and include_D:
                                           cur_block = cur_block/((np.sum(cur_block**2, 0)**0.5).reshape((1,-1)) + 1e-9)

                                    else: # I.E. UPDATE BASED ON 1 TRALS

                                            
                                        choice = i# hhhhhhh#s = np.arange(len(data)) # here is the problem
                                        #for choice in choices:
                                        cur_block = update_D_lasso(lambda_D,cur_block, data[choice][row_min: row_max,:], latent_dyns[choice][col_min:col_max,:], 
                                                            update_type_D, lambdas = lambdas[block_count], D_graph_driven = D_graph_driven, 
                                                          params = params_D)
                                        if norm_D_cols and include_D:
                                           cur_block = cur_block/((np.sum(cur_block**2, 0)**0.5).reshape((1,-1)) + 1e-9)
                                        
                                            #update_D(D, step_D, latent_dyns[i], data[i], reg1,reg_f, bias_out_val)  #for i, D_i in enumerate(D)]
                                else:

                                    cur_block , step_D  = update_D(cur_block , step_D, latent_dyns[i][col_min:col_max,:], data[i][row_min: row_max,:], reg1,reg_f, bias_out_val)  
                   
                                        
                                    if norm_D_cols and include_D:
                                           cur_block = cur_block/((np.sum(cur_block**2, 0)**0.5).reshape((1,-1)) + 1e-9)
              
                      
                            if len(block_cols) > 1:
                                D[row_min: row_max, col_min:col_max] = cur_block                 
                            else:
                                D[row_min: row_max, block_cols[0]] = cur_block.flatten()

                D = D_or.copy()
                indices_regs = indices_regs_or.copy()
                
    print('udapted D!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    """
    OBSERVATION ERROR
    """
    if include_D:
      if one_dyn_original:       
          data_reco_error = np.mean((data - D @ latent_dyn)**2)
      elif not multiple_D:
          data_reco_error = np.median( [ np.mean((data_i - D @ latent_dyns[i])**2)  
                                                      for i, data_i in enumerate(data) ])
      else:             
          data_reco_error = np.median( [ np.mean((data_i - D[i] @ latent_dyns[i])**2)  
                                                      for i, data_i in enumerate(data) ])
      data_reco_error_list.append(data_reco_error)
     
       
    step_D *= step_D_decay
    """
    Update bias_term
    """

   
    if one_dyn:
        bias_val = np.zeros((latent_dyn.shape[0], 1))
    else:
        if same_c:                
            bias_val = np.zeros((latent_dyns[0].shape[0], 1))
        else:
            bias_val = [np.zeros((latent_dyn.shape[0], 1)) for i,latent_dyn in enumerate(latent_dyns)]

   
   
    """
    Update F with gradient descent
    """
    
    if not fix_f:
        count_gd_step = 0
        while count_gd_step < num_gradient_steps:
            if one_dyn or update_D_based_on_one_trial:
                if update_D_based_on_one_trial and not one_dyn:
                    np.random.seed(counter)
                    i = np.random.randint(len(latent_dyns))
                    coefficients_local = coefficients[i]
                    latent_dyn = latent_dyns[i]
                else:
                    coefficients_local = coefficients
                F, step_f, count_gd_step =  update_f(latent_dyn,F,coefficients_local,step_f, acumulated_error, 
                               error_order, action_along_time, weights_orders,                
                              normalize_eig ,  bias_val,  include_identity,  type_norm ,
                              min_step_f, count_gd_step , one_dyn or update_D_based_on_one_trial, decay_decay=decay_decay
                              , GD_decay=GD_decay
                              , counter = counter, num_subdyns = num_subdyns, same_c = same_c,**parameters_f_wise_step)
                
            elif one_dyn_original:
                coefficients_stack =  np.hstack([coefficients[:,initial_val:initial_val+patch_size] 
                                           for count_val, initial_val in enumerate(initial_vals)])
                latent_dyn_stack = np.hstack(latent_dyns)
                F, step_f, count_gd_step =  update_f(latent_dyn_stack,F,coefficients_stack,step_f, acumulated_error, 
                               error_order, action_along_time, weights_orders,                
                              normalize_eig ,  bias_val,  include_identity,  type_norm ,
                              min_step_f, count_gd_step , one_dyn_original, decay_decay=decay_decay, GD_decay=GD_decay
                              , counter = counter, num_subdyns = num_subdyns, same_c = same_c,**parameters_f_wise_step)
                
            else:

                F, step_f, count_gd_step =  update_f(latent_dyns,F,coefficients,step_f, acumulated_error, 
                               error_order, action_along_time, weights_orders,                
                              normalize_eig ,  bias_val,  include_identity,  type_norm ,
                                 min_step_f, count_gd_step , one_dyn, decay_decay=decay_decay, GD_decay=GD_decay
                               , counter = counter, num_subdyns = num_subdyns, same_c = same_c, latent_dyns = latent_dyns,
                               **parameters_f_wise_step)

               
                
            
        if sparse_f:
            if sparse_f_params['percent0'] < sparsity_on_f_max:
                sparse_f_params['percent0'] += increase_in_sparsity_f
                
            
            F = [nullify_part(f, **sparse_f_params) for f in F]
            
        if normalize_F:
            F = [f/np.linalg.norm(f, 2)     for f in F]
    """
    Patch back
    """
    if include_patch and counter > 1:
          one_dyn = one_dyn_original
          if one_dyn:
              if num_patch == 1:
                  latent_dyn[:,initial_vals[0]:initial_vals[0]+patch_size] =   latent_dyns
              else:    
                  for i,initial_val in enumerate(initial_vals):
                      latent_dyn[:,initial_val: initial_val+patch_size] =   latent_dyns[i]
               
              
          else:

              order_patches = lists2list([ [i
                                          for initial_val in initial_vals[i]]
                             for i,latent_dyn_copy in enumerate(latent_dyns_copy)])  
              latent_dyns = lists2list([ [latent_dyn_copy[:,initial_val:initial_val+patch_size] 
                                          for initial_val in initial_vals[i]]
                             for i,latent_dyn_copy in enumerate(latent_dyns_copy)])  
              for num_dyn in range(len(latent_dyns_copy)): #oroiginal small narrow count
               
                  for j, order_patch in enumerate( order_patches): # go over the wide list
                      if order_patch == num_dyn:
                          ini = initial_valslists2list[j]
                          latent_dyns_copy[order_patch][:,ini:ini+patch_size] = latent_dyns[j]
                          
                      
                
                      
              latent_dyns = latent_dyns_copy.copy()
              raise ValueError('need to add back')
              
                  
    """
    update error
    """
    if error_order !=1 :    weights_orders = weights_orders * grad_vec

    if one_dyn :
      mid_reco = latent_dyn

    else:
      mid_reco = latent_dyns
     

    error_reco_all = np.inf*np.ones((1,max(error_order_max_display,error_order_max))) #[]

    error_reco_all_med = np.inf*np.ones((1,max(error_order_max_display,error_order_max)))
    
      
    """
    update how sparse c is
    """
    if one_dyn:
        sparse_cur = np.mean(np.sum(np.abs(coefficients),0))
    else:
        sparse_cur = np.mean([np.mean(np.sum(np.abs(coefficients_i),0)) for coefficients_i in coefficients])
    sparse_cur_list.append(sparse_cur)
    
    for n_error_order in range(max(error_order_max_display,error_order_max)):   
        
        if one_dyn:
            try:
                mid_reco = create_reco(mid_reco, coefficients, F, acumulated_error)
                error_reco = np.mean((latent_dyn -mid_reco)**2)
                error_reco_all[0,n_error_order] = error_reco
                error_reco_all_med[0,n_error_order] = np.median((latent_dyn -mid_reco)**2)
            except:
                print('mid_reco does not work')
        else:

            if same_c or len(clean_dyn) > 0 :
                if same_c:
                    mid_reco = [create_reco(mid_reco_spec, coefficients, F, acumulated_error) for mid_reco_spec in mid_reco]
                else:
                    mid_reco = [create_reco(mid_reco_spec, coefficients[:,:,i], F, acumulated_error) for i,mid_reco_spec in enumerate(mid_reco)]
           
                error_reco = np.mean((clean_dyn.reshape((clean_dyn.shape[0], clean_dyn.shape[1],1))-np.dstack(mid_reco))**2)
                error_reco_all[0,n_error_order] = error_reco
                error_reco_all_med[0,n_error_order] = np.median((clean_dyn.reshape((clean_dyn.shape[0], clean_dyn.shape[1],1))-np.dstack(mid_reco))**2)
                
               
            else:
     
                
              
                
                mid_reco = [create_reco(mid_reco_spec, coefficients[i], F, acumulated_error) for i,mid_reco_spec in enumerate(mid_reco)]
                error_reco = np.mean([np.mean((latent_dyns[i] - mid_reco[i])**2) for  i in np.arange(len(mid_reco))] )
                
                error_reco_all[0,n_error_order] = error_reco
                error_reco_all_med[0,n_error_order] = np.median([np.median((latent_dyns[i] - mid_reco[i])**2) for  i in np.arange(len(mid_reco))] )
                
                
        if error_while=='median': error_while_ar = error_reco_all_med
        if error_while=='mean': error_while_ar = error_reco_all
      

    error_reco_array = np.vstack([error_reco_array,np.array(error_reco_all).reshape((1,-1))])
    error_reco_array_med = np.vstack([error_reco_array_med,np.array(error_reco_all_med).reshape((1,-1))])
    
    
    if np.mean(np.abs(np.diff(error_reco_array[-num_no_change:,:],axis = 0))) < epsilon_error_change and to_mix_F:
        if mix_f_method == 'all':
            F = [f_i + sigma_mix_f*np.random.randn(f_i.shape[0],f_i.shape[1]) for f_i in F]
        elif mix_f_method == 'rand':
            np.random.seed(counter)
            rand_dyn = np.random.randint(low = 0, high = len(F))
            F[rand_dyn] = F[rand_dyn] + sigma_mix_f*np.random.randn(F[rand_dyn].shape[0],F[rand_dyn].shape[1])
            
        else:
            raise NameError('Invalid mix_f_method')
        print('mixed F')

    if to_print:
        print('Error order %s'%str(error_order))
        print('f step = %s'%str(step_f))
        print(counter)
        if error_while == 'mean':
            print('Error Mean:    ' + '; '.join(['order' + str(i) + '=' + str(error_reco_all[0,i]) for i in range(   min(len(error_reco_all[0]),error_order_max_display))]))
        print('Error med:' + '; '.join(['order' + str(i) + '=' + str(error_reco_all_med[0,i]) for i in range(min(len(error_reco_all_med[0]),error_order_max_display) )]))
        if include_D:
            print('Error recy y:' + '; '.join(['order' + str(j) + '=' + str(data_reco_error) for j in range(1)]))
            
    if to_save_mid and np.mod(counter, save_freq) == 0:

        if one_dyn:
            latent_dyns_save  = latent_dyn
        else:
            latent_dyns_save  = latent_dyns
        if save_comparison_to_ground_truth:
            if single_session <= -1:
                c_comp = ((cs_ground_truth[1][:,1] - coefficients[1][:,1] )**2).sum()
                D_comp = ((Ds_ground_truth[1][:,1] - D[1][:,1] )**2).sum()                
                F_comp = ((F_ground_truth[1][:,1] - F[1][:,1] )**2).sum()
                compare_g_truth = {'c_comp':c_comp, 'F_comp':F_comp, 'D_comp':D_comp}
            else:
                c_comp = ((cs_ground_truth[0][:,1] - coefficients[0][:,1] )**2).sum()
                D_comp = ((Ds_ground_truth[0][:,1] - D[0][:,1] )**2).sum()                
                F_comp = ((F_ground_truth[0][:,1] - F[0][:,1] )**2).sum()
                compare_g_truth = {'c_comp':c_comp, 'F_comp':F_comp, 'D_comp':D_comp}
                
        else:
            compare_g_truth = {}
        print(path_save + os.sep + 'iter%d.npy'%counter)        
        #print('ok?!?!?!')
        #input('ok?!?!?!?!')
        reg_type_on_c = str(params.get('update_c_type'))
        np.save(path_save + os.sep + 'iter%d_reg_type_%s.npy'%(counter, reg_type_on_c),{'F': F,'coefficients':coefficients,
                                             'error_reco_array':error_reco_array, 'error_reco_array_med':error_reco_array_med, 
                                             'data_reco_error':data_reco_error_list, 'sparse_cur_list':sparse_cur_list, 
                                             'params_infer_x_no_prior':params_infer_x_no_prior,
                                                                     'counter':counter, 'latent_dyns_save': latent_dyns_save, 'D':D, 
                                             'data_shape':data[0].shape, 'addi_save':addi_save, 'compare_g_truth': compare_g_truth})
        print('saved in %s'%(path_save + os.sep + 'iter%d.npy'%counter))
        #input('save!')
        
        if save_locals :
            input_args = {k: v for k, v in locals().items() if k != 'func' and not callable(v)}
            input_args = {k:v for k,v in input_args.items() if k != 'axs'}
            input_args = {key:val for key,val in input_args.items() 
                          if not isinstance(val,matplotlib.axes._axes.Axes)
                          and not isinstance(val,matplotlib.figure.Figure )}
            np.save(path_save + os.sep + 'params.npy', input_args)
            #np.save(path_save + os.sep + 'params.npy',locals() )
            
            save_locals = False

    counter += 1
    #print(counter)

  if include_patch:
      if one_dyn: 
          latent_dyn = latent_dyn_copy.copy()
          if num_patch > 1:
              one_dyn = False 
      else: latent_dyns = latent_dyns_copy.copy()
      
  if counter == max_iter: 
      print('Arrived to max iter')
      print('Counter %s'%str(counter))
  if not fix_c:
      if one_dyn:
          if include_last_up:
              coefficients = update_c(F, latent_dyn,params,  
                                      {'reg_term': 0, 'update_c_type':'inv','smooth_term' :0, 'num_iters': 10, 'threshkind':'soft'},
                                      include_identity = include_identity)
          else:
              coefficients = update_c(F, latent_dyn, params,other_params=other_params_c)  
      else:
          if same_c:
              if include_last_up:
                  coefficients = update_c(F, clean_dyn,params,  
                                      {'reg_term': 0, 'update_c_type':'inv','smooth_term' :0, 'num_iters': 10, 'threshkind':'soft'},
                                      include_identity = include_identity)
                
              else:
                  coefficients = update_c(F, clean_dyn, params,other_params=other_params_c, include_identity = include_identity)  
          else:
              if include_last_up:
                  coefficients = [update_c(F, latent_dyn,params, 
                                          {'reg_term': 0, 'update_c_type':'inv','smooth_term' :0, 'num_iters': 10, 'threshkind':'soft'},
                                          include_identity = include_identity ) for latent_dyn in latent_dyns]
                  
              else:
                  print('coeffs before')
                  print(coefficients)
            
                  coefficients = [update_c(F, latent_dyn,params, other_params=other_params_c,
                                           include_identity = include_identity ) for latent_dyn in latent_dyns]
                  
                  print('coeffs after')
                  print(coefficients)
                  print('kg;dfk;lkdg;lkkkkkkkkkkkkkkkkkkkkkk')
                  #input('ojkjkjkj')
                          
  if return_evolution:
    store_iter_restuls['F'].append(F)
    store_iter_restuls['coefficients'].append(coefficients)
    store_iter_restuls['L1'].append(np.sum(np.abs(coefficients),1))
  
  
  if center_dynamics:
      if include_D:
          if one_dyn:  bias_out_val = bias_out_val - to_center_vals
          else:        
              bias_out_val = [bias_out_val - to_center_vals[i] for i in range(len(to_center_vals))]
      else:          
          if one_dyn:  bias_val = bias_val + to_center_vals
          else:   
              if same_c:
                  bias_val = [bias_val + to_center_vals[i] for i in range(len(to_center_vals))]
              else:
                  bias_val = [bias_val[i] + to_center_vals[i] for i in range(len(to_center_vals))]
          
  additional_return = {'bias_val': bias_val, 'to_center_vals': to_center_vals,'bias_out_val':bias_out_val,
                       'error_reco_all_med': error_reco_all_med, 'counter':counter, 'res_intermediate':res_intermediate,'min_step_f ':min_step_f }   
  


  if saving_graphs:



        if 'multi_reg' in dynamics_type or 'synth_multi_' in dynamics_type:
          
            plot_mid(path_save , 'final', F, error_reco_array_med, coefficients, x = latent_dyns,
                         D = D, dynamics_type = dynamics_type, regions = regions,info_keep_order =  info_keep_order, sparse_cur_list = sparse_cur_list, 
                         data_reco_error_list = data_reco_error_list, latent_dim_per_region = latent_dim_per_region)
        else:
            plot_mid(path_save , 'final', F, error_reco_array_med, coefficients, x = latent_dyn,
                         D = D, dynamics_type = dynamics_type)                

        print('figs saved in ' + path_save + '.!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      
  else:
      #print('did not save graph')
      pass

  
  if not return_evolution:
      print('NOT return evolution type')
      if not include_D: D = [];
      if one_dyn:      return coefficients, F, latent_dyn, error_reco_array, error_reco_array_med,D,additional_return
      else:  return coefficients, F, latent_dyns, error_reco_array, error_reco_array_med,D,additional_return
  else:
      print('return evolution type')
      if not include_D: D = [];

      if one_dyn:      return coefficients, F, latent_dyn, error_reco_array, error_reco_array_med #,D,store_iter_restuls, additional_return
      else:  return coefficients, F, latent_dyns, error_reco_array, error_reco_array_med,D,store_iter_restuls,additional_return
      



    
def merge_dicts(list_of_dicts, dict_01 = {}):
    """
    Merge a list of dictionaries into a single dictionary.
    
    This function takes a list of dictionaries and merges them into a single dictionary.
    It can handle merging any number of dictionaries in the list.
    
    Args:
        list_of_dicts (list): A list of dictionaries to be merged.
        dict_01 (dict, optional): An optional initial dictionary to start the merging process.
            Defaults to an empty dictionary.
    
    Returns:
        dict: A dictionary containing all the key-value pairs from the input dictionaries
        in the list merged together.
    
    Example:
        dict_list = [{'a': 1, 'b': 2}, {'b': 3, 'c': 4}, {'d': 5}]
        result = merge_dicts(dict_list)
        # Output: {'a': 1, 'b': 3, 'c': 4, 'd': 5}
    """
    if len(list_of_dicts) == 1:
            return {**dict_01, **list_of_dicts[0]}
    else:
        dict_01 =  {**dict_01, **{**list_of_dicts[0], **list_of_dicts[1]}}
    if len(list_of_dicts) == 2:
        return dict_01
    
    else:
        return merge_dicts(list_of_dicts[2:], dict_01)



#%% Saving
  
def check_save_name(save_name, invalid_signs = '!@#$%^&*.,:;', addi_path = [], sep=sep)  :
    """
    Check if the name is valid
    """
    for invalid_sign in invalid_signs:   save_name = save_name.replace(invalid_sign,'_')
    if len(addi_path) == 0:    return save_name
    else:   
        path_name = sep.join(addi_path)
        return path_name +sep +  save_name

def save_file_dynamics(save_name, folders_names,to_save =[],
                       invalid_signs = '!@#$%^&*.,:;', sep  = sep , type_save = '.npy',
                       path_name = ''):
    """
    Save dynamics & model results
    """                  
    save_name = check_save_name(save_name, invalid_signs)
    if len(path_name) == 0:
        path_name = os.getcwd() + os.sep + sep.join(folders_names)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    if type_save == '.npy':
        if not save_name.endswith('.npy'): save_name = save_name + '.npy'
        np.save(path_name +sep +  save_name, to_save)
        print('saved in '+ path_name +sep +  save_name)
    elif type_save == '.pkl':
        if not save_name.endswith('.pkl'): save_name = save_name + '.pkl'
        dill.dump_session(path_name +sep +  save_name)


def load_pickle(path):
    """
    This function loads a pickled file from a given path.
    
    Args:
    path (str): The path where the pickled file is located.
    
    Returns:
    dct (dict): The pickled file in a dictionary format.
    """
    with open(path,'rb') as f:
        dct = pickle.load(f)
    return dct
    

def saveLoad(opt,filename):
    """
    This function either saves or loads a pickled file based on the given option.
    
    Args:
    opt (str): The operation option. It can be either 'save' or 'load'.
    filename (str): The file name of the pickled file to either save to or load from.
    
    Returns:
    None
    """

    global calc
    if opt == "save":
        f = open(filename, 'wb')
        pickle.dump(calc, f, 2)
        f.close
     
    elif opt == "load":
        f = open(filename, 'rb')
        calc = pickle.load(f)
    else:
        print('Invalid saveLoad option')
        
def load_vars(folders_names ,  save_name ,sep=sep , ending = '.pkl',full_name = False):
    """
    Load results previously saved
    Example:
        load_vars('' ,  'save_c.pkl' ,sep=sep , ending = '.pkl',full_name = False)
    """
    if full_name: 
        dill.load_session(save_name)    
    else:
        if len(folders_names) > 0: path_name = sep.join(folders_names)
        else: path_name = ''
      
        if not save_name.endswith(ending): save_name = '%s%s'%(save_name , ending)
        dill.load_session(path_name +sep +save_name)

    
        

 
def str2bool(str_to_change):
    """
    Transform 'true' or 'yes' to True boolean variable 
    Example:
        str2bool('true') - > True
    """
    if isinstance(str_to_change, str):
        str_to_change = (str_to_change.lower()  == 'true') or (str_to_change.lower()  == 'yes')  or (str_to_change.lower()  == 't')
    return str_to_change


#%% Plots
def compare_orders(latent_dyn, coefficients, F, max_delay_plot, max_to_run, to_plot = True, axs = [], params_plot = {},interval_show = 0):
  """
  Compare the reconstruction of the dynamics under different reconstruction orders
  """
  if interval_show == 0 : interval_show = int(max_delay_plot/10)
  max_to_run = np.max([max_delay_plot, max_to_run])
  delays_options = np.arange(0,max_delay_plot,interval_show)
  if isinstance(axs,list):
    if len(axs) == 0:
        if latent_dyn.shape[0] == 3:
            fig, axs = plt.subplots(2,int(np.ceil(len(delays_options)/2)),figsize = (35,15), subplot_kw={'projection':'3d'})  #, sharex  = True, sharey = True
        elif latent_dyn.shape[0] == 2:
            fig, axs = plt.subplots(2,int(np.ceil(len(delays_options)/2)),figsize = (35,15))
        else:
            raise ValueError('Invalid dimension for the dynamics')
        
  axs_flat = axs.flatten()
  mid_reco = latent_dyn
  counter_plot =0
  error_orders = [0]
  for level_reco in range(max_to_run+1):
    mse = (np.mean((latent_dyn - mid_reco)**2))**0.5
    if to_plot:
      if level_reco in delays_options:    
        visualize_dyn(mid_reco,axs_flat[counter_plot],params_plot); axs_flat[counter_plot].set_title('Reconstructed order %g, rmse:%g'%(level_reco, mse))
        counter_plot += 1    
    mid_reco = create_reco(mid_reco,coefficients, F)
    error_orders.append(mse)
  plt.subplots_adjust(hspace = 0.2, wspace=0.2)
  fig,ax = plt.subplots()
  ax.plot(error_orders,'*-'); add_labels(ax, xlabel ='Reconstruction Order', ylabel = 'rMSE',zlabel = None); ax.set_yscale('log'); ax.axhline(1,color = 'r', ls = '--',alpha = 0.3)
  plt.suptitle('Reconstruction under different reconstruction orders',fontsize = 16)  
  plt.subplots_adjust()
  return error_orders, mid_reco, level_reco    

def find_closest(vec1, vec2, metric = 'mse'):
    """
    find to which elements in vec2 each element in vec1 is the closest
    """
    if metric == 'mse':
        tiled_vec1 = np.tile(vec1.reshape((1,-1)), [len(vec2),1]) 
        tiled_vec2 = np.tile(vec2.reshape((1,-1)), [len(vec1),1]).T
        v1_closest_to_v2_args = np.argmin((tiled_vec1 - tiled_vec2)**2, 1)
        v1_closest_to_v2 = vec1[v1_closest_to_v2_args]
        return v1_closest_to_v2, v1_closest_to_v2_args


#%% Compare Initial Conditions
def cal_f_lst(F_init_list, ind = -1):
    """
    Extract the final results for F in cases were the training returned the evolution results as well
    """
    F_lasts = [f_list_i[ind] for f_list_i in F_init_list]
    return F_lasts

def plot_f_different_initialization(F_init_list, ax = [], ind = -1,annot = True, to_plot = True):
    """
    Plot the set of sub-dynamics that were obtained under different initializations
    """
    if isinstance(F_init_list[0][0],list):
        F_lasts = cal_f_lst(F_init_list, ind = ind)
    else:
        F_lasts = F_init_list
    if isinstance(ax,list):
        if len(ax) == 0 and to_plot:
            fig, ax = plt.subplots(len(F_init_list),len(F_lasts[0]), sharex = True,sharey = True,figsize = (len(F_lasts[0])*5,len(F_init_list)*5))        
    if to_plot:
        [plot_subs(F_last,ax[i,:],annot = annot) for i,F_last in enumerate(F_lasts)]
    return F_lasts

def match_corrs(F1,F2,c2 = []):
    """
    Match different pairs of sub-dynamics based on correlation. Organize F2 by F1
    Inputs:
        F1   = list of np.arrays, each np.array is kXk
        F2   = list of np.arrays, each np.array is kXk
        c2   = the coefficients associated with the sub-dynamics F2
    Outputs:
        F1  = same as input
        F2_org = ordered list of the F2 sub-dynamics, ordered according to the correlation with F1
        c2 = ordered c2, ordered according to the correlation with F1
    """
    store_corr = np.zeros((len(F1),len(F2)))
    store_best = {} # keys are inds of F1, vals of F2
    for i1, f1_i in enumerate(F1):
        for i2, f2_i in enumerate(F2):
            corr_cur = spec_corr(f1_i.flatten(), f2_i.flatten())
            store_corr[i1,i2] = corr_cur
    while np.sum(store_corr) > 0:
        B = np.unravel_index(np.argmax(store_corr, axis=None), store_corr.shape)
        #store_best.append(B)
        store_best[B[0]]   = B[1]
        store_corr[B[0],:] = 0
        store_corr[:,B[1]] = 0

    order_keys = np.sort(np.array(list(store_best.keys())))
    F2_org = [F2[store_best[key]] for key in order_keys]
    if len(c2) > 0:    c2 = [c2[store_best[key],:] for key in order_keys]
    return F1, F2_org, c2
        
        
        
def check_initialization(F_init_list,coeffs_init_list,error_reco_init_list,num_subdyns, ax = [], ax_eigen = [], ax_co = [], ax_co_heat = [], error_max_show = 15,init_point = 70,ax_reco =[],reco_order = 10, latent_dyn_init_list = [], annot = True,name_var = 'IC'):
    """
    Check, explore and visualize changes in initial conditions effects
    
    """
    num_subdyns = len(F_init_list[0])
    if isinstance(ax,list):
        if len(ax) == 0:
          fig, ax = plt.subplots(1,num_subdyns, figsize = (num_subdyns*8,int(0.4*len(F_init_list))))
    if isinstance(ax_co,list):
        if len(ax_co) == 0:
          fig_co, ax_co = plt.subplots(1,num_subdyns, figsize = (num_subdyns*5,int(0.3*len(F_init_list))))
    if isinstance(ax_co_heat,list):
        if len(ax_co_heat) == 0:
          fig_co_heat, ax_co_heat = plt.subplots(1,num_subdyns, figsize = (num_subdyns*8,int(0.3*len(F_init_list))))

    F_lasts = plot_f_different_initialization(F_init_list,annot = annot)
    if len(latent_dyn_init_list) > 0:
        if isinstance(ax_reco,list):
            if len(ax_reco) == 0:
                
              fig_reco, ax_reco = plt.subplots(2,len(F_lasts), figsize = (num_subdyns*8,10) , subplot_kw={'projection':'3d'})
    if isinstance(F_init_list[0][0],list):
        c_lasts = cal_f_lst(coeffs_init_list)
    else: 
        c_lasts = coeffs_init_list

    Fpair1 = F_lasts[0]
    F_lasts_orgs = [Fpair1]
    c1 = c_lasts[0]
    c_lasts_orgs = [c1]
    for idf in range(len(F_lasts)-1):        
        Fpair2 = F_lasts[idf+1]
        c2    = c_lasts[idf+1]
        _,Fpair1,c2org = match_corrs(Fpair1,Fpair2, c2) 
        F_lasts_orgs.append(Fpair1)
        c_lasts_orgs.append(c2org)
    all_stacked_subdyns = []    

    for sub_dyn_num in range(num_subdyns):
        stacked_subdyns = np.vstack([F_lasts_orgs[i][sub_dyn_num].flatten() for i in range(len(F_lasts_orgs))])
        sns.heatmap(np.abs(np.corrcoef(stacked_subdyns)),annot = True,ax = ax[sub_dyn_num],cmap = 'Greens')
        all_stacked_subdyns.append(stacked_subdyns)
    fig.suptitle('Subdynamics correlations for different %s'%name_var)
    [add_labels(ax = ax[i], title = 'f%g'%i, xlabel = '%s Iteration #'%name_var, ylabel = '%s Iteration #'%name_var, zlabel = None) for i in range(num_subdyns)]

    colors = np.random.rand(3,num_subdyns)
    [check_eigenspaces(F_lasts_orgs[init_num], colors = colors, ax = [], title2 = 'Eigenspaces (%s#%g)'%(name_var,init_num),title1= 'Eigenvalues (%s#%g)'%(name_var,init_num)) for init_num in range(len(F_init_list))];

    [[ax_co[sub_dyn_spec].plot(c_lasts_org[sub_dyn_spec][init_point:], alpha = 0.4) for sub_dyn_spec in range(num_subdyns)] for c_lasts_org in c_lasts_orgs]
    ax_co[-1].legend(['%s#%g'%(name_var,num_IC) for num_IC in range(len(c_lasts_orgs))])
    [add_labels(ax_spec, title = 'c#%g'%sub_dyn_num, xlabel = 'Time',ylabel = 'Coeffs',zlabel = None) for sub_dyn_num, ax_spec in enumerate(ax_co)]
    fig_co.suptitle('Coefficients for different %s'%name_var)
    

    [sns.heatmap(np.vstack([c_lasts_org[sub_dyn_spec][init_point:]for c_lasts_org  in c_lasts_orgs]) , ax = ax_co_heat[sub_dyn_spec], alpha = 0.4)  for sub_dyn_spec in range(num_subdyns) ]
    [add_labels(ax_spec, title = 'c#%g'%sub_dyn_num, xlabel = 'Time',ylabel = 'Different IC',zlabel = None) for sub_dyn_num, ax_spec in enumerate(ax_co_heat)]
    fig_co.suptitle('Coefficients for different %s, each subplot i describes the coefficients corresponding to sub-dynamic i'%name_var)

    fig, ax_bar = plt.subplots(figsize = (7,7))
    error_max_show = np.min([error_max_show, error_reco_init_list[0].shape[1]])
    last_error = np.vstack([error_reco_init_list[num_IC][-1,:error_max_show] for num_IC in range(len(F_init_list))])
    columns=['order %g'%order for order in range(error_max_show)]
    pd.DataFrame(last_error,columns = columns, index = ['%s #%g'%(name_var,iteration) for iteration in range(len(F_init_list))]).T.plot(ax = ax_bar)
    ax_bar.set_yscale('log')
    add_labels(ax_bar, xlabel = 'Reconstruction Order', ylabel = 'Error',zlabel = None, title = 'Error for different reconstruction orders, under different initial conditions' )
    #    
    if len(latent_dyn_init_list) > 0:
        
        [visualize_dyn(create_reco(latent_dyn_init_list[i], c_lasts[i],F_lasts[i], step_n = 1), ax = ax_reco[0,i], color_by_dominant = True, coefficients =c_lasts[i]) for i in range(len(F_lasts)) ]

        [visualize_dyn(create_reco(latent_dyn_init_list[i], c_lasts[i],F_lasts[i], step_n = reco_order), ax = ax_reco[1,i], color_by_dominant = True, coefficients =c_lasts[i]) for i in range(len(F_lasts)) ]


        [add_labels(ax = ax_reco[0,i], title = 'reco order 1, sample #%g'%i) for i in range(len(F_lasts)) ]
        #[visualize_dyn(create_reco(latent_dyn_init_list[i], c_lasts_orgs[i],F_lasts_orgs[i], step_n = reco_order), ax = ax_reco[1,i], color_by_dominant = False, coefficients =c_lasts_orgs[i]) for i in range(len(F_lasts_orgs)) ]
        [add_labels(ax = ax_reco[1,i], title = 'reco order %g, sample #%g'%(reco_order, i)) for i in range(len(F_lasts)) ]
        
    return F_lasts_orgs,c_lasts_orgs,all_stacked_subdyns
        
def create_colors(len_colors, perm = [0,1,2], style = 'random', cmap  = 'viridis'):
    """
    Create a set of discrete colors with a one-directional order
    Input: 
        len_colors = number of different colors needed
    Output:
        3 X len_colors matrix decpiting the colors in the cols
    """
    if style == 'random':
        colors = np.vstack([np.linspace(0,1,len_colors),(1-np.linspace(0,1,len_colors))**2,1-np.linspace(0,1,len_colors)])
        colors = colors[perm, :]
    else:
        
        # Define the colormap you want to use
        cmap = plt.get_cmap()  # Replace 'viridis' with the desired colormap name

        
        # Create an array of values ranging from 0 to 1 to represent positions in the colormap
        positions = np.linspace(0, 1, len_colors)
        
        # Generate the list of colors by applying the colormap to the positions
        colors = [cmap(pos) for pos in positions]
        
        # You can now use the 'colors' list as a list of colors in your application


    return colors
        
def plot_dict_array(dict_to_plot, cmap = 'PiYG', axs = [], key_to_plot = 'coefficients',type_plot = 'plot',min_time = 50,sharey= 'row',rows_plot = -10,logscale = False, zero_ref = 0,xlabel = 'Time',ylabel = 'coeffs'):
    """
    Plot dynamics with different regularization values
    type_plot: can be plot or heatmap
    """
    
    if isinstance(axs,list):
        if len(axs) == 0:
            fig, axs = plt.subplots(len(dict_to_plot.keys()), len(dict_to_plot[list(dict_to_plot.keys())[0]]), figsize = (15,len(dict_to_plot.keys())*4), sharey = sharey)  #, sharex  = True, sharey = True
            axs = axs.reshape(len(dict_to_plot.keys()), len(dict_to_plot[list(dict_to_plot.keys())[0]]))
    reg_ordered_to_plot = list(dict_to_plot.keys())
    num_dyns_values = dict_to_plot[reg_ordered_to_plot[0]]
    if type_plot == 'plot':
       
        if rows_plot <= -5:         [[axs[reg_val_num,num_dyns_count].plot(dict_to_plot[reg_val][num_dyns_val][key_to_plot][:,min_time:].T) for num_dyns_count, num_dyns_val in enumerate(num_dyns_values.keys())] for reg_val_num, reg_val in enumerate(reg_ordered_to_plot)]
        else:         
            [[axs[reg_val_num,num_dyns_count].plot(dict_to_plot[reg_val][num_dyns_val][key_to_plot][rows_plot, min_time:]) for num_dyns_count, num_dyns_val in enumerate(num_dyns_values.keys())] for reg_val_num, reg_val in enumerate(reg_ordered_to_plot)]
        if not np.isnan(zero_ref):
            [ax.axhline(zero_ref,color = 'r',ls = '--',alpha = 0.5) for ax in axs.flatten()]

            
    elif    type_plot == 'heat':
        if rows_plot <= -5: [[sns.heatmap(dict_to_plot[reg_val][num_dyns_val][key_to_plot][:,min_time:].T,ax = axs[reg_val_num,num_dyns_count],vmin = 0,vmax = 0.1,cmap = cmap) for num_dyns_count, num_dyns_val in enumerate(num_dyns_values.keys())] for reg_val_num, reg_val in enumerate(reg_ordered_to_plot)]
        else:  [[sns.heatmap(dict_to_plot[reg_val][num_dyns_val][key_to_plot][rows_plot,min_time:].reshape((1,-1)),ax = axs[reg_val_num,num_dyns_count],vmin =0,vmax = 0.1, cmap = cmap) for num_dyns_count, num_dyns_val in enumerate(num_dyns_values.keys())] for reg_val_num, reg_val in enumerate(reg_ordered_to_plot)]
       
    else:
        raise NameError('Unknown type plot!')
    [[add_labels(ax = axs[reg_val_num,num_dyns_count], title = 'reg =%g, for %g dynamics'%(reg_val, num_dyns_val), xlabel = xlabel,ylabel = ylabel,zlabel = None) for num_dyns_count, num_dyns_val in enumerate(num_dyns_values.keys())] for reg_val_num, reg_val in enumerate(reg_ordered_to_plot)]
    if logscale:
        [ax_spec.set_yscale('log') for ax_spec in axs.flatten()]
    fig.subplots_adjust(hspace = 0.7,wspace = 0.5)    
    

    
def plotfig(dict_to_plot,axs = [],cmap_base = 'viridis',name1 = ' reg',name2 = ' # sub-dyns',step_n = 1, accumulation = False,params_plot = {} , suptitle = ''):
    """
    Plot specific dynamics
    dict_to_plot: a dictionary with the dynamics to plot
    """
    if isinstance(axs,list):
        if len(axs) == 0:
            fig, axs = plt.subplots(len(dict_to_plot.keys()), len(dict_to_plot[list(dict_to_plot.keys())[0]]), figsize = (15,len(dict_to_plot.keys())*5) , subplot_kw={'projection':'3d'})  #, sharex  = True, sharey = True
            axs = axs.reshape(len(dict_to_plot.keys()), len(dict_to_plot[list(dict_to_plot.keys())[0]]))

    cmap = plt.cm.get_cmap(cmap_base, 3)
    [[visualize_spec_dyn(dict_to_plot[reg_val][sub_dyn_val],step_n = step_n, ax = axs[reg_num,sub_dyn_num],  accumulation = accumulation, cmap = cmap,params_plot = {**{'title':'%s%g%s%g'%(name1, reg_val, name2,sub_dyn_val)},**params_plot}) for sub_dyn_num, sub_dyn_val in enumerate(value_full_sub.keys())] for reg_num, (reg_val,value_full_sub) in enumerate(dict_to_plot.items())]
    if len(suptitle)>0: fig.suptitle(suptitle)

def visualize_spec_dyn(part_dic,ax, step_n = 1, accumulation = False,cmap = 'PiYG',return_fig  = True,params_plot = {}):
    """
    Visualize a set of reconstructed dynamics
    Inputs:
        part_dic = dictionary with keys  'latent_dyn', 'coefficients', 'F'
        ax        = subplot to plot into
        step_n   = order of the reconstruction
        accumulation = whether the reconstruction should be limited by order or a full reconstruction
        
    """
    visualize_dyn(create_reco(part_dic['latent_dyn'],part_dic['coefficients'],part_dic['F'],step_n = step_n,accumulation = accumulation)[:,:-1], turn_off_back=True, color_by_dominant = True, coefficients = part_dic['coefficients'], ax = ax,cmap = cmap, return_fig = return_fig, colorbar = False, params_plot = params_plot)    
    
    
def calcul_contribution(reco, real, direction = 'forward',dict_store ={}, func = np.nanmedian):
    """
    Calculate the error and the % close points for specific reconstruction matrix
    Inputs:
        reco: k X T reconstructed dynamics matrix
        real: k X T real dynamics matrix (ground truth)
        direction: can be forward or backward
        func: the function to apply on the relative error of each point
    Outputs:
        error: relative error
        percent_close: % of points which are within the range
    """
    error = relative_eror(reco,real, return_mean = True, func = func)
    if direction == 'forward': error = error #1-error
    elif direction == 'backward': error = error
    else: raise NameError('Unknown direction')
    percent_close = claculate_percent_close(reco, real)
    if direction == 'forward': percent_close = 1-percent_close#percent_close
    elif direction == 'backward': percent_close =  1-percent_close#percent_close #
    else: raise NameError('Unknown direction')

    return error,percent_close

def relative_eror(reco,real, return_mean = True, func = np.nanmean):
    """
    Calculate the relative reconstruction error
    Inputs:
        reco: k X T reconstructed dynamics matrix
        real: k X T real dynamics matrix (ground truth)
        return_mean: reaturn the average of the reconstruction error over time
        func: the function to apply on the relative error of each point
    Output:
        the relative error (or the mean relative error over time if return_mean)
    """
    error_point = np.sqrt(((reco - real)**2)/(real)**2)
    if return_mean:
        return func(error_point )
    return func(error_point,0)


def claculate_percent_close(reco, real, epsilon_close = 0.1, return_quantiles = False, quantiles = [0.05,0.95]):
    """
    Calculte the ratio of close (within a specific distance) points among all dynamics' points
    Inputs:
        reco: k X T reconstructed dynamics matrix
        real: k X T real dynamics matrix (ground truth)
        epsilon_close: Threshold for distance
        return_quantiles: whether to return confidence interval values
        quantiles: lower / higher limits for the quantiles
        
    reco: k X T
    real: k X T
    """
    close_enough = np.sqrt(np.sum((reco - real)**2,0)) < epsilon_close

    if return_quantiles:
        try:
            q1,q2 = stats.proportion.proportion_confint(np.sum(close_enough),len(close_enough),quantiles[0])
        except:
            q1 = np.mean(close_enough)
            q2 = np.mean(close_enough)
        return np.mean(close_enough), q1, q2
    return np.mean(close_enough)
    

def plot_bar_contri(contri, ax = [],suptitle = '', fig = [], colors = [], colors_sim = [], remove_back = True, legend_prop = {}):
    """
    Plot a bar-plot of the values in the dict contri
    Inputs:
        contri   = dictionary whose values are dataframes to plot
        ax       = np.arrays of subplots (its len is the same as the number of keys in contri) (optional)
        suptitle = overall title of all subplots (optional)
        fig      = figure to use (optional)
        
    """
    legend_prop = {**{'loc':'upper right', 'prop':{'size':18}}, **legend_prop}
    titles = list(contri.keys())
    if isinstance(colors, list) and len(colors) == 0: 
        colors = ['r','g','b','gray','orange','m','cyan']
    if isinstance(colors_sim, list) and len(colors_sim) == 0: 
        colors = ['brown','darkgreen','darkblue','dimgray','darkorange','crimson','C']
    colors_full = np.vstack([colors[:contri[titles[0]].shape[0]], colors_sim[:contri[titles[0]].shape[0]]]).T.flatten()
    if isinstance(ax,list):
        if len(ax) == 0:
            fig, ax = plt.subplots(1,len(contri.keys()), figsize = (15,4))

    [contri[title].T.plot.bar(ax = ax[i], alpha = 0.6, color = colors[:contri[title].shape[0]]) for i,title in enumerate(titles)]
    [ax_spec.grid(axis = 'y') for ax_spec in ax]
    [ax_spec.legend().set_visible(False) for i, ax_spec in enumerate(ax) if i<len(ax)-1]
    ax[-1].legend(**legend_prop)
    [add_labels(ax = ax[i],title = title, xlabel = 'Metrics', zlabel = None,ylabel = None) for i,title in enumerate(titles)]

    fig.suptitle(suptitle)
    fig.subplots_adjust(wspace = 0.4,hspace = 0.5)
    
def plot_dots_close(reco,real, range_close = [], conf_int = 0.05, ax = [], color = 'blue',label='', lw = 5):
    """
    For a given reconstructed dynamics ('reco')-> plot a graph of the ratio of dots that are located within a specific distance threshold from the ground truth, \n as a function of the distance threshold.
    + confidence interval (5%-95%)
    
    Inputs:
        reco        = reconstructed dynamics (the dynamics oobtained by the model). np.array of k X T
        real        = ground-truth dynamics. np.array of k X T.
        range_close = array of possible distances to consider
        conf_int    = confidence interval value (scalar < 0.5)
        ax          = subplot to use (optional)
        color       = color to use (optional)
        label       = curve label
    Output: 
        a np.array of the ratio of 'correct' points, as they are located within the threshold defined by range_close, for each array's index

    """
    if len(range_close) == 0: range_close = np.linspace(10**-8, 10,30)
    if isinstance(ax,list):
        if len(ax) == 0:
            fig, ax = plt.subplots(1,1, figsize = (4,4))    
    vals_close = [claculate_percent_close(reco, real, epsilon_close = close_val, return_quantiles = True, quantiles = [conf_int,1-conf_int]) for close_val in range_close]
    array_close = np.vstack(vals_close)
    ax.plot(range_close, array_close[:,0],color = color, label = label, alpha = 0.7, lw = lw)
    ax.fill_between(range_close, array_close[:,1],array_close [:,2], alpha = 0.2, color = color)
    ax.grid(axis = 'y')
    ax.set_xlabel('Distance')
    ax.set_ylabel('% points')
    ax.grid(axis = 'y')
    return vals_close    

def plot_coefficients_under_speeds(coefficients_mat, dt_range,ax = [] ,min_time = 10000, ax_heat = [], fig = [], fig_heat = [], colors = [], count_plot = 1):
    """"
    Plot the model coefficients obtained under different speeds, as a heatmap. 
    The goal is to compare the coefficients obtained for the same dnamics but with different sampling rates.
    Inputs:
        coefficients_mat   =
        dt_range           = 
        ax                 = subplot to plot the coefficients in
        min_time           = the min. time to plot (left x lim)
        ax_heat            = subplot to draw heatmap
        fig                =  figure to use
        fig_heat           = -||- (for heatmap(
        colors             = colors to use. Shoud be 3 X number_of_different_speeds
        count_plot         =  current # of plot (for labeling)
    """
    if isinstance(ax,list):
        if len(ax) == 0:
            fig,ax = plt.subplots(len(dt_range),1,figsize = (8,10), sharex = True)
    if isinstance(ax_heat,list):
        if len(ax_heat) == 0:
            fig_heat,ax_heat = plt.subplots(2,1,figsize = (5,10), sharex = True)
    if len(colors) == 0:
        len_colors =coefficients_mat.shape[0]
        colors = np.vstack([np.linspace(0,1,len_colors)  ,np.zeros((1,len_colors)), 1-np.linspace(0,1,len_colors)])
    [ax[i].scatter(range(len(coefficients_mat[i,min_time:].T)), coefficients_mat[i,min_time:].T,color = 'b',marker = '*', s=4) for i in range(len(dt_range))] 
    [ax[i].set_title('dt = %.2f'%cur_dt) for i, cur_dt in enumerate(dt_range)]
    ax[0].set_title('c%g \n dt = %.2f'%(count_plot,dt_range[0]))
    ax[-1].set_xlabel('Time')
    try:
        [ax_spec.set_ylim(top = np.nanquantile(coefficients_mat[i,min_time:],0.99)+100) for i,ax_spec in enumerate(ax)]
    except:
        print('Invalid y limit')
    ax_heat[0].plot(pd.DataFrame(coefficients_mat[:,min_time:].T).interpolate(axis = 0).values,lw = 1, alpha = 0.5)
    for i,j in enumerate(ax_heat[0].lines):
        j.set_color(colors[:,i])
   
    ax_heat[0].legend(['dt = %.2f'%cur_dt for i, cur_dt in enumerate(dt_range)], fontsize = 9, loc = 'upper right')
    sns.heatmap(pd.DataFrame(coefficients_mat[:,10000:]).interpolate(axis = 1), cmap = 'PiYG', ax = ax_heat[1])
    add_labels(ax_heat[1], xlabel = 'Time', ylabel = 'Speeds', zlabel = None, yticklabels = dt_range, title = 'c%g'%count_plot )
    add_labels(ax_heat[0], xlabel = 'Time', ylabel = 'Coeffs', zlabel = None, title = 'c%g'%count_plot )
    
    if ~isinstance(fig,list):        fig.subplots_adjust(hspace= 0.83, wspace= 0.4);    fig.suptitle('Coefficients')
    if ~isinstance(fig_heat,list):        fig_heat.subplots_adjust(hspace= 0.23, wspace= 0.4);    fig_heat.suptitle('Coefficients')   
    
    
#%% Working with files

    
    
    
def load_mat_file(mat_name , mat_path = '',sep = sep, squeeze_me = True,simplify_cells = True):
    """
    Function to load mat files. Useful for uploading the c. elegans data. 
    Example:
        load_mat_file('WT_Stim.mat','E:\CoDyS-Python-rep-\other_models')
    """
    try:
        if mat_path == '':
            data_dict = sio.loadmat(mat_name, squeeze_me = squeeze_me,simplify_cells = simplify_cells)
        else:
            data_dict = sio.loadmat(mat_path+sep+mat_name, squeeze_me = True,simplify_cells = simplify_cells)
    except: 
        try:
            data_dict = mat73.loadmat(mat_path+sep+mat_name)
        except:
            data_dict = scipy.io.loadmat(mat_path+sep+mat_name)
    return data_dict    
    
    
    
def plot_most_likely_dynamics(reg, dynamics_distns,xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=10,
        alpha=0.8,     ax=None, figsize=(3, 3)):
    K = len(dynamics_distns)
    D_latent = dynamics_distns[0].D_out
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    Ts = reg.get_trans_matrices(xy)
    prs = Ts[:, 0, :]
    z = np.argmax(prs, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        A = dynamics_distns[k].A[:, :D_latent]
        b = dynamics_distns[k].A[:, D_latent:]
        dydt_m = xy.dot(A.T) + b.T - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dydt_m[zk, 0], dydt_m[zk, 1],
                      color=colors[k], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax
    
    
#%% Plot Multi-colored line
def try_norm_coeffs(coefficients,x_highs_y_highs = [], x_lows_y_lows = [] , choose_meth = 'both',
                            same_width = True,factor_power = 0.9, width_des = 0.7, initial_point = 'start', latent_dyn = [], quarter_initial = 'low'):
    if len(latent_dyn) == 0: raise ValueError('Empty latent dyn was provided')
    coefficients_n = norm_over_time(coefficients, type_norm = 'normal')
    coefficients_n = coefficients_n - np.min(coefficients_n,1).reshape((-1,1))
    if same_width:
        coefficients_n = width_des*(coefficients_n**factor_power) / np.sum(coefficients_n**factor_power,axis = 0)   
    else:
        coefficients_n = coefficients_n / np.sum(coefficients_n,axis = 0)  
    return coefficients_n





# def plot_weighted_colored_line(dyn, coeffs, ax = [], fig=None ):
#     coefficients = norm_over_time(coefficients, type_norm = 'normal')
#     if isinstance(ax,list) and len(ax) == 0:
#         fig, ax = plt.subplots()
 
            
    
def min_dist(dotA1, dotA2, dotB1, dotB2, num_sects = 500):
    x_lin = np.linspace(dotA1[0], dotA2[0])
    y_lin = np.linspace(dotA1[1], dotA2[1])
    x_lin_or = np.linspace(dotB1[0], dotB2[0])
    y_lin_or = np.linspace(dotB1[1], dotB2[1])
    dist_list = []
    for pairA_num, pairAx in enumerate(x_lin):
        pairAy = y_lin[pairA_num]
        for pairB_num, pairBx in enumerate(x_lin_or):
            pairBy = y_lin_or[pairB_num]
            dist = (pairAx - pairBx)**2 + (pairAy - pairBy)**2
            dist_list.append(dist)
    return dist_list
            
    
    
#%% FHN model
# taken from https://www.normalesup.org/~doulcier/teaching/modeling/excitable_systems.html    
    
def create_FHN(dt = 0.01, max_t = 100, I_ext = 0.5,
               b = 0.7, a = 0.8 , tau = 20, v0 = -0.5, w0 = 0, params = {'exp_power' : 0.9, 'change_speed': False}):
    time_points = np.arange(0, max_t, dt)
    if params['change_speed']:
        time_points = time_points**params['exp_power']
    
        
    w_full = []
    v_full = []
    v = v0
    w = w0
    for t in time_points:
        v, w =  cal_next_FHN(v,w, dt , max_t , I_ext , b, a , tau)
        v_full.append(v)
        w_full.append(w)
    return v_full, w_full


        
def cal_next_FHN(v,w, dt = 0.01, max_t = 300, I_ext = 0.5, 
                 b = 0.7, a = 0.8 , tau = 20) :
    v_next = v + dt*(v - (v**3)/3 - w + I_ext)
    w_next = w + dt/tau*(v + a - b*w)
    return v_next, w_next
    
#%% Plot tricolor
def norm_over_time(coefficients, type_norm = 'normal'):
    if type_norm == 'normal':
        coefficients_norm = (coefficients - np.mean(coefficients,1).reshape((-1,1)))/np.std(coefficients, 1).reshape((-1,1))
    return coefficients_norm

def find_perpendicular(d1, d2, perp_length = 1, prev_v = [], next_v = [], ref_point = [],choose_meth = 'intersection',initial_point = 'mid',  
                       direction_initial = 'low', return_unchose = False, layer_num = 0):
    """
    This function find the 2 point of the orthogonal vector to a vector defined by points d1,d2
    d1 =                first data point
    d2 =                second data point
    perp_length =       desired width
    prev_v =            previous value of v. Needed only if choose_meth == 'prev'
    next_v =            next value of v. Needed only if choose_meth == 'prev'
    ref_point =         reference point for the 'smooth' case, or for 2nd+ layers
    choose_meth =       'intersection' (eliminate intersections) OR 'smooth' (smoothing with previous prediction) OR 'prev' (eliminate convexity)
    direction_initial = to which direction take the first perp point  
    return_unchose =    whether to return unchosen directions   
    
    """       
    # Check Input    
    if d2[0] == d1[0] and d2[1] == d1[1]:
        raise ValueError('d1 and d2 are the same point')
    
    # Define start point for un-perp curve
    if initial_point == 'mid':
        perp_begin = (np.array(d1) + np.array(d2))/2
        d1_perp = perp_begin
    elif initial_point == 'end':        d1_perp = d2
    elif initial_point == 'start':        d1_perp = d1
    else:        raise NameError('Unknown intial point')       
    
    # If perpendicular direction is according to 'intersection' elimination
    if choose_meth == 'intersection':
        if len(prev_v) > 0:        intersected_curve1 = prev_v
        else:                      intersected_curve1 = d1
        if len(next_v) > 0:        intersected_curve2 = next_v
        else:                      intersected_curve2 = d2
        
    # If a horizontal line       
    if d2[0] == d1[0]:        d2_perp = np.array([d1_perp[0]+perp_length, d1_perp[1]])
    # If a vertical line
    elif d2[1] == d1[1]:        d2_perp = np.array([d1_perp[0], d1_perp[1]+perp_length])
    else:
        m = (d2[1]-d1[1])/(d2[0]-d1[0]) 
        m_per = -1/m                                       # Slope of perp curve        
        theta1 = np.arctan(m_per)
        theta2 = theta1 + np.pi
        
        # if smoothing
        if choose_meth == 'smooth' or choose_meth == 'intersection':
            if len(ref_point) == 0: 
                #print('no ref point!')
                smooth_val =[]
            else:                smooth_val = np.array(ref_point)
        
        # if by convexity
        if choose_meth == 'prev':
            if len(prev_v) > 0 and len(next_v) > 0:                     # both sides are provided
                prev_mid_or = (np.array(prev_v) + np.array(next_v))/2
            elif len(prev_v) > 0 and len(next_v) == 0:                  # only the previous side is provided
                prev_mid_or = (np.array(prev_v) + np.array(d2))/2
            elif len(next_v) > 0 and len(prev_v) == 0:                  # only the next side is provided               
                prev_mid_or = (np.array(d1) + np.array(next_v))/2
            else:
                raise ValueError('prev or next should be defined (to detect convexity)!')        

        if choose_meth == 'prev':
            prev_mid = prev_mid_or
        elif choose_meth == 'smooth':
            prev_mid = smooth_val
        elif choose_meth == 'intersection':
            prev_mid = smooth_val
            
        x_shift = perp_length * np.cos(theta1)
        y_shift = perp_length * np.sin(theta1)
        d2_perp1 = np.array([d1_perp[0] + x_shift, d1_perp[1]+ y_shift])            
        
        x_shift2 = perp_length * np.cos(theta2)
        y_shift2 = perp_length * np.sin(theta2)
        d2_perp2 = np.array([d1_perp[0] + x_shift2, d1_perp[1]+ y_shift2])
        options_last = [d2_perp1, d2_perp2]
        
        # Choose the option that goes outside
        if len(prev_mid) > 0:
            
          
            if len(ref_point) > 0 and layer_num > 0:                               # here ref point is a point of a different dynamics layer from which we want to take distance
                dist1 = np.sum((smooth_val - d2_perp1)**2)
                dist2 = np.sum((smooth_val - d2_perp2)**2)
                max_opt = np.argmax([dist1, dist2])

            elif choose_meth == 'intersection':
                dist1 = np.min(min_dist(prev_mid, d2_perp1, intersected_curve1, intersected_curve2))
                dist2 = np.min(min_dist(prev_mid, d2_perp2, intersected_curve1, intersected_curve2))
                max_opt = np.argmax([dist1,dist2]) 
         
            else:
                dist1 = np.sum((prev_mid - d2_perp1)**2)
                dist2 = np.sum((prev_mid - d2_perp2)**2)
                max_opt = np.argmin([dist1,dist2])  
                     
     
                                 
       
                          
                
        else:
        
            if len(ref_point) > 0 and layer_num >0:                               # here ref point is a point of a different dynamics layer from which we want to take distance
                dist1 = np.sum((ref_point - d2_perp1)**2)
                dist2 = np.sum((ref_point - d2_perp2)**2)
                max_opt = np.argmax([dist1, dist2])
             
            elif direction_initial == 'low':
                max_opt = np.argmin([d2_perp1[1], d2_perp2[1]])
            elif direction_initial == 'high':
                max_opt = np.argmax([d2_perp1[1], d2_perp2[1]])
            elif direction_initial == 'right' :
                max_opt = np.argmax([d2_perp1[0], d2_perp2[0]])
            elif direction_initial == 'left':
                max_opt = np.argmin([d2_perp1[0], d2_perp2[0]])

                
            else:
                raise NameError('Invalid direction initial value') 
    
    d2_perp = options_last[max_opt] # take the desired direction
    if return_unchose:
        d2_perp_unchose = options_last[np.abs(1 - max_opt)] 
        return d1_perp, d2_perp, d2_perp_unchose
    return d1_perp, d2_perp


def find_lows_high(coeff_row, latent_dyn,   choose_meth ='intersection',factor_power = 0.9, initial_point = 'start',
                   direction_initial = 'low', return_unchose = False, ref_point = [], layer_num = 0):
    """
    Calculates the coordinates of the 'high' values of a specific kayer
    """
    
    if return_unchose: unchosen_highs = []
    ### Initialize
    x_highs_y_highs = []; x_lows_y_lows = []
    if isinstance(ref_point, np.ndarray):
        if len(ref_point.shape) > 1:
            ref_shape_all = ref_point
        else:
            ref_shape_all = np.array([])

    else:
        ref_shape_all = np.array([])
    # Iterate over time
    for t_num in range(0,latent_dyn.shape[1]-2): 
  
        d1_coeff = latent_dyn[:,t_num]
        d2_coeff = latent_dyn[:,t_num+1]
        prev_v = latent_dyn[:,t_num-1] 
        next_v = latent_dyn[:,t_num+2]
        c_len = (coeff_row[t_num] + coeff_row[t_num+1])/2

        if len(ref_shape_all) > 0 and ref_shape_all.shape[0] > t_num and layer_num > 0: # and ref_shape_all.shape[1] >1
            ref_point = ref_shape_all[t_num,:]

          
            if len(ref_point) >  0 and layer_num > 0 :  #and t_num  < 3
                 pass
          
        
        # if do not consider layer
        
        elif t_num > 2 and (choose_meth == 'smooth' or choose_meth == 'intersection'):   
            ref_point  = d2_perp

          
        else:              
            ref_point = []       

        
        if return_unchose:  d1_perp, d2_perp, d2_perp_unchosen = find_perpendicular(d1_coeff, d2_coeff,c_len**factor_power, prev_v = prev_v, next_v=next_v,ref_point  = ref_point , choose_meth = choose_meth, initial_point=initial_point, direction_initial =direction_initial, return_unchose = return_unchose,layer_num=layer_num)# c_len
        else:               d1_perp, d2_perp = find_perpendicular(d1_coeff, d2_coeff,c_len**factor_power, prev_v = prev_v, next_v=next_v,ref_point  = ref_point , choose_meth = choose_meth, initial_point=initial_point, direction_initial= direction_initial, return_unchose = return_unchose,layer_num=layer_num)# c_len
        # Add results to results lists
        x_lows_y_lows.append([d1_perp[0],d1_perp[1]])
        x_highs_y_highs.append([d2_perp[0],d2_perp[1]])
        if return_unchose: unchosen_highs.append([d2_perp_unchosen[0],d2_perp_unchosen[1]])
    # return
    if return_unchose: 
        return x_lows_y_lows, x_highs_y_highs, unchosen_highs
    return x_lows_y_lows, x_highs_y_highs        


def plot_multi_colors(store_dict,min_time_plot = 0,max_time_plot = -100,  colors = ['green','red','blue'], ax = [],
                      fig = [], alpha = 0.99, smooth_window = 3, factor_power = 0.9, coefficients_n = [], to_scatter = False, 
                      to_scatter_only_one = False ,choose_meth = 'intersection', title = ''):
    """
    store_dict is a dictionary with the high estimation results. 
    example:        
        store_dict , coefficients_n = calculate_high_for_all(coefficients,choose_meth = 'intersection',width_des = width_des, latent_dyn = latent_dyn, direction_initial = direction_initial,factor_power = factor_power, return_unchose=True)
    
    """
    if len(colors) < len(store_dict):                raise ValueError('Not enough colors were provided')
    if isinstance(ax, list) and len(ax) == 0:        fig, ax = plt.subplots(figsize = (20,20))
    for key_counter, (key,set_plot) in enumerate(store_dict.items()):
        if key_counter == 0:
            x_lows_y_lows = store_dict[key][0]
            x_highs_y_highs = store_dict[key][1]
            #choose_meth_initial = 
            low_ref =[]
            high_ref = []
        else:
            low_ref = [np.array(x_highs_y_highs)[min_time_plot,0],   np.array(x_highs_y_highs)[min_time_plot,1]]
            high_ref = [np.array(x_highs_y_highs)[max_time_plot,0],np.array(x_highs_y_highs)[max_time_plot,1]]
        if len(coefficients_n) > 0:
            # Define the length of the last perp. 
            c_len = (coefficients_n[key,max_time_plot-1] + coefficients_n[key,max_time_plot])/2
            # Create perp. in the last point            
            d1_p, d2_p =find_perpendicular([np.array(x_lows_y_lows)[max_time_plot-2,0],np.array(x_lows_y_lows)[max_time_plot-2,1]], 
                                           [np.array(x_lows_y_lows)[max_time_plot-1,0],np.array(x_lows_y_lows)[max_time_plot-1,1]], 
                                           perp_length = c_len**factor_power, 
                                           ref_point = high_ref, 
                                           choose_meth = 'intersection',initial_point = 'end')
            # Define the length of the first perp. 
            c_len_start = (coefficients_n[key,min_time_plot-1] + coefficients_n[key,min_time_plot])/2
            # Create perp. in the first point   
            d1_p_start =[np.array(x_highs_y_highs)[min_time_plot,0],np.array(x_highs_y_highs)[min_time_plot,1]]
                                                       
            d2_p_start=  [np.array(x_highs_y_highs)[min_time_plot+1,0],np.array(x_highs_y_highs)[min_time_plot+1,1]]                                                        

            x_lows_y_lows = store_dict[key][0]
            x_highs_y_highs = store_dict[key][1] 

            stack_x = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,0].flatten(), np.array([d2_p[0]]), np.array(x_highs_y_highs)[max_time_plot-1:min_time_plot+1:-1,0].flatten(),np.array([d2_p_start[0]])])
            stack_y = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,1].flatten(), np.array([d2_p[1]]),np.array(x_highs_y_highs)[max_time_plot-1:min_time_plot+1:-1,1].flatten(),np.array([d2_p_start[1]])])
            
        else:
            stack_x = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,0].flatten(), np.array(x_highs_y_highs)[max_time_plot:min_time_plot:,0].flatten()])
            stack_y = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,1].flatten(), np.array(x_highs_y_highs)[max_time_plot:min_time_plot:,1].flatten()])
        stack_x_smooth = stack_x
        stack_y_smooth = stack_y
        if key_counter !=0:
            ax.fill(stack_x_smooth, stack_y_smooth, alpha=0.3, facecolor=colors[key_counter], edgecolor=None, zorder=1, snap = True)#
        else:
            ax.fill(stack_x_smooth, stack_y_smooth, alpha=alpha, facecolor=colors[key_counter], edgecolor=None, zorder=1, snap = True)#

    if to_scatter or (to_scatter_only_one and key == np.max(list(store_dict.keys()))):
        

          ax.scatter(np.array(x_lows_y_lows)[min_time_plot:max_time_plot,0].flatten(), np.array(x_lows_y_lows)[min_time_plot:max_time_plot,1].flatten(), c = 'black', alpha = alpha, s = 45)

    remove_edges(ax)
    if not title  == '':
        ax.set_title(title, fontsize = 20)
    


def remove_edges(ax, include_ticks = True, top = False, right = False, bottom = True, left = True):
    ax.spines['top'].set_visible(top)    
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)  
    if not include_ticks:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

def norm_coeffs(coefficients, type_norm, same_width = True,width_des = 0.7,factor_power = 0.9, min_width = 0.01):
    """
    type_norm can be:      'sum_abs', 'norm','abs'
    """
    if type_norm == 'norm':
        coefficients_n =      norm_over_time(np.abs(coefficients), type_norm = 'normal')   
        coefficients_n =      coefficients_n - np.min(coefficients_n,1).reshape((-1,1))
        #plt.plot(coefficients_n.T)
    elif type_norm == 'sum_abs':
        coefficients[np.abs(coefficients) < min_width] = min_width
        coefficients_n = np.abs(coefficients) / np.sum(np.abs(coefficients),1).reshape((-1,1))
    elif type_norm == 'abs':
        coefficients[np.abs(coefficients) < min_width] = min_width
        coefficients_n = np.abs(coefficients) #/ np.sum(np.abs(coefficients),1).reshape((-1,1))
    elif type_norm == 'no_norm':
        coefficients_n = coefficients
    else:
        raise NameError('Invalid type_norm value')


    coefficients_n[coefficients_n < min_width]  = min_width
    if same_width:        coefficients_n = width_des*(np.abs(coefficients_n)**factor_power) / np.sum(np.abs(coefficients_n)**factor_power,axis = 0)   
    else:                 coefficients_n = np.abs(coefficients_n) / np.sum(np.abs(coefficients_n),axis = 0)  
    coefficients_n[coefficients_n < min_width]  = min_width
    return coefficients_n

    
def calculate_high_for_all(coefficients, choose_meth = 'both', same_width = True,factor_power = 0.9, width_des = 0.7, 
                           initial_point = 'start', latent_dyn = [],
                          direction_initial = 'low', return_unchose = False, type_norm = 'norm',min_width =0.01):
    """
    Create the dictionary to store results
    """
    if len(latent_dyn) == 0: raise ValueError('Empty latent dyn was provided')
    
    # Coeffs normalization
    coefficients_n = norm_coeffs(coefficients, type_norm, same_width = same_width, width_des = width_des,factor_power =factor_power,min_width=min_width )
    
    # Initialization
    store_dict      = {}
    dyn_use         = latent_dyn
    ref_point       = []
    
    for row in range(coefficients_n.shape[0]):
        #print(row)
        coeff_row = coefficients_n[row,:]
        # Store the results for each layer
        if return_unchose:
            x_lows_y_lows, x_highs_y_highs,x_highs_y_highs2 = find_lows_high(coeff_row,dyn_use, choose_meth = choose_meth, factor_power=factor_power, 
                                                                             initial_point = initial_point,direction_initial = direction_initial,
                                                                             return_unchose = return_unchose, ref_point = ref_point,layer_num = row )             
            store_dict[row] = [x_lows_y_lows, x_highs_y_highs,x_highs_y_highs2]
        else:
            x_lows_y_lows, x_highs_y_highs = find_lows_high(coeff_row,dyn_use, choose_meth = choose_meth, factor_power=factor_power, 
                                                            initial_point = initial_point, direction_initial = direction_initial ,
                                                            return_unchose = return_unchose,ref_point = ref_point ,layer_num=row)             
            store_dict[row] = [x_lows_y_lows, x_highs_y_highs]
        # Update the reference points    
        if initial_point == 'mid':
            dyn_use = np.array(x_highs_y_highs).T
            dyn_use = (dyn_use[:,1:] + dyn_use[:,:-1])/2
            dyn_use = np.hstack([latent_dyn[:,:2], dyn_use, latent_dyn[:,-2:]])
        else:
            dyn_use = np.array(x_highs_y_highs).T
        #if row > 0:
        ref_point = np.array(x_lows_y_lows)#[0,:]

    return store_dict, coefficients_n    

def add_bar_dynamics(coefficients_n, ax_all_all = [],min_max_points = [10,100,200,300,400,500], 
                     colors = np.array(['r','g','b','yellow']), centralize = False):
    if isinstance(ax_all_all, list) and len(ax_all_all) == 0:
        fig, ax_all_all  = plt.subplots(1,len(min_max_points), figsize = (8*len(min_max_points), 7))

    max_bar = np.max(np.abs(coefficients_n[:,min_max_points]))
    for pair_num,val in enumerate(min_max_points):
        ax_all = ax_all_all[pair_num]

        
        ax_all.bar(np.arange(coefficients_n.shape[0]),coefficients_n[:,val], 
                   color = np.array(colors)[:coefficients_n.shape[0]],
                   alpha = 0.3)

        ax_all.get_xaxis().set_ticks([]) #for ax in ax_all]
        ax_all.get_yaxis().set_ticks([]) #for ax in ax_all]
        ax_all.spines['top'].set_visible(False)
        
        ax_all.spines['right'].set_visible(False)
        ax_all.spines['bottom'].set_visible(False)
        ax_all.spines['left'].set_visible(False)  
        ax_all.axhline(0, ls = '-',alpha = 0.5, color = 'black', lw = 6)
        ax_all.set_ylim([-max_bar,max_bar])
def create_name_fhn_files(reg_value, num_dynamics):
    new_name = r'E:\CoDyS-Python-rep-\fhn\multifhn_%ssub%sreg.npy'%(str(num_dynamics), str(reg_value).replace('.','_'))
    return new_name
    
#%% Plot 2d axis of coeffs for fig 2

def plot_3d_dyn_basis(F, coefficients, projection = [0,-1], ax = [],  fig = [], time_emph = [], n_times = 5,
                      type_plot = 'quiver',range_p = 10,s=200, w = 0.05/3, alpha0 = 0.3, 
                      time_emph_text  = [10, 20, 30, 50,80, 100,200,300,400,500], turn_off_back = True, lim1 = np.nan, 
                      ax_qui = [], 
                      ax_base = [], to_title = True, loc_title = 'title', include_bar =True, axs_basis_colored = [],
                      colors_dyns = np.array(['r','g','b','yellow']) , plot_dyn_by_colorbase = False, remove_edges_ax = False, include_dynamics = False,
                      latent_dyn = [],fontsize_times = 16,delta_text = 0.1, delta_text_y = 0,delta_text_z = 0, 
                      new_colors = True, include_quiver = True, base_narrow = True,colors = [],color_by_dom = False,
                      quiver_3d = False, s_all = 10,to_remove_edge = True, to_grid = False, cons_color = False):   
    """
    ax = subplot to plot coefficients over time
    colors = should be a mat of k X 3
    """
    if not F[0].shape[0] ==3: quiver_3d = False
    if len(colors) ==0:    
        if color_by_dom:
            color_sig_tmp = find_dominant_dyn(np.abs(coefficients))
            colors = colors_dyns[color_sig_tmp]
            colors_base = np.zeros(coefficients.shape[1])
            
        else:
            colors_base = np.linspace(0,1,coefficients.shape[1]).reshape((-1,1))    
            colors = np.hstack([colors_base, 1-colors_base, colors_base**2])    

    if isinstance(ax,list) and len(ax) == 0:
        if len(F) == 3:        fig, ax = plt.subplots(subplot_kw={'projection':'3d'}, figsize= (10,10))
        elif len(F) == 2:      fig, ax = plt.subplots(figsize= (10,10))
        else: raise ValueError('Invalid dim for F')
    if len(time_emph) == 0: 
        time_emph =np.linspace(0,coefficients.shape[1]-2, n_times+1)[1:].astype(int)

    if include_dynamics:
        
        if len(latent_dyn) == 0: raise ValueError('You should provide latent dyn as input if "include dynamics" it True')
        if len(F[0]) == 3:
            fig_dyn,ax_dyn = plt.subplots(figsize = (15,15),subplot_kw={'projection':'3d'})
            if new_colors:

                ax_dyn.scatter(latent_dyn[0,:len(colors_base)], latent_dyn[1,:len(colors_base)],latent_dyn[2,:len(colors_base)], color = colors,alpha = 0.3)
                ax_dyn.scatter(latent_dyn[0,time_emph],latent_dyn[1,time_emph],latent_dyn[2,time_emph], c = 'black', s = 300)
            else:
                c_sig = np.arange(latent_dyn.shape[1])
                ax_dyn.scatter(latent_dyn[0,:], latent_dyn[1,:],latent_dyn[2,:], c = c_sig,alpha = 0.3, cmap = 'viridis', s = 100)
                ax_dyn.scatter(latent_dyn[0,time_emph],latent_dyn[1,time_emph],latent_dyn[2,time_emph], c = c_sig[time_emph], s = 300, cmap = 'viridis')
            [ax_dyn.text(latent_dyn[0,t] + delta_text,latent_dyn[1,t]+delta_text_y,latent_dyn[2,t]+delta_text_z, 't = %s'%str(t), fontsize =fontsize_times, fontweight = 'bold') for t in time_emph]
            ax_dyn.set_axis_off()
            
        else:
            fig_dyn,ax_dyn = plt.subplots(figsize = (10,10))
            if new_colors:
                
                
                ax_dyn.scatter(latent_dyn[0,:len(colors_base)], latent_dyn[1,:len(colors_base)], color = colors,alpha = 0.3, s = 50)
                ax_dyn.scatter(latent_dyn[0,time_emph],latent_dyn[1,time_emph], c = 'black', s = 200)
            else:
                c_sig = np.arange(latent_dyn.shape[1])
                ax_dyn.scatter(latent_dyn[0,:], latent_dyn[1,:], c = c_sig,alpha = 0.3, cmap = 'viridis', s = 100)
                ax_dyn.scatter(latent_dyn[0,time_emph],latent_dyn[1,time_emph], c = c_sig[time_emph], s = 300, cmap = 'viridis')
            [ax_dyn.text(latent_dyn[0,t] + delta_text,latent_dyn[1,t]+delta_text_y, 't = %s'%str(t), fontsize =fontsize_times, fontweight = 'bold') for t in time_emph]
            remove_edges(ax_dyn)
            #ax_dyn.set_axis_off()
    if len(F[0]) == 3: 

        if quiver_3d:
            if type_plot == 'streamplot': 
                type_plot = 'quiver'
                print('If quiver_3d then type_plot need to be quiver (currently is streamplot)')
            if  include_quiver:
                if isinstance(ax_qui, list) and len(ax_qui)== 0:         
                    fig_qui, ax_qui = plt.subplots(1,len(time_emph), figsize= (7*len(time_emph),5) ,subplot_kw={'projection':'3d'})
            if isinstance(ax_base, list) and len(ax_base)==0:         
                if base_narrow:
                    fig_base, ax_base = plt.subplots(len(F),1, figsize= (5,7*len(F)) ,subplot_kw={'projection':'3d'})
                else:
                    fig_base, ax_base = plt.subplots(1,len(F),figsize= (7*len(F),5 ),subplot_kw={'projection':'3d'})
        else:
          
            F = [f[:,projection] for f in F]
            F = [f[projection, :] for f in F]
            if  include_quiver:
                if isinstance(ax_qui, list) and len(ax_qui)== 0:         fig_qui, ax_qui = plt.subplots(1,len(time_emph), figsize= (7*len(time_emph),5) )
            if isinstance(ax_base, list) and len(ax_base)==0:         
                if base_narrow:
                    fig_base, ax_base = plt.subplots(len(F),1, figsize= (5,7*len(F)) )
                else:
                    fig_base, ax_base = plt.subplots(1,len(F),figsize= (7*len(F),5 ))

    elif len(F[0]) == 2:  
        if isinstance(ax_qui, list) and len(ax_qui)==0:     fig_qui, ax_qui = plt.subplots(1,len(time_emph), figsize= (7*len(time_emph),5))
        if isinstance(ax_base, list) and len(ax_base)==0:   
            if base_narrow:
                fig_base, ax_base = plt.subplots(len(F), 1,figsize= (5,7*len(F) ))
            else:
                fig_base, ax_base = plt.subplots(1,len(F),figsize= (7*len(F),5 ))
    if len(F[0]) == 3:

        cmap = matplotlib.cm.get_cmap('viridis')
        if new_colors:

            ax.scatter(coefficients[0,:],coefficients[1,:],coefficients[2,:], c = colors, alpha = alpha0, s = s_all)

            ax.scatter(coefficients[0,time_emph],coefficients[1,time_emph],coefficients[2,time_emph], c = 'black',
                       s = s)

            [plot_reco_dyn(coefficients, F, time_point, type_plot = type_plot, range_p = range_p, color =colors[time_point] ,
                       w = w, ax = ax_qui[i], quiver_3d = quiver_3d, cons_color = cons_color) for i, time_point in enumerate(time_emph)]
        else:
            
            cmap = matplotlib.cm.get_cmap('viridis')
            colors_base = np.arange(coefficients.shape[1])

            ax.scatter(coefficients[0,:],coefficients[1,:],coefficients[2,:], c = colors_base, alpha = alpha0, s = s_all)

            ax.scatter(coefficients[0,time_emph],coefficients[1,time_emph],coefficients[2,time_emph], c = 'black', s = s, alpha = np.min([alpha0*2, 1]))
            if include_quiver:
                [plot_reco_dyn(coefficients, F, time_point, type_plot = type_plot, range_p = range_p,  
                           color = cmap(time_point/colors.shape[0])  ,
                       w = w, ax = ax_qui[i], quiver_3d = quiver_3d, cons_color = cons_color) for i, time_point in enumerate(time_emph)]
            
        if to_title and include_quiver:
            if loc_title == 'title':
                [ax_qui[i].set_title('t = ' + str(time_point), fontsize =fontsize_times*3 , fontweight = 'bold') for i, time_point in enumerate(time_emph)]
            else:
                [ax_qui[i].set_ylabel('t = ' + str(time_point), fontsize =fontsize_times, fontweight = 'bold') for i, time_point in enumerate(time_emph)]

        [ax.text(coefficients[0,time_point]+delta_text,coefficients[1,time_point]+delta_text_y,coefficients[2,time_point]+delta_text_z,'t = ' + str(time_point), fontsize =fontsize_times, fontweight = 'bold') for time_point in time_emph_text]
        ax.set_xlabel('f1');ax.set_ylabel('f2');ax.set_zlabel('f3');

    else:
        


        cmap = matplotlib.cm.get_cmap('viridis')
        if new_colors:
            ax.scatter(coefficients[0,:],coefficients[1,:], c = colors, alpha = alpha0, s = s_all)
            ax.scatter(coefficients[0,time_emph],coefficients[1,time_emph], c = colors[time_emph], s = s)
            if include_quiver:
                [plot_reco_dyn(coefficients, F, time_point, type_plot = type_plot, range_p = range_p, color =colors[time_point] , w = w, ax = ax_qui[i],cons_color=cons_color ) for i, time_point in enumerate(time_emph)]
        else:
            colors_base = np.arange(coefficients.shape[1])
            ax.scatter(coefficients[0,:],coefficients[1,:], c = colors_base, alpha = alpha0, s = s_all)
            ax.scatter(coefficients[0,time_emph],coefficients[1,time_emph], c = colors_base[time_emph], s = s)
            if include_quiver:
                [plot_reco_dyn(coefficients, F, time_point, type_plot = type_plot, range_p = range_p, color = cmap(time_point/colors.shape[0]) , w = w, ax = ax_qui[i],cons_color= cons_color ) for i, time_point in enumerate(time_emph)]
        if latent_dyn.shape[0] == 3:
            [ax.text(coefficients[0,time_point]+delta_text,coefficients[1,time_point]+delta_text_y, coefficients[2,time_point]+delta_text_z,'t = ' + str(time_point),fontsize = fontsize_times ,fontweight = 'bold') for time_point in time_emph_text]
    
        else:
            [ax.text(coefficients[0,time_point]+delta_text,coefficients[1,time_point]+delta_text_y,'t = ' + str(time_point),fontsize = fontsize_times ,fontweight = 'bold') for time_point in time_emph_text]
        
        ax.set_xlabel('f1');ax.set_ylabel('f2');
        if  remove_edges_ax:        remove_edges(ax)

        if to_title and  include_quiver:
            if loc_title == 'title':
                [ax_qui[i].set_title('t = ' + str(time_point), fontsize = 30) for i, time_point in enumerate(time_emph)]
            else:
                [ax_qui[i].set_ylabel('t = ' + str(time_point), fontsize = 30) for i, time_point in enumerate(time_emph)]
    if to_remove_edge:  
        if  include_quiver:        [remove_edges(ax_spec) for ax_spec in ax_qui]
        [remove_edges(ax_spec) for ax_spec in ax_base]
        ax.set_xticks([])
        ax.set_yticks([])
        if quiver_3d:
            ax.set_zticks([])    
    #if include_quiver:
    [quiver_plot(f,-range_p, range_p, -range_p, range_p, ax = ax_base[f_num],chosen_color =  'black', w = w, type_plot = type_plot,cons_color =cons_color,quiver_3d = quiver_3d ) for f_num, f in enumerate(F)]
    [ax_base_spec.set_title('f %s'%str(i), fontsize = 16) for i, ax_base_spec in enumerate(ax_base)]
    

    if turn_off_back and  len(F) == 3:
      ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
      ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
      ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    if not to_grid and  len(F) == 3:
       
      ax.grid(False)
      # Hide axes ticks

      ax.set_zticks([])
      
      ax.xaxis._axinfo['juggled'] = (0,0,0)
      ax.yaxis._axinfo['juggled'] = (1,1,1)
      ax.zaxis._axinfo['juggled'] = (2,2,2)
    if not np.isnan(lim1):
        ax.set_xlim([-lim1,lim1])
        ax.set_ylim([-lim1,lim1])
        
        
    if include_bar:
        if base_narrow:
            fig_all_all, ax_all_all = plt.subplots(len(time_emph),1, figsize = (6,len(time_emph)*7))
        else:
            ax_all_all = []
        add_bar_dynamics(coefficients, ax_all_all = ax_all_all, min_max_points = time_emph, colors = colors_dyns, 
                         centralize = True)

        if isinstance( axs_basis_colored ,list) and len( axs_basis_colored ) == 0:
            if base_narrow:
                if quiver_3d: fig_basis_colored , axs_basis_colored = plt.subplots( len(F),1,figsize = (5,6*len(F)),subplot_kw={'projection':'3d'})
                else: fig_basis_colored , axs_basis_colored = plt.subplots( len(F),1,figsize = (5,6*len(F)))
                
            else:
                if quiver_3d: fig_basis_colored , axs_basis_colored = plt.subplots( 1, len(F), figsize = (6*len(F),5),subplot_kw={'projection':'3d'})
                else: fig_basis_colored , axs_basis_colored = plt.subplots( 1, len(F), figsize = (6*len(F),5))
        [quiver_plot(f,-range_p, range_p, -range_p, range_p, ax = axs_basis_colored[f_num],alpha = 0.7, chosen_color =  colors_dyns[f_num], w = w, type_plot = type_plot, cons_color = cons_color, quiver_3d=quiver_3d ) for f_num, f in enumerate(F)]
        [remove_edges(ax_spec) for ax_spec in axs_basis_colored]
        if quiver_3d:        [ax.set_zticks([]) for ax in axs_basis_colored]
       

            
def plot_reco_dyn(coefficients, F, time_point, type_plot = 'quiver', range_p = 10, color = 'black',
                  w = 0.05/3, ax = [], cons_color= False, to_remove_edges = False, projection = [0,1], 
                  return_artist = False,
                  xlabel = 'x',ylabel = 'y',quiver_3d = False):
    if isinstance(ax,list) and len(ax) == 0:
  
        fig, ax = plt.subplots()

    if len(F) == 3:
        merge_dyn_at_t_break = coefficients[0,time_point] * F[0]+coefficients[1,time_point] * F[1]+coefficients[2,time_point] * F[2]

        if not quiver_3d:      

            merge_dyn_at_t_break = merge_dyn_at_t_break[:, projection]
            merge_dyn_at_t_break = merge_dyn_at_t_break[projection,:]

    elif len(F) == 2:
        merge_dyn_at_t_break = coefficients[0,time_point] * F[0]+coefficients[1,time_point] * F[1]

    art = quiver_plot(sub_dyn = merge_dyn_at_t_break, chosen_color = color,  xmin = -range_p, 
                      xmax = range_p, ymin= -range_p,ymax= range_p, ax = ax, w = w, type_plot=type_plot,
                      cons_color= cons_color, return_artist = return_artist, xlabel = xlabel, ylabel = ylabel,
                      quiver_3d = quiver_3d)
    if to_remove_edges: remove_edges(ax)
    if return_artist:
        return art



def plot_c_space(coefficients,latent_dyn = [], axs = [], fig = [], xlim = [-50,50], ylim = [-50,50], add_midline = True, d3 = True, cmap = 'winter',
                 color_sig = [], legend = True,remove_back = True,return_map = False, lw = 2, ls_c = '--',
                 title = '', times_plot= [], cmap_f = [], elev = 0, azim = 0,colors_coeffs  = [], zlim =[], size_legend = 24,
                 projection = [0,1,2],colorbar = True):
    if len(times_plot) > 0 and isinstance(cmap_f, list): cmap_f = plt.cm.get_cmap(cmap)
    if len(color_sig) == 0:    color_sig = latent_dyn[0,:-1]
    if coefficients.shape[0] > 3:
        coefficients = coefficients[projection,:]
        latent_dyn = latent_dyn[projection,:]
    #if isinstance(axs, list) and len(axs) == 0:
      
    if coefficients.shape[0] == 3:
        fig, axs = create_ax(axs, return_fig = True,proj = 'd3', size = (15,15))

        d3 = True
        h = axs.scatter(coefficients[0,:], coefficients[1,:],coefficients[2,:], c = color_sig, cmap = cmap)
        if len(times_plot) > 0:
            axs.scatter(coefficients[0,times_plot], coefficients[1,times_plot], coefficients[2,times_plot],
                        c =cmap_f(color_sig[times_plot]/np.max(color_sig)),s = 500 )
    elif coefficients.shape[0] == 2:
        if d3:       
          
            fig, axs = create_ax(axs, return_fig = True,proj = 'd3', size = (15,15))
            h = axs.scatter(coefficients[0,:], coefficients[1,:], np.arange(coefficients.shape[1]), c = color_sig, cmap = cmap)
            if len(times_plot) > 0:
                zax = np.arange(coefficients.shape[1])
                axs.scatter(coefficients[0,times_plot], coefficients[1,times_plot], zax[times_plot], c =cmap_f(color_sig[times_plot]/np.max(color_sig)),s = 500 )
        else:
      
            fig, axs = create_ax(axs, return_fig = True,proj = 'd2', size = (15,15))
            h = axs.scatter(coefficients[0,:], coefficients[1,:], c = color_sig, cmap = cmap)
            if len(times_plot) > 0:
                axs.scatter(coefficients[0,times_plot], coefficients[1,times_plot],  c =cmap_f(color_sig[times_plot]/np.max(color_sig)),s = 500 )
    else:
        print('Invalid coefficients shape in axis 0')
    
    if len(xlim) > 0:    axs.set_xlim(xlim)
    if len(ylim) > 0:    axs.set_ylim(ylim)
    if not isinstance(fig, list) and colorbar:    
        fig.colorbar(h)

    if d3: 
        if remove_back:
            axs.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs.grid(False)
            axs.set_axis_off()
        if len(zlim) > 0:    axs.set_zlim(zlim)
    if add_midline:
        if d3:
            if len(colors_coeffs) != 3:   def_colors = ['black']*3
            else:  def_colors = colors_coeffs 
            axs.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs.plot([0,0],[np.min(coefficients[1,:]),np.max(coefficients[1,:])],[0,0], color = def_colors[1], ls =ls_c, alpha = 0.3, label = '$c_%d$'%(projection[0]+1), lw = lw)
            axs.plot([np.min(coefficients[0,:]),np.nanmax(coefficients[0,:])],[0,0],[0,0], color = def_colors[0], ls = ls_c, alpha = 0.3, label = '$c_%d$'%(projection[1]+1), lw = lw)
            
            if coefficients.shape[0] ==3:
            
                axs.plot([0,0],[0,0],[np.min(coefficients[2,:]),np.max(coefficients[2,:])],color = def_colors[2], alpha = 0.3, ls = ls_c, label =  '$c_%d$'%(projection[2]+1), lw = lw)
                
            else:
                axs.plot([0,0],[0,0],[0,1.3*coefficients.shape[1]],color = def_colors[2], alpha = 0.3, ls = '--')
        
            if legend:
                axs.legend(prop = {'size' : size_legend}, loc = 'upper right')
        else:
            axs.axhline(0, color = 'black', ls = '--', alpha = 0.3)
            axs.axvline(0, color = 'black', ls = '--', alpha = 0.3)
    if len(title) > 0:
        axs.set_title(title)
    if elev !=0 or azim != 0 and d3:
        axs.view_init(elev=elev, azim=-30)
    if return_map: return h

def matching_Fs(F1,F2, keepF1= True, include_scalar = False):
    """    
    Parameters
    ----------
    F1 : list or arrays
        list of kXk np.arrays describing results from one execution.
    F2 : list of arrays
        list of kXk np.arrays describing results from second execution.
    keepF1 : boolean, optional
        to keep order of F1?. The default is True.
    include_scalar : boolean, optional
        To consider multipication by scalar during the matching process?. The default is False.

    Raises
    ------
    ValueError
        if the length of F1 is not equall to the length of F2.

    Returns
    -------
    matched_F1 : list
        ordered F1.
    matched_F2 : list
        ordered F2.

    """
    if len(F1) != len(F2):
        raise ValueError('Length of F1 should be equal to length of F2, but len(F1) = %d and len(F2) = %d'%(len(F1), len(F2)))
    
    perm_list = list(itertools.product(np.arange(len(F1)), repeat=2))

    dist_mat = np.inf*np.ones((len(F1), len(F2)))
    if include_scalar: 
        scalar_mul_base = [1,-1]
        scalar_mat = np.inf*np.ones((len(F1), len(F2)))
        scalar_store = []
    else:
        scalar_mat =  np.ones((len(F1), len(F2)))
    for perm in perm_list:
        if include_scalar:
            opt1 = (F1[perm[0]] - F2[perm[1]])**2
            opt2 = (F1[perm[0]] - (-1)*F2[perm[1]])**2
            opts = [opt1,opt2]
            argmin_er = np.argmin(opts)
            dist_mat[perm] = opts[argmin_er]
            scalar_mat[perm] = scalar_mul_base[argmin_er]
            
        else:
            dist_mat[perm] = np.mean((F1[perm[0]] - F2[perm[1]])**2)
    dist_mat_copy = dist_mat.copy()
    matched_F1 = []
    matched_F2 = []
    if keepF1: ris = [np.nan]*len(F1)
    for i in range(len(F1)):
        ri, ci = np.unravel_index( np.argmin(dist_mat), dist_mat.shape)
        dist_mat[:,ci] = np.inf
        dist_mat[ri,:] = np.inf
        matched_F1.append(F1[ri])        
        matched_F2.append(scalar_mat[ri, ci]*F2[ci])
        if include_scalar:
            scalar_store.append(scalar_mat[ri, ci]) 
        if keepF1: ris[ri] = i
    if keepF1:         
        matched_F1 = list(np.array(matched_F1)[ris])
        matched_F2 = list(np.array(matched_F2)[ris])
        if include_scalar:
            scalar_store = list(np.array(scalar_store)[ris])
    if include_scalar:
        return matched_F1, matched_F2,  scalar_store   
    return matched_F1, matched_F2    

def run_alg(x,y, alg_name = 'knn', problem_type = 'reg', alg_params = {}, rng = 0,   return_score = True,
            to_plot = True, ax = [], fig = [], cmap = 'Purples', test_size = 0.5,  colorbar = True):
    """
    
    """
    mapping = {}
    importance = []
    if problem_type == 'reg':
        if alg_name == 'knn':
            alg_params = {**{'n_neighbors':3}, **alg_params}
            model = KNeighborsRegressor(**alg_params)
            #importance = model.coef_
        elif alg_name == 'log':
            #alg_params = {**{'n_neighbors':3}, **alg_params}
            model = LinearRegression(**alg_params)
        elif alg_name == 'svm':
            model = svm.SVR(kernel='linear', **alg_params)
            #importance = model.coef_
        else:
            raise NameError('Unknown Algorithm Name')
    elif problem_type == 'class':
        unique_y = np.unique(y)
        unique_y_text = ['{:.2f}'.format(theta_option) for theta_option in unique_y]
        mapping = {val:i for i,val in enumerate(unique_y)}
        y = np.array([mapping[y_i] for y_i in y])
        #print(y)
        
        if alg_name == 'knn':
            alg_params = {**{'n_neighbors':3}, **alg_params}
            model =  KNeighborsClassifier(**alg_params)
        elif alg_name == 'log':
            #alg_params = {**{'n_neighbors':3}, **alg_params}
            model = LogisticRegression(multi_class ='multinomial', **alg_params)
            #importance = model.coef_
        elif alg_name == 'svm':
            model = svm.SVC(kernel='linear', **alg_params)
            
        else:
            raise NameError('Unknown Algorithm Name')
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=rng, test_size=test_size, stratify = y)

    model.fit(x_train, y_train)
    try: importance = model.coef_
    except: print('Model does not provide coeffs')
    predictions = model.predict(x_test)
    r2 =model.score(x_test, y_test)

    if problem_type == 'reg':

        if to_plot:
            fig, ax = create_ax(ax, return_fig = True, fig = fig)
        h = ax.scatter(predictions,y_test, c = np.abs(predictions-y_test), cmap = cmap, vmin = -2)

        ax.plot([np.min(predictions),np.max(predictions)], [np.min(predictions),np.max(predictions)], alpha = 0.3, color = 'gray')
        add_labels(ax,ylabel = 'Predictions',xlabel = 'Real', zlabel = None)
        return r2, predictions, importance
    else:
        
        cm = confusion_matrix(y_test, predictions, labels=model.classes_)
        if to_plot:
            fig, ax = create_ax(ax, return_fig = True, fig = fig)
            ConfusionMatrixDisplay(confusion_matrix=cm,  display_labels=unique_y_text).plot(cmap = cmap, ax = ax, colorbar = colorbar)#model.classes_
        #raise ValueError('stop')
        if return_score:
            return cm, predictions, importance, r2
        else:
            return cm, predictions, importance

def lists2list(xss)    :
    return [x for xs in xss for x in xs] 

def mean_change(signal, axis = 0):
    return np.mean(np.abs(np.diff(signal, axis = axis)), axis = axis)
    
    
def add_feature(func, dfs_data, func_name, coefficients, args = {}, type_add = 'each'):

    hstack_dfs = np.vstack([func(coeffs, axis =1, **args).reshape((1,-1)) for coeffs in coefficients])

    for df_num, df in enumerate(dfs_data):
        for rep in range(hstack_dfs.shape[0]):
            dfs_data[df_num].loc[rep,func_name] = hstack_dfs[rep,df_num] #hstack_dfs[]#[df_num,:].flatten()
    return dfs_data
    
def ML_for_shift(coefficients, theta_options, features_to_use = {}, algs =  ['knn', 'svm','log'], 
                 include_imp = True, title = 'predicting orientations', problem_type = 'class', colorbar = True,
                 cmap ='Purples', cmaps = ['Reds','Greens','Blues','Greys','Oranges','Purples'], add_title = '', 
                 type_ml = 'each', add_title_feat = '', ax_cm_class = [], fig_cm_class = [],
                 fig_r2_heat = [], ax_r2_heat = [], top = True, annot_size = 16):
    imp = False
    if problem_type == 'class': pre_name = 'Acc'
    else:  pre_name = 'R2'
    features_to_use = {**{'mean':True, 'median':True, 'std':True, 'perc10':True,'perc90':True, 'mean_change':True}, **features_to_use }

    features_names = [feature for feature in list(features_to_use.keys()) if features_to_use[feature]]
    list_featues_names = lists2list([[feature+str(i) for feature in list(features_to_use.keys()) if features_to_use[feature]]
                          for i in range(coefficients[0].shape[0])])
    
    dfs_data = [pd.DataFrame(columns = features_names, index = np.arange(len(coefficients))) 
                for i in range(coefficients[0].shape[0])]
    collect_acc = pd.DataFrame(index = np.arange(len(dfs_data)), columns = algs )
    if features_to_use.get('mean'):
        dfs_data = add_feature(np.mean, dfs_data, 'mean', coefficients)
    if features_to_use.get('median'):
        dfs_data = add_feature(np.median, dfs_data, 'median', coefficients)
    if features_to_use.get('std'):
        dfs_data = add_feature(np.std, dfs_data,  'std', coefficients)   
    if features_to_use.get('perc10'):
        dfs_data = add_feature(np.percentile, dfs_data, 'perc10', coefficients, args = {'q':10})
    if features_to_use.get('perc90'):
        dfs_data = add_feature(np.percentile, dfs_data,  'perc90', coefficients, args = {'q':90})        
    if features_to_use.get('mean_change'):
        dfs_data = add_feature(mean_change, dfs_data, 'mean_change', coefficients)         
    labels = theta_options
    
    """
    Prediction
    """
    
    fig_cm_class, ax_cm_class = create_ax(ax_cm_class , nums = (len(algs), len(dfs_data)), 
                                          size = (len(dfs_data)*5, 6) ,return_fig = True, fig = fig_cm_class)
    fig_r2_heat, ax_r2_heat = create_ax(ax_r2_heat, nums = (1,1), return_fig = True)
    fig_cm_class.suptitle(add_title)
    if len(algs) == 1 and len(dfs_data) == 1: ax_cm_class = np.array([ax_cm_class]).reshape((-1,1))
    elif len(algs) == 1: ax_cm_class = ax_cm_class.reshape((1,-1))
    elif len(dfs_data) == 1: ax_cm_class = ax_cm_class.reshape((-1,1))    
    dict_results = {alg: [None]*coefficients[0].shape[0] for alg in algs}
    importance_all = [{}]*len(dfs_data)
    for i, df_data_i in enumerate(dfs_data):
        if isinstance(cmap,list):
            cmap_cur = cmap[i]
        else:
            cmap_cur = cmap
        for alg_num, alg in enumerate(algs):
        
            if problem_type == 'class':
                cm, _ , importance, r2 = run_alg(df_data_i,labels, alg_name = alg, problem_type = problem_type, 
                            alg_params = {}, rng = 0, colorbar = colorbar, 
                    to_plot = True, ax = ax_cm_class[alg_num,i], fig = fig_cm_class, cmap = cmap_cur)
                dict_results[alg][i] = cm
            elif problem_type == 'reg':
                r2, predictions, importance = run_alg(df_data_i,labels, alg_name = alg, problem_type = problem_type, 
                            alg_params = {}, rng = 0,  colorbar = colorbar, 
                    to_plot = True, ax = ax_cm_class[alg_num,i], fig = fig_cm_class, cmap = cmap_cur)
            else:
     
                raise ValueError('Invalid problem type')
            collect_acc.loc[i, alg] = r2
            if len(importance) > 0: 
                importance_all[i][alg]= np.abs(importance).mean(0)/np.sum(np.abs(importance).mean(0))
                imp = True
            
    if isinstance(cmap, list): cmap_cur = 'Greys'          
    else:   cmap_cur = cmap
    sns.heatmap(collect_acc.astype(float).abs(), ax = ax_r2_heat, cmap = cmap_cur, 
                alpha = 0.5, annot = True, vmin = 0, vmax = 1 , cbar = colorbar, annot_kws={"size":annot_size})
    add_labels(ax_r2_heat, xlabel = 'Alogrithm', ylabel = '', zlabel = None, title = '%s of algorithm \n '%pre_name + add_title_feat)
    ax_r2_heat.set_yticks(np.arange(len(collect_acc))+0.5)
    ax_r2_heat.set_yticklabels(['$c_%d$'%(i+1) for i in range(len(collect_acc))], fontsize = 16)
    
    if len(dfs_data) > 1:
        if top:
            [[ax_cm_class[alg_num, i].set_title('{:s} \n $c_{:d}$ \n {:s} = {:.2f}'.format(alg, i+1, pre_name, collect_acc.loc[i, alg])) 
              for i in range(len(dfs_data))] 
             for alg_num, alg in enumerate(algs)]
        else:
            [[ax_cm_class[alg_num, i].set_title('{:s} = {:.2f}'.format(pre_name, collect_acc.loc[i, alg])) 
              for i in range(len(dfs_data))] 
             for alg_num, alg in enumerate(algs)]            
    else:
        [[ax_cm_class[alg_num, i].set_title('All Features \n {:s}= {:.2f}'.format(pre_name, collect_acc.loc[i, alg])) for i in range(len(dfs_data))] 
         for alg_num, alg in enumerate(algs)]
    [ax.set_xticklabels(ax.get_xticks(), rotation = 90) for ax in ax_cm_class.flatten()]
    if include_imp and imp:
        fig_importance, ax_importance = create_ax([], return_fig = True, nums = (1, coefficients[0].shape[0]), sharex = True, sharey = True)
        if coefficients[0].shape[0] == 1:
            ax_importance = np.array([ax_importance]).reshape((1,-1)).flatten()
        cmaps_f = [plt.cm.get_cmap(cmap) for cmap in cmaps]
        colors_dyns_rgb = [cmap_f(np.linspace(0.5,1,len(features_names) )) for cmap_f in cmaps_f]

        [pd.DataFrame(importance_all_spec).T.plot.bar(alpha =0.7, ax = ax_importance[imp_num],
                                                      legend = False, color = colors_dyns_rgb[imp_num],) 
        for imp_num, importance_all_spec in enumerate(importance_all)]#
    
        if type_ml == 'each' and len(ax_importance) > 1:
            [ax_imp.set_title('$c_%d$'%(i+1)) for i,ax_imp in enumerate(ax_importance)]
        else:
            [ax_imp.set_title('Feature Importance for All') for i,ax_imp in enumerate(ax_importance)]
        [ax_importance[-1].legend(features_names, loc = 'upper right', prop = {'size':16})]
        fig_importance.suptitle('Model Abs Coefficients')
        
    
    fig_cm_class.tight_layout()
    fig_cm_class.suptitle(title)
    
    return dict_results, dfs_data
    
    
def show_repeats(path, params_update_c = {}, plot_f = True, plot_heat = True, plot_bar = True ):
    """
    Parameters
    ----------
    path : TYPE
        path of the folder. e.g., r'./aug6/cyl_repeats/'.
    params_update_c  : dictionary
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #mydir = Path(cyl_path)
    files = os.listdir(path)
    files = [file for file in files if file.endswith('npy')]
    params_update_c = {**{'update_c_type':'OMP','reg_term':0}, **params_update_c}    
    
    
    files_list  = []
    F_lasts = []
    c_lasts = []

    for file_count, file in enumerate(files):  
        file = path + sep + file
        iter_results = np.load(file, allow_pickle=True).item()
        F = iter_results['F']
        if plot_f and file_count == 0:
            fig, ax= create_ax([], proj= 'd2',nums = (len(files),len(F)), size = (30,5*len(files)), return_fig=True)
        latent_dyn = iter_results['latent_dyn']
        coefficients = iter_results['coefficients'] 
        coefficients = update_c(F = F, latent_dyn=latent_dyn, params_update_c =params_update_c )
        reco = create_reco(latent_dyn, coefficients=coefficients, F = F)
        
        """
        plot f of different iterations
        """
        if plot_f:            
            [sns.heatmap(f, ax = ax[file_count, i], annot = True) for i,f in enumerate(F)]
        F_lasts.append(F)
        c_lasts.append(coefficients)        
        
    paired_distance = [np.zeros((len(F_lasts), len(F_lasts))) for i in range(len(F_lasts[0]))]
    list_iter = list(itertools.product(np.arange(len(F_lasts)),repeat = 2))
    for pair in list_iter:
        mse = [np.mean((F_lasts[pair[0]][i]-F_lasts[pair[1]][i] )**2) for i in range(len(F))]
        for i in range(len(F)):
            paired_distance[i][pair[0],pair[1]] = mse[i]
            
    """
    plot f of different iterations
    """  
    if plot_f:
        fig, ax = create_ax([], nums = (1,len(F)), size = (25,5), return_fig=True)
        [sns.heatmap(paired_distance[i], ax = ax[i], annot = True) for i in range(len(F))]
        [add_labels(ax[i], title = '$f_%d$'%(i+1), zlabel  = None) for i in range(len(F))]
    if plot_bar:
        mean_dist = [paired_distance[i].mean() for i in range(len(paired_distance))]
        std_dist = [paired_distance[i].std()/np.sqrt(len(paired_distance[i].flatten())) for i in range(len(paired_distance))]        
        plt.bar(np.arange(len(mean_dist)), mean_dist, yerr = std_dist)    
        
        
#%% GD functions and tricks        
def adagrad(f, grad, x0, alpha=0.01, eps=1e-8, max_iters=1000):
    """
    Performs gradient descent using Adagrad algorithm.
    :param f: objective function
    :param grad: gradient of the objective function
    :param x0: initial guess for the solution
    :param alpha: learning rate
    :param eps: small number to avoid division by zero
    :param max_iters: maximum number of iterations
    :return: optimal solution
    """
    x = x0
    s = np.zeros_like(x)
    for i in range(max_iters):
        grad_val = grad(x)
        s += grad_val**2
        x -= alpha * grad_val / (np.sqrt(s) + eps)
        if np.linalg.norm(grad_val) < eps:
            break
    return x        



def rmsprop(grad, params, lr, gamma, epsilon=1e-8):
    """
    Parameters
    ----------
    grad : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    lr : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.
    epsilon : TYPE, optional
        DESCRIPTION. The default is 1e-8.

    Returns
    -------
    updates : TYPE
        DESCRIPTION.
        
    In the code above, grad is the gradient of the loss with respect to the parameters, params is a list of parameters, lr is the learning rate, gamma is the decay rate for the moving average of the squared gradient, and epsilon is a small constant to avoid dividing by zero.

    The rmsprop function calculates the updates for each parameter using the RMSProp algorithm, and subtracts the updates from the original parameters.

    """
    grad_sq = [np.zeros_like(param) for param in params]
    grad_sq_hat = [np.zeros_like(param) for param in params]
    updates = [np.zeros_like(param) for param in params]
    for g, p, g_sq, g_sq_hat, update in zip(grad, params, grad_sq, grad_sq_hat, updates):
        g_sq = gamma * g_sq + (1 - gamma) * np.power(g, 2)
        g_sq_hat = g_sq / (1 - np.power(gamma, 2))
        update = lr * g / (np.sqrt(g_sq_hat) + epsilon)
        p -= update
    return updates


def adam(grad, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, m=None, v=None, t=0):
    """
    The m and v inputs are the moving averages of the gradient and the squared gradient, respectively. If not provided, they will be initialized as arrays of zeros with the same shape as the parameters. The lr input is the learning rate, beta1 and beta2 are the decay rates for the moving averages, and epsilon is a small value added to the denominator for numerical stability. The t input is the iteration number, which is used to compute the corrected learning rate lr_t
   
    Implements the Adam optimization algorithm.
    
    Parameters:
    - grad: List of arrays, the gradient of the loss with respect to the parameters.
    - params: List of arrays, the current parameters.
    - lr: float, optional (default=0.001), the learning rate.
    - beta1: float, optional (default=0.9), the decay rate for the first moment estimate.
    - beta2: float, optional (default=0.999), the decay rate for the second moment estimate.
    - epsilon: float, optional (default=1e-8), a small value added to the denominator for numerical stability.
    - m: List of arrays, optional (default=None), the moving average of the gradient. If not provided, it will be initialized as zeros with the same shape as the parameters.
    - v: List of arrays, optional (default=None), the moving average of the squared gradient. If not provided, it will be initialized as zeros with the same shape as the parameters.
    - t: int, optional (default=0), the iteration number.
    
    Returns:
    - params: List of arrays, the updated parameters.
    - m: List of arrays, the moving average of the gradient after the update.
    - v: List of arrays, the moving average of the squared gradient after the update.
    """

    if m is None:
        m = [np.zos_like(param) for param in params]
    if v is None:
        v = [np.zeros_like(param) for param in params]
    
    t += 1
    lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
    for i, (param, grad_i, m_i, v_i) in enumerate(zip(params, grad, m, v)):
        m_i = beta1 * m_i + (1 - beta1) * grad_i
        v_i = beta2 * v_i + (1 - beta2) * np.power(grad_i, 2)
        param = param - lr_t * m_i / (np.sqrt(v_i) + epsilon)
        m[i] = m_i
        v[i] = v_i
        params[i] = param
    
    return params, m, v

def adadelta(grad, params, rho=0.9, epsilon=1e-6, delta=None, delta_m=None):
    """
    Implements the Adadelta optimization algorithm.
    takes in the gradient of the loss with respect to the parameters and the current parameters, and returns the updated parameters using the Adadelta update rule 
    Parameters:
    - grad: List of arrays, the gradient of the loss with respect to the parameters.
    - params: List of arrays, the current parameters.
    - rho: float, optional (default=0.9), the decay rate for the moving average of the squared gradient.
    - epsilon: float, optional (default=1e-6), a small value added to the denominator for numerical stability.
    - delta: List of arrays, optional (default=None), the moving average of the gradient. If not provided, it will be initialized as zeros with the same shape as the parameters.
    - delta_m: List of arrays, optional (default=None), the moving average of the squared gradient. If not provided, it will be initialized as zeros with the same shape as the parameters.
    
    Returns:
    - params: List of arrays, the updated parameters.
    - delta: List of arrays, the moving average of the gradient after the update.
    - delta_m: List of arrays, the moving average of the squared gradient after the update.
    """
    if delta is None:
        delta = [np.zeros_like(param) for param in params]
    if delta_m is None:
        delta_m = [np.zeros_like(param) for param in params]
    
    for i, (param, grad_i, delta_i, delta_m_i) in enumerate(zip(params, grad, delta, delta_m)):
        delta_m_i = rho * delta_m_i + (1 - rho) * np.power(grad_i, 2)
        delta_i = np.sqrt(delta_i + epsilon) / np.sqrt(delta_m_i + epsilon) * grad_i
        param = param - delta_i
        delta[i] = delta_i
        delta_m[i] = delta_m_i
        params[i] = param
    
    return params, delta, delta_m



#%% create synethetic data

def apply_zero_on_periods(time_signal, min_periods = [0], max_periods = [50]):
    non_zeros = lists2list([list(np.arange(min_periods[i], max_period))
                 for i, max_period in enumerate(max_periods)])
    zeros = np.setdiff1d(np.arange(len(time_signal)), non_zeros)             
    if len(time_signal) < np.max(max_periods):
        raise ValueError('time signal is shorter then period to cut!')

    time_signal[zeros] = 0
    return time_signal
    
def run_create_sig_dim(n = 3, num_els = 15, n_p = 300,  seed_factor = 1):
    """
    Generates an array of signals with specified dimensions.
    
    Parameters:
    n (int, optional): The number of signals to generate. Defaults to 3.
    num_els (int, optional): The number of elements in each signal (i.e. how many sin / cos are there?). Defaults to 5.
    n_p (int, optional): The number of points in each signal. Defaults to 100.
    min_t (int, optional): The minimum time value for each signal. Defaults to 0.
    max_t (int, optional): The maximum time value for each signal. Defaults to 10.
    
    Returns:
    sig_dim: An array of signals, each represented as a 1D numpy array.
    """    
    x = np.linspace(0,10,n_p)
    sig_dim = []
    for n_sig in range(n):
        np.random.seed(n_sig*seed_factor)
        sig = np.zeros(x.shape)
        for n_els in range(num_els):
            np.random.seed(n_els + n_sig*n_els + 5*seed_factor)
            sign = np.random.choice([-1,1])
            ff = np.random.choice([np.sin, np.cos])
  
            freq = np.random.rand()*5
            bb = np.random.rand(*x.shape)
            sig += sign*ff(freq*x )
        sig_dim.append(sig)
    return np.vstack(sig_dim)



def generate_sparse_coefficients(to_smooth = True, wind = 3,M = 5, T = 1000 ,num_nonzero = 10,
                                 frequency_range = (0.1, 0.5),amplitude_range = (0.5, 1.5),
                                 phase_range = (0, 2 * np.pi), period_duration = 40,
                                 num_periods = 5):
    """
    Generate sparse coefficients representing a mixture of sine waves with periods of sparsity.
    
    Parameters:
    ----------
    to_smooth : bool, optional
        Whether to apply Gaussian smoothing to the generated coefficients, by default True.
    wind : int, optional
        The size of the Gaussian window for smoothing, by default 3.
    M : int, optional
        The number of components in the mixture, by default 5.
    T : int, optional
        The length of the coefficient time series, by default 1000.
    num_nonzero : int, optional
        The number of non-zero coefficients in each component, by default 10.
    frequency_range : tuple, optional
        The range of frequencies for the sine waves, by default (0.1, 0.5).
    amplitude_range : tuple, optional
        The range of amplitudes for the sine waves, by default (0.5, 1.5).
    phase_range : tuple, optional
        The range of phases for the sine waves, by default (0, 2 * np.pi).
    period_duration : int, optional
        The average duration of periods of non-sparse coefficients, by default 40.
    num_periods : int, optional
        The number of periods for each component, by default 5.
    
    Returns:
    -------
    np.ndarray
        Sparse coefficients matrix with shape (M, T), where M is the number of components and T is the length of the time series.
    """
    coefficients = np.zeros((M, T))

    for m in range(M):
        frequency = np.random.uniform(*frequency_range)
        amplitude = np.random.uniform(*amplitude_range)
        phase = np.random.uniform(*phase_range)

        time = np.linspace(0, 1, T)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        sigmoid_wave = sigmoid(sine_wave)

        coefficients[m] = sigmoid_wave

    """
    make sparse
    """
    full_num_periods = M*num_periods
    full_durations = np.random.poisson(period_duration , size = full_num_periods)
    full_periods_start = (np.linspace(0, T, full_num_periods)).astype(int)
    np.random.seed(4)
    np.random.shuffle(full_periods_start)
    full_periods_end = (full_periods_start  + full_durations).astype(int)
    full_periods_end[ full_periods_end > T] = T
    coefficients = np.vstack([apply_zero_on_periods(coefficients[m,:], min_periods =  full_periods_start.reshape((M, num_periods))[m,:], 
                                                    max_periods =  full_periods_end.reshape((M, num_periods))[m,:] )
                              for m in range(M)])
    

    if to_smooth:
        coefficients = gaussian_convolve(coefficients, wind = wind, direction = 1, sigma = 1)
    return coefficients


def add_noise_to_signal_and_resmooth(signal, noise_dist = 'normal',noise_std = 1, wind = 3, sigma_smooth = 1):
    """
    Add noise to a signal and then apply Gaussian smoothing.
    
    This function takes a signal and adds noise to it based on the specified noise distribution
    and noise standard deviation. The noisy signal is then smoothed using Gaussian convolution.
    
    Args:
        signal (numpy.ndarray): The input signal to which noise will be added and then smoothed.
        noise_dist (str, optional): The noise distribution. 'normal' (default) for Gaussian noise.
        noise_std (float, optional): Standard deviation of the noise distribution. Default is 1.
        wind (int, optional): Window size for Gaussian convolution. Default is 3.
        sigma_smooth (float, optional): Standard deviation of the Gaussian kernel for smoothing.
                                        Default is 1.
    
    Returns:
        numpy.ndarray: The smoothed signal after adding noise and applying Gaussian smoothing.
    """
    noisy_signal = signal + noise_std*np.random.randn(*signal.shape)
    return gaussian_convolve(noisy_signal, wind,  1, sigma_smooth )
    



def create_mask_D_given_neurons_and_regions(counts, latent_dim_per_region, num_regions, regions_id = [],  
                                            latent_dim_per_region_full = [], regions_id_full = []):
    """
    Create a mask matrix for neural ensembles based on given counts, latent dimensions, and regions.

    This function generates a mask matrix that represents neural ensembles based on the provided counts,
    latent dimensions, and number of regions. Each region contains a certain number of ensembles with the specified
    latent dimension. The mask matrix is constructed using the `block_diag` function from `scipy.linalg`.

    Parameters:
    counts (list or array-like): Number of neurons per region.
    latent_dim_per_region (list or array-like): Number of ensembles per region.
    num_regions (int or array-like): Number of regions or a list of region indices.

    Returns:
    numpy.ndarray: A mask matrix where each row corresponds to a neural ensemble and each column corresponds
    to a latent dimension. The values indicate the region to which the ensemble belongs.
    """
    
    if len(regions_id) == 0:
        if isinstance(num_regions, (list, tuple, np.ndarray)):
            num_regions = len(np.unique(num_regions))
        return block_diag(*[np.ones((counts[i], latent_dim_per_region[i])) * (i + 1) for i in range(num_regions)])
    else:
        print('pay attention! the first neuron must have an id of 0')
        if len(regions_id) != len(counts):
            raise ValueError('len of regions_id must be = len of counts')
        if len( latent_dim_per_region_full) == []:
            latent_dim_per_region_full = latent_dim_per_region.copy()
        if checkEmptyList(regions_id):
            regions_id = np.arange(num_regions)
        if checkEmptyList(regions_id_full):
            regions_id_full = regions_id.copy()
        if (np.array(regions_id) != np.sort(regions_id)).any():
            regions_id = np.array(regions_id)
            argsort = np.argsort(regions_id)
            counts = counts[argsort]
            latent_dim_per_region = latent_dim_per_region[argsort]
            regions_id = regions_id[argsort]
        if regions_id[0] != 0:
            raise ValueError('first id must be 0')

        dict_id_to_index = {region_id: iterator for iterator, region_id in enumerate(regions_id)}
        block_diag_mat = block_diag(*[np.ones((counts[dict_id_to_index[counter]], 
                                         latent_dim_per_region_full[counter])) * (i + 5)  if i in regions_id
                            else np.ones((1, latent_dim_per_region_full[counter])) * np.nan
                            for counter, i in enumerate( regions_id_full)])

        isna = np.isnan(block_diag_mat.sum(1) ) == False
        block_diag_mat = block_diag_mat[isna,:]

        return block_diag_mat 
def create_3d_ax(num_rows, num_cols, params = {}):
    fig, ax = plt.subplots(num_rows, num_cols, subplot_kw = {'projection': '3d'}, **params)
    return  fig, ax    


def plot_3d(mat, params_fig = {}, fig = [], ax = [], params_plot = {}, type_plot = 'plot'):
    # 
    if checkEmptyList(ax):
        fig, ax = create_3d_ax(1,1, params_fig)
    if type_plot == 'plot':    
        ax.plot(mat[0], mat[1], mat[2], **params_plot)
    else:
        ax.scatter(mat[0], mat[1], mat[2], **params_plot)
    

def create_reco_new(x,  coefficients, F, type_reco = 'lookahead', plus_one = 0,
                    thres_max = 40, thres_min = 5.5):
    T = coefficients.shape[1] + 1
    if not check_1d(x) and type_reco == 'lookahead':
        x = x[:,0]
        x = np.array(x).reshape((-1,1))
    elif  type_reco == 'lookahead':
        x = np.array(x[:,0]).reshape((-1,1))
        
    #t = 0
    if type_reco == 'lookahead':
        print('type_reco')
        print(type_reco)
        print(len(F))
        print(coefficients.shape)
        for t in range(T-1):
            x = np.hstack([x,   
                           (np.sum(np.dstack([coefficients[i,t]*F[i] 
                                                   for i in range(len(F))]), 2) @  x[:,-1] ).reshape((-1,1)) ])
        
        
    else:
        print(len(F))
        x_hat = np.hstack([
            (np.sum(np.dstack([ coefficients[i,t]*F[i] for i in range(len(F)) ]), 2) @   x[:,t].reshape((-1,1))).reshape((-1,1))
            for t in range(x.shape[1]-1)])
        x = np.hstack([x[:,0].reshape((-1,1)), x_hat]) 

        
        
    return x  




        
def promise_unit_norm_F(F, c_t, thres = 1):    
    
    F_mat = return_transition_matrix(F, c_t)
    biggest_eval = return_biggest_eval(F_mat)
    while (np.abs(biggest_eval - thres)) > 0.00005:
        c_t *= thres/np.abs(biggest_eval)
        F_mat = return_transition_matrix(F, c_t)
        
        biggest_eval = return_biggest_eval(F_mat)
        #print(biggest_eval)
    F_mat = return_transition_matrix(F, c_t)
    biggest_eval = return_biggest_eval(F_mat)
   
    return c_t
    
    

def return_transition_matrix(F, c):
    # create the f_mat matrix
    if not check_1d(c):
        return [np.sum(np.dstack([F_i * c[i,t] for i,F_i in enumerate(F)]),2)
            for t in range(c.shape[1])]
    else:
        c = c.flatten()
        return np.sum(np.dstack([F_i * c[i] for i,F_i in enumerate(F)]),2)
    

    
def return_biggest_eval(cur_mat):
    eigenvalues, _ =  np.linalg.eig(cur_mat)
    return np.max([np.real(eval_spec) for eval_spec in eigenvalues])



def dLDS_synth_create_Fs(num_Fs = 3, p = 3, theta = 1, latent_dim = 3):
    #if not discrete:
    F_basic = [create_rotation_mat(theta = theta, axes = axis, dims = p) 
               for axis in ['x','y','z'][:num_Fs]]
    
    if num_Fs <= 3 and latent_dim <= 3:    
        return  F_basic
    
    elif num_Fs <= 6 and latent_dim <= 6:
        F2 = [np.zeros((latent_dim,latent_dim)) for _ in range(num_Fs)]
        counter = 0
        for F_count, F_i in enumerate(F2):
            if np.mod(F_count, 2) == 0:
                F2[F_count][:3,:3] = F_basic[counter]
            elif np.mod(F_count, 2) == 1:
                F2[F_count][-3:,-3:] = F_basic[counter]
                counter += 1
        return F2
    else:
        F_basic = []

        for m in range(num_Fs):
            # Input vector and rotation angle
            np.random.seed(m)
            v = (1*(np.random.rand(latent_dim, latent_dim -2) >= 0.5))

            # Compute the rotated matrix
            result = rotmnd(v, theta)
            F_basic.append(result)
        return F_basic

                
                

    
def rotmnd(v, theta):
    n = v.shape[0]
    M = np.eye(n)
    for c in range(n - 2):
        for r in range(n, c + 1, -1):
            t = np.arctan2(v[r - 1, c], v[r - 2, c])
            R = np.eye(n); 
            R[r - 2:r , r - 2:r ] = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            v = np.dot(R, v)
            M = np.dot(R, M)
    R = np.eye(n)
    R[n - 2:n, n - 2:n] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    M = np.dot(np.linalg.inv(R), M)
    return M



def create_snyth_x_one_session(c, F, x , seed = 0, num_regions = 3):
    c = c.copy()
    np.random.seed(seed)    
    x = np.random.rand(num_regions,1) 
    x_full = create_reco_new(x, c, F)     
    return x_full  
    
def create_snyth_num_per_region(min_per_region, max_per_region, num_regions = 3, seed = 5):
    np.random.seed(seed)
    return np.random.randint(min_per_region, max_per_region + 1, size = num_regions)
    
    
def create_snyth_D(num_per_region = [], num_regions = 3, num_ens_per_region = 1, seed = 5):
    
    
    if checkEmptyList(num_per_region):
        num_per_region = create_num_per_region(min_per_region, max_per_region, num_regions, seed = seed)
    if len(num_per_region) != num_regions:
        raise ValueError('length of num_per_region must be == num_regions')
    if not isinstance(num_ens_per_region, (list,tuple, np.ndarray)):
        num_ens_per_region = [num_ens_per_region]*num_regions
    D_mask = create_mask_D_given_neurons_and_regions(num_per_region, num_ens_per_region, num_regions)
    
    D_dense = np.random.rand(np.sum(num_per_region), np.sum(num_ens_per_region))
    D_dense[D_mask == 0] = 0
    return D_dense, D_mask
    
def create_periodic_sparse_c(M, T, value_insert = 1, period_min = 5, period_max = 20, seed = 0):
    """
    Generate a periodic sparse matrix 'c' with specified parameters.

    Parameters:
    - M (int): Number of rows in the matrix.
    - T (int): Number of columns in the matrix.
    - value_insert (float, optional): Value to be inserted at nonzero indices. Default is 1.11.
    - period_min (int, optional): Minimum period for the periodic pattern. Default is 5.
    - period_max (int, optional): Maximum period for the periodic pattern. Default is 20.
    - seed (int, optional): Seed for reproducibility. Default is 0.

    Returns:
    - c (numpy.ndarray): Sparse matrix with periodic patterns.
    """
    
    np.random.seed(seed)
    c = np.zeros((M,T))
    max_num_periods = int(np.ceil(T/period_min))
    periods = np.random.randint(period_min, period_max, size = max_num_periods)
    keep_nonzero = np.random.randint(0, M, max_num_periods)
    nonzero_indices = np.repeat(keep_nonzero, periods)    
    c[list(nonzero_indices[:T]), list(np.arange(T))] = value_insert
    return c
    
    
    
    
    
    
    
    
    

def create_synth_data_one_session(seed = 0, M = 3, T = 100, sigma = 0.8, 
                                  wind = 25, num_regions =3 , min_per_region = 3, max_per_region = 8,
                                  num_ens_per_region = 1, return_F = True, F = [],
                                  c_convolve = [], x_full = [], 
                                  period_min = 43, period_max = 50, 
                                  std_noise = 0.05, value_insert = 1,
                                  w_noise = False, theta = 0.2, perc0 = 0):
    """
    Generate synthetic data for a single session.
    
    Parameters:
    seed (int): Random seed for reproducibility.
    M (int): Number of dynamical operators
    
    T (int): Number of time steps.
    sigma (float): Standard deviation for synthetic noise.
    wind (int): Window size for Gaussian convolution.
    num_regions (int): Number of regions.
    min_per_region (int): Minimum number of elements per region.
    max_per_region (int): Maximum number of elements per region.
    
    Returns:
    tuple: A tuple containing generated data.
        - c_convolve (ndarray): Convolved and thresholded factor matrix.
        - F (ndarray): Factor loading matrix.
        - x_full (ndarray): Latent state sequence.
        - y (ndarray): Observed data sequence.
        - D (ndarray): Mixing matrix.
    old: min_per_region = 3, max_per_region = 8,        
    """
    latent_dim = num_regions*num_ens_per_region
    np.random.seed(seed)
    if not return_F and checkEmptyList(F):
        raise ValueError('you must provide F or calculate F')
        
    # create D
    num_per_region = create_snyth_num_per_region(min_per_region, max_per_region, num_regions = num_regions, seed = seed)

    D, D_mask = create_snyth_D(num_per_region, num_regions, num_ens_per_region, seed = seed)
    zero_cols = np.sum(D_mask,0) == 0
    D = D/((np.sum(D**2, 0)**0.5).reshape((1,-1)) + 1e-19)
    D[:,zero_cols] = 0

    M = num_ens_per_region * M

    if return_F:
        # create F
        F = dLDS_synth_create_Fs(num_Fs = M, theta = theta, latent_dim = latent_dim)  
        
    
    # create c
    
    c = create_periodic_sparse_c(M, T, value_insert = value_insert, period_min = period_min, 
                                 period_max = period_max, seed = seed)
    c_convolve = gaussian_convolve(c, wind = wind, sigma = sigma, norm_sum = True, plot_gaussian = True)

    # create x
    np.random.seed(0)
    x = np.random.rand(latent_dim,1)
    if w_noise:     
        cs_noisy = c_convolve + np.random.randn(*c_convolve.shape)*std_noise
    else:
        cs_noisy = c_convolve
    cs_noisy = cs_noisy[:,15:]
    

    
    x_full = create_reco_new(x, cs_noisy, F, type_reco='lookahead')
    
    # MAKE D SPARSE
    print()
    if perc0 > 0:
        new_perc = perc_to_nulify_in_block_mat(D, perc_or = perc0, 
                                    num_ens_per_region = num_ens_per_region, num_regions = num_regions)
        D_sparse   = nullify_part(D, axis = '1', percent0 = new_perc)
    else:
        D_sparse = D.copy()        
      
    
    
    # create y
    y = D_sparse @ x_full    
    
    return c_convolve, F, x_full, y, D_sparse, D_mask, cs_noisy, num_per_region
    
def create_synth_data_all_sessions(M = 3,T = 100, sigma = 0.7, wind = 25, num_regions = 3, 
                                   min_per_region = 4, max_per_region = 9,  period_min = 40,  period_max = 50, 
                                   num_sessions  = 5,  num_ens_per_region = 1,  std_noise = 0.1, 
                                   value_insert = 1, region_names = ['Tha','Cer','PFC','COAp','MOs','BMA'],  
                                   w_noise = False, theta = 0.2, perc0 = 0):    
    region_names =  region_names[:num_regions]
    cs = {};     Fs = {};     ys = {};     Ds = {};     xs = {};     F = [];     cs_clean = []
    num_per_region_full  = {};     labels = {};     Ds_masks = {}
    
    for session in range(num_sessions):
        np.random.seed(session)
        cs_clean, F, x_full, y, D, D_mask, cs_noisy, num_per_region = create_synth_data_one_session(session, M, 
                                                                                                    T, sigma, 
                                  wind, num_regions, min_per_region, max_per_region, return_F = session == 0, F = F,
                                  c_convolve = cs_clean, x_full = [], std_noise = std_noise, 
                                  value_insert = value_insert,  period_min = period_min,  
                                  period_max = period_max,  w_noise =  w_noise ,
                                  theta = theta, num_ens_per_region = num_ens_per_region, perc0 = perc0)
        cs[session] = cs_noisy
        Fs[session] = F
        ys[session] = y
        Ds[session] = D        
        labels_i = lists2list([[reg]*num_per_region[i] for i, reg in enumerate(region_names)])
        labels[session] = labels_i
        Ds_masks[session] = D_mask
        xs[session] = x_full
        num_per_region_full[session] = num_per_region
        
    return cs, F, ys, Ds, xs, Ds_masks, labels,num_per_region_full
        
    
    
def repeated_index_to_columns(df, col = "pt_position"):
    return df.groupby(level=0).agg(list)[col].apply(pd.Series)

def gaussian_array(length,sigma = 1 , to_norm_type = 'max' ):
    """
    Generate an array of Gaussian values with a given length and standard deviation.
    
    Args:
        length (int): The length of the array.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Default is 1.
        to_norm_type can be 'not', 'max', 'sum'
    Returns:
        ndarray: The array of Gaussian values.
    """
    x = np.linspace(-3, 3, length)  # Adjust the range if needed
    gaussian = np.exp(-(x ** 2) / (2 * sigma ** 2))
    if to_norm_type == 'not':
        pass
    elif  to_norm_type == 'max':
       gaussian = gaussian / np.max(gaussian)
    elif  to_norm_type == 'sum':
        gaussian = gaussian / np.sum(gaussian)
    else:
        raise ValueError('?!')
        
    return gaussian
    

#%% MESOSCALE FUNCTIONS
from scipy.stats import norm    
def spike_times_to_rate_single_neuron(spike_times_single, max_time = 0, 
                                      padded = False,
                                      window_params = {'wind_type':'gauss', 'wind':1, 'std':0.1, 'interval':0.3}, time_axis = []):
    # CONTINUES!!!
    """
    Calculate the firing rate of a single neuron given its spike times.
    
    Parameters:
    - spike_times_single (array): Array of spike times for a single neuron.
    - max_time (float, optional): Maximum time duration. If not provided, it is set to the maximum spike time.
    - padded (bool, optional): Whether to pad spike times beyond the max_time. Default is False.
    - window_params (dict, optional): Parameters for the window function. Default is a Gaussian window with parameters {'wind_type': 'gauss', 'wind': 1, 'std': 0.1, 'interval': 0.3}.
    - time_axis (array, optional): Time axis values. If not provided, it is generated based on window_params['interval'].
    
    Returns:
    - vals_rate (array): Firing rate values.
    - time_axis (array): Time axis values.
    
    Raises:
    - ValueError: If the window type is undefined.
    
    Example:
    ```
    spike_times = np.array([0.1, 0.3, 0.7, 1.2, 1.5])
    rate, time = spike_times_to_rate_single_neuron(spike_times)
    ```
    
    """
    wind = window_params['wind']

    if max_time == 0:
        max_time = np.max(spike_times_single)

    wind_p = wind/2
    if not padded:
        spike_times_single = np.hstack([ -spike_times_single[spike_times_single <= wind_p].reshape((1,-1)), spike_times_single.reshape((1,-1)) , 
                                        max_time + spike_times_single[spike_times_single >= max_time - wind_p].reshape((1,-1))]).flatten()
    if  checkEmptyList(time_axis):
        time_axis = np.arange(0, max_time, window_params['interval'])
    if window_params['wind_type'] == 'gauss':
        vals_rate =  np.array([gaussian_val_given_t(t, spike_times_single, wind_p, max_time, window_params['std'], 
                                                    to_plot_example=False) for t in time_axis])
        return vals_rate, time_axis
    else:
        raise ValueError('wind type undefined!')
    
def gaussian_val_given_t(t, vals_all, wind_p, max_t, sigma, to_plot_example = False, path_save_fig = '.'):

    min_max_t = [np.max(t - wind_p), np.min(t+wind_p)]
    vals_in_wind = vals_all[(vals_all < min_max_t[1]) & (vals_all >= min_max_t[0])]


    g_vals = gaussian_pdf(vals_in_wind, t, sigma)


    return np.sum(g_vals)


def gaussian_pdf(x, mu, sigma):
    """
    Calculate Gaussian Probability Density Function (PDF) values.

    Parameters:
    - x: array-like, values at which to evaluate the PDF
    - mu: mean of the distribution
    - sigma: standard deviation of the distribution

    Returns:
    - y: array, Gaussian PDF values corresponding to the input x values
    """
    y = norm.pdf(x, loc=mu, scale=sigma)
    return y


def spike_times_to_rate_several_neurons(spike_times, to_plot = False, window_params = {}, max_time = 0,
                                        return_time_axis = False):
    window_params = {**{'wind_type':'gauss', 'wind':0.25, 'std':0.03, 'interval':0.05}, **{}}
    # ideally max time will be the end of the trial time
    if max_time == 0:
        max_time = np.max([np.max(el) for el in spike_times])
    time_axis = np.arange(0, max_time, window_params['interval'])
    
    rates = np.vstack([spike_times_to_rate_single_neuron(spike_times_i, window_params = window_params, max_time = max_time, time_axis = time_axis)[0]  for spike_times_i in spike_times])
    if return_time_axis:
        return rates, time_axis
    return rates


def plot_rate_vs_spikes(spikes, rate, time_axis, fig = [], ax = [], params_scatter = {}, params_plot = {}, gen_value = 0):

    if checkEmptyList(ax):
        fig, ax = plt.subplots()
    if gen_value == 0:
        gen_value = np.mean(rate)
        
    ax.scatter(spikes, gen_value*np.ones(len(spikes)), marker = '.', **params_scatter)
    ax.plot(time_axis, rate, **params_plot)
    
    
def take_spikes_in_time_range(spikes, min_t, max_t, reduce_min = True):
    if isinstance(spikes, list) or isinstance(spikes[0], np.ndarray):
        return [ take_spikes_in_time_range(spikes_i, min_t, max_t, reduce_min) for spikes_i in spikes]

    spikes_in_range =  spikes[(spikes >= min_t) & (spikes <= max_t)]
    if reduce_min:
        spikes_in_range -= min_t
    return  spikes_in_range
    




def is_pal(s):
    return s == s[::-1]



def return_longest_pal(s, ans = 0, chosen_s = '', was = []):

    if len(s) == 1:

        was.append(s)
     
        if ans < 1:
            chosen_s = s
        return max([ans, 1]), chosen_s 
    elif len(s) == 2 and s[0] == s[1]:
        print('yay!!')
        was.append(s)
        if ans < 2:
            chosen_s = s
        return max([ans, 2]), chosen_s 
    elif len(s) == 2 :
      
        was.append(s)
        if ans < 1:
            chosen_s = s
        return max([ans, 1]), chosen_s[0] 
    elif is_pal(s):
     
        was.append(s)
        if ans < len(s):
            chosen_s = s
        return max([ans, len(s)]), chosen_s 
    else:

        if s[:-1] not in was:
      
            was.append(s[:-1])
            ans1, chosen_s1 = return_longest_pal(s[:-1] , ans, chosen_s, was)
            
            
        else:
            ans1 = 0
            chosen_s1 =  s[:-1]

        
        if ans1 > ans:
            ans = ans1
            chosen_s =  chosen_s1

        if s[1:] not in was:
        
            was.append(s[1:])
            ans2, chosen_s2 = return_longest_pal(s[1:] , ans, chosen_s, was)
        else:
            ans2 = 0
            chosen_s2 =  s[1:]

        if ans2 > ans:
            ans = ans2
            chosen_s =  chosen_s2

         
        return ans, chosen_s







    
    
