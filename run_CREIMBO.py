# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:51:05 2023

@author:  anonymous
"""
import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import date
from datetime import datetime as datetime2
from main_CREIMBO import *
wind_size_opts = [0.1,0.2,0.3,0.4,1]

"""
to run ibl data multi region:
    - 'multi_reg_neuron_per_trial'
    
The options for synth_type include:
    -'simple': Generates synthetic data with a simple structure.
    -'simple2': Generates another type of synthetic data with a simple structure.
    -'wide_low_neuron': Generates synthetic data with a wider range of neurons and lower neuron density.
    -'wide_more_neuron': Generates synthetic data with a wider range of neurons and higher neuron density.
    -'wide_more_neuron_multiple_trials': Generates synthetic data with a wider range of neurons and higher neuron density for multiple trials.
    -'multiple_ensembles': Generates synthetic data with multiple ensembles of neurons.
    -'multiple_ensembles_more_regions': Generates synthetic data with multiple ensembles of neurons and more regions.
    
"""
params_D = {}

ass = 0.3

firing_type = 'gaussian'
wind_size = 0.2 
if wind_size not in wind_size_opts:
    raise ValueError('invalid window size')   
if 'ALL_PHD' in os.getcwd():    
    nersc_or_home = 'h' 
else:
    nersc_or_home = 'n'  

block_D = True
ss = int(str(datetime2.now()).split('.')[-1])
np.random.seed(ss)



#%% to run ALL REGIONS together?
all_regs_together = False # True
if all_regs_together:
    input('pay attention all regs together!')
#%%
"""
choosing levels
"""
noise_level = 3
"""
DEFINE THE TYPE!
"""
# List of options for synth_type
synth_types = [ 'simple2', 'wide_low_neuron', 'wide_more_neuron', 'wide_more_neuron_multiple_trials',
               'multiple_ensembles', 'multiple_ensembles_more_regions', 'simplesimple' ,'three_ensembles_four_regions']
new_synth_type = 'simplesimple'

# if to run it only on a single session?
# single_session  == -1 ->  all sessions
# single_session  == -2 -> first 5 session
# single_session > -1 -> single sesion
single_session = int(input('session num?!'))
if single_session > -1:
    #np.random.seed(3)
    np.random.seed(datetime2.now().hour)

# with dynamics prior?
dynamics_prior = True #  to run CREIMBO set it to True


# D with pca?
PCA_type  = 'global' #'local' #'local'
D_with_PCA = False #True 


if nersc_or_home == 'n':
    type_synth = 'simplesimple'

else:
    type_synth = 'simplesimple' 
       
latent_dim_ok = 'human' not in type_synth
while not latent_dim_ok:
    ss = ss + 1
    
    np.random.seed(ss)
    latent_dim_per_region = np.random.randint(3,8)
  
    latent_dim_ok = True 

    
if 'human' in type_synth.lower():
    if nersc_or_home == 'h':
        kind_data = 'small' 
    else:
        kind_data = False 
        if kind_data:
            kind_data = 'small'
        else:
            kind_data = 'large'
            
# Print the selected type
print("Selected synthetic data type:", type_synth)
lambda_x = 0.4 + np.random.rand()    
"""
Parameters
"""
sparse_f = True
num_subdyns                                         = [3]
include_D                                           = True  
include_patch = False 
update_D_based_on_one_trial = False
infer_x_c_together = False 
D_graph_driven = False
null_D = False

save_comparison_to_ground_truth = False


multiple_D = True
D_with_lasso = False
step_D = 0.0001
reg_vals_new                                        = [1.4] 
update_c_types                                       = ['spgl1']
smooth_term                                          = 0 
add_avg = True
increase_in_sparsity_f = 0
step_D_decay = 0.99999
wind_avg = 3 
use_both_obs_and_latent =  np.random.choice(['obs', 'alternate'] )
k_hard_thres_c = 2
hard_thres_c_freq = 5

nullify_D = False
    
    
addi_save         = date.today().strftime('%d%m%y') # For saving
if 'addition_save' not in locals():    addition_save = []

version_type = 'f'
if version_type.startswith('f'):  
    solver_input = 's' 
    if   solver_input.startswith('s')                           :
        update_c_types                                       = ['spgl1'] 
    else:
        update_c_types                                       = ['inv'] 
    patch_size = np.random.randint(100,500)
    num_patch = np.random.randint(5,100)
    data_min = 0
    data_max = np.inf
else: #short
    update_c_types                                       = ['spgl1']   
    patch_size = np.random.randint(10, 12)
    num_patch = np.random.randint(2,4)
    data_min = 0
    data_max = 100

type_x_infer = 'inv'     # relevant only if infer together is false
  

max_iter                                             = 100 
num_iter                                             = max_iter
is_D_I                                               = False
smooth_term                                          = 5 + np.random.rand()*20 
step_f                                               = 0.2 
GD_decay                                             = 0.99 + 0.01*np.random.rand()
multiply_data                                        = 1
include_patch                                        = np.random.choice([True,True,True,True,False])
step_D_decay                                         = 0.999999
save_freq                                            = 1
"""
Parameters to choose
"""
dt   = 0.1                                                         
l1_D = 0.3 
p = 5                          
max_time                                            = 0 
"""
dynamics type options
    - 'multi_reg_neuron' - neuron level ibl
    - 'vals_logs_source_sink.npy' - epilepsy  - ask amir for data
    - hippocampus - # ask adam for data  
"""
reg_vals_new                                        = [1.5+ np.random.rand()*5]
sparsity_on_f_max = np.random.randint(40,50)
latent_dim = 5
w_noise = False
"""
IMPORTANT
"""
if type_synth != 'ms':
    if   'multi_reg_meso' in type_synth  :

        dynamic_type =  type_synth
        
    elif 'human' in  type_synth:
        dynamic_type =  type_synth
    elif 'three_ensembles' in  type_synth:

        day_created =  '2024_05_15' 
        dynamic_type =   'synth_multi_%s'%type_synth 
    else:
        if type_synth == 'simple':
            day_created =  '2023_12_14'
        else:
            day_created =  '2023_12_15'  
            
        
      
        dynamic_type =   'synth_multi_%s'%type_synth 
else:
    dynamic_type = 'ms'    

sparse_f = True
sparse_f_params = {'axis':'1', 'percent0':20}

increase_in_sparsity_f =  np.random.rand()*6

infer_x_c_together = True
D_graph_driven = False

add_avg = True

params_update_x = {}
D_with_lasso = True
wind_avg = np.random.randint(6,35)
fix_x0 = False
save_comparison_to_ground_truth = False
saving_graph_freq = 10



norm_D_cols = np.random.choice([False,True]) 
if  dynamic_type.startswith('ms'):
    dynamic_type = 'multi_reg_neuron_per_trial'
    num_subdyns                                         = [np.random.randint(5,12)] 
    include_D                                           = True  
    include_patch = False 
    infer_x_c_together = False  
    D_graph_driven = True
    update_D_based_on_one_trial = False
    step_D = 0.0001
    fix_dict = {'fix_c':False, 'fix_D':False, 'fix_f':False, 'fix_x':False}
    change_var = 'non_fixed'
    to_hard_thres_c = np.random.choice([True, False])
    fix_x0 = False
    multiple_D = False
    saving_graph_freq = 2
    D_with_lasso = False
    
elif  dynamic_type.startswith('human_all'): 
       

    num_trials = 40
    num_files = 2
  
    dynamic_type = 'multi_reg_%s_%s'%(type_synth,kind_data  )
    add_avg = np.random.choice([False,True])
    smooth_term = np.random.rand()
    normalize_F = True
    wind_avg = 3    
    num_subdyns  =  np.random.randint(6,17, size = 5)
    include_D                                           = True  
    include_patch = False 
    infer_x_c_together = False  
   
    update_D_based_on_one_trial = False
    step_D = 0.0001
    fix_dict = {'fix_c':False, 'fix_D':False, 'fix_f':False, 'fix_x':False}
    change_var = 'non_fixed'
    to_hard_thres_c = False 
    fix_x0 = False
    multiple_D = True
    saving_graph_freq = 20

    sparse_F =  np.random.choice([False,True]) 
    addition = 'HUMAN'

    D_graph_driven = False
    params_D = {'update_c_type':'spgl1'}
    if kind_data == 'small':
        update_c_types   = ['inv'] #['spgl1']  # ['inv'] #['inv'] # ['spgl1']  
        D_with_lasso =  True# False
    else:
        update_c_types   = ['spgl1']  # ['inv'] #['inv'] # ['spgl1']['inv'] #  
        D_with_lasso = True
   
    reg_vals_new= [np.random.rand()*3]
    sparse_f_params = {'axis':'1', 'percent0':np.random.randint(1,10)}
    update_type_D = 'spgl1'
    lambda_D = l1_D# np.random.rand()*3 #0.3
    
elif  dynamic_type.startswith('human_sub10'):     
    #l1_D = np.random.rand()*3 + 0.2
    num_trials = 40
    num_files = 2

    dynamic_type = 'multi_reg_human_subject10_%s'%kind_data  
    add_avg = np.random.choice([False,True])
    smooth_term = 0.02*np.random.rand()
    normalize_F = True
    wind_avg = 3    
    num_subdyns  =  [np.random.randint(7,17)]  
    include_D                                           = True  
    include_patch = False 
    infer_x_c_together = False  
    D_graph_driven = True
    update_D_based_on_one_trial = False
    step_D = 0.0001
    fix_dict = {'fix_c':False, 'fix_D':False, 'fix_f':False, 'fix_x':False}
    change_var = 'non_fixed'
    to_hard_thres_c = False #np.random.choice([True, False])
    fix_x0 = False
    multiple_D = True
    saving_graph_freq = 1
    D_with_lasso = True #False
    sparse_F = False
    addition = 'HUMAN'


    params_D = {'update_c_type':'spgl1'}
    update_c_types   =  ['spgl1']  #['inv'] #
    l1_D = np.random.rand() + 0.2
    reg_vals_new= [np.random.rand()*20]
    sparse_f_params = {'axis':'1', 'percent0':2}
    lambda_D = np.random.rand()*3 #0.3
    
elif  dynamic_type.startswith('multi_reg_meso'):    
    l1_D = np.random.rand()*3 + 0.2
    num_trials = 40
    num_files = 2
    if 'large' in dynamic_type.lower():
        dynamic_type = 'multi_reg_meso_%d_%d'%(num_trials, num_files)

    
    add_avg = np.random.choice([False,True])
    smooth_term = 0.02*np.random.rand()
    normalize_F = True
    wind_avg = 3

    num_subdyns  =  [np.random.randint(5,17)] 
    include_D                                           = True  
    include_patch = False 
    infer_x_c_together = False  
    D_graph_driven = True
    update_D_based_on_one_trial = False
    step_D = 0.0001
    fix_dict = {'fix_c':False, 'fix_D':False, 'fix_f':False, 'fix_x':False}
    change_var = 'non_fixed'
    to_hard_thres_c = False 
    fix_x0 = False
    multiple_D = True
    saving_graph_freq = 1
    D_with_lasso = True #False
    lambda_D = np.random.rand()*3 #0.3
    sparse_F = False
    addition = 'mseo'

    D_graph_driven = False
    params_D = {'update_c_type':'spgl1'}
    
    update_c_types                                       = ['spgl1'] #'spgl1']#['inv']'inv']
    reg_vals_new= [np.random.rand()*20]
    
    
    
    
    
    
    
    
    
    
    
    
    
elif  dynamic_type.startswith('synth_multi'): # THE SYNTH. CASE
    #update_type_D = 
    addition = 'synth_may24'
    if 'three_ensembles' in dynamic_type: # or ('simplesimple' in dynamic_type and all_regs_together):
        latent_dim_per_region= 3
    elif 'multiple_ensembles' in dynamic_type:
        latent_dim_per_region= 2
    else:
        latent_dim_per_region = 1
    params_D = {'update_c_type':'spgl1'}    
    lambda_x = 0.4 + np.random.rand()    
    sparse_f = np.random.choice([False,True]) # True
    norm_D_cols = np.random.choice([False,True]) #True# False
    num_subdyns                                         = [3] #  [np.random.randint(3,6)] 
    include_D                                           = True  
    include_patch = False 
    update_D_based_on_one_trial = False
    infer_x_c_together = np.random.choice([False,True]) #False 
    D_graph_driven = False # True # False
    null_D = False
    normalize_F = np.random.choice([False,True])
    save_comparison_to_ground_truth = True
    
    
    multiple_D = True
    D_with_lasso = False 
    step_D = np.random.rand()*0.0001
    reg_vals_new                                        = np.random.rand(20)*10 
    update_c_types                                       = ['spgl1'] 
    smooth_term                                          = np.random.choice([0,np.random.rand()*5])
    lambda_D = np.random.rand()*3 
    add_avg = np.random.choice([False,True])
    increase_in_sparsity_f = 0
    step_D_decay = 0.99999999
    wind_avg = 3 
    use_both_obs_and_latent = 'alternate'
    k_hard_thres_c = 2
    hard_thres_c_freq = 5
    fix_x0 = True
    update_type_D = 'spgl1'    
  
    fix_dict = {'fix_c':False,'fix_D':False, 'fix_f':False, 'fix_x':False}
    num_fix = np.random.randint(0,len(fix_dict) - 2)
    fix = np.random.choice(list(fix_dict.keys()),num_fix )

    change_var = '_'.join(fix)
    to_hard_thres_c = np.random.choice([True, False])
    
   
    
    
elif  dynamic_type.startswith('m'):
    dynamic_type = 'multi_reg_neuron'
    num_subdyns                                         = [np.random.randint(5,10)] 
    include_D                                           = True   
    infer_x_c_together = False  
   

elif  dynamic_type.startswith('h2'):
    dynamic_type = 'hippocampus2'
    num_subdyns                                         = [np.random.randint(5,12)] 
    l1_D = 10
    update_type_D  = 'inv'
    include_D                                           = True    
elif  dynamic_type.startswith('h3'):
    infer_x_c_together = False 
    dynamic_type = 'hippocampus2_avg'
    latent_dym = 5
    num_subdyns                                         = [np.random.randint(5,12)] 
    l1_D = 10
  
    include_D                                           = True     
    include_patch = False 
elif  dynamic_type.startswith('h'):
    dynamic_type = 'hippocampus'
    num_subdyns                                         = [np.random.randint(5,12)] 
    include_D                                           = True       
elif  dynamic_type.startswith('t'):
    dynamic_type = 'test'
  
    num_subdyns                                         = [5] 
    include_D                                           = True    
    include_patch  = False
elif  dynamic_type.startswith('es'):
    reg_vals_new                                        = [np.random.rand()]
    dynamic_type = 'epi_amir_short'     
    update_c_types = ['spgl1'] 
    include_patch = False 
    include_D                                           =  False #np.random.choice([False, True])
    if not include_D:
        sparsity_on_f_max = np.random.randint(30,95)
        num_subdyns                                         = [np.random.randint(8,20)] 
    else:
        num_subdyns                                         = [np.random.randint(3,7)]     
elif  dynamic_type.startswith('e2'):
    dynamic_type = 'epi_amir'     
    include_patch = False 
    include_D                                           = False # np.random.choice([False, True])
    if not include_D:
        sparsity_on_f_max = np.random.randint(90,95)
        num_subdyns                                         = [np.random.randint(8,20)] 
    else:
        num_subdyns                                         = [np.random.randint(5,10)] 
                                                            #'lorenz' # input('dynamic type (e.g. lorenz, FHN)')
                                                            
elif  dynamic_type.startswith('e3'):
    dynamic_type = 'epi_amir_normalize'      
    include_D                                           = False # np.random.choice([False, True])
    if not include_D:
        sparsity_on_f_max = np.random.randint(90,95)
        num_subdyns                                         = [np.random.randint(8,20)] 
    else:
        num_subdyns                                         = [np.random.randint(5,10)] 
                                    
                                                            
else:
    raise   ValueError('invalid data type!')                                                         

                                                                


addi_name                                           = str(np.random.randint(9000)) 
include_last_up                                     = False
addition_save.append(addi_save)
weight_observation_eq = 100                
to_load                                             = False



name_auto             = True
normalize_eig         = True
to_print              = False
seed_f                = 0
dt_range              = np.linspace(0.001, 1, 20)
exp_power             = 0.1
max_error             = 1e-9

take_multiple_gd  = np.random.choice([False, True])

# change in future
combine_session = False  # change in future
saving_graphs =  False #True #False #True #False #True #False #True True #

params_infer_x_no_prior = {'lambda_frob': 0, 'lambda_smooth_iters': 0 , 'lambda_smooth_time': 0 , 'lambda_decor': 1}


num_gradient_steps = np.random.randint(1,5)

parameters_f_wise_step = {
    "size_batch": 0.5,
    "ratio_min": 1/20,
    "ratio_max": 20,
    "wise_step": True,
    "num_steps": 5,
    'only_dec_f':True,
    'update_by_mat': False,
    'dur_update': 1
}

type_norm = 'none'


include_mask = True
to_mix_F = True




"""
Runnining over the parameters
"""
if infer_x_c_together:
    lambda_x = lambda_x  + np.random.randint(3000, 10000)
    reg_vals_new = [el + lambda_x for el in reg_vals_new]
for rep in range(20):   
    np.random.seed(ss * rep + rep**2)
    seed = ss * rep + rep**2
    l1_D = np.random.rand()*5 
    for reg_term in reg_vals_new:     

        for update_c_type in update_c_types :
                for num_subs in num_subdyns:
                    if dynamic_type.startswith('synth_multi'): 
                        ranger = np.random.rand(3)*5
                    else:                      
                        ranger = [0] 
                    for noise_level in ranger: 
                        to_save_without_ask   = True                                      
                        sigma_mix_f           = 0.1                

                            
                        if D_with_lasso and single_session <= -1:
                            save_name             = '%s_%g_sub%greg%s_iters%s'%(dynamic_type,num_subs,reg_term, update_c_type, str(num_iter))
                        elif single_session > -1 and D_with_lasso:
                            save_name             = 'SINGLE_AND_SPARSE_%s_%gsub%greg%s_iters%s'%(dynamic_type,num_subs,reg_term, update_c_type, str(num_iter))                              
                        elif single_session > -1:
                            save_name             = 'SINGLE%s_%gsub%greg%s_iters%s'%(dynamic_type,num_subs,reg_term, update_c_type, str(num_iter))                            
                        
                        else:
                            save_name             = 'NOSPARSE%s_%gsub%greg%s_iters%s'%(dynamic_type,num_subs,reg_term, update_c_type, str(num_iter))
                        data                  = [] 
        
                        params_update_c = {'reg_term': reg_term, 'update_c_type':update_c_type,'smooth_term' :smooth_term,
                                            'num_iters': num_iter, 'threshkind':'soft'}
                    
                        if to_save_without_ask: to_save = True
                        else: to_save = str2bool(input('To save?'))
                        if nersc_or_home.startswith('n') and os.sep + 'm' + os.sep in os.getcwd():
                            if 'u2/m/' in os.getcwd():
                                name = os.getcwd().split('u2/m/')[1].split(os.sep)[0]
                            else:
                                name = 'anonymous'
                            path_begin = r'/pscratch/sd/m/%s'%name
                        else:
                            path_begin = '.'
                            
                     
                        #if D_with_lasso:
                        if all_regs_together:
                            if D_with_lasso and single_session <= -1 and not D_with_PCA: # DEFAULT
                                path_save = path_begin + os.sep + 'results' + os.sep +'all_regs_results_%s'%dynamic_type[:-4] +  os.sep + str(datetime2.now()).split()[0].replace('-','_')
                                
                            elif D_with_PCA:
                                if single_session > -1:
                                    path_save = path_begin + os.sep + 'results' + os.sep +'all_regs_PCA_session_res_%s'%dynamic_type[:-4] +  os.sep + 'session_num_%d'%single_session + os.sep + str(datetime2.now()).split()[0].replace('-','_')

                                else:
                                    path_save = path_begin + os.sep + 'results' + os.sep +'all_regs_PCA_session_res_all_tog_%s'%dynamic_type[:-4] + os.sep + str(datetime2.now()).split()[0].replace('-','_')
                            else:
                                path_save = path_begin + os.sep + 'results' + os.sep +'all_regs_single_session_results_%s'%dynamic_type[:-4] +  os.sep + str(datetime2.now()).split()[0].replace('-','_')




                        elif D_with_lasso and single_session == -1 and not D_with_PCA: # DEFAULT
                            path_save = path_begin + os.sep + 'results' + os.sep +'DEFAULT_res_%s'%dynamic_type[:-4] +  os.sep + str(datetime2.now()).split()[0].replace('-','_')
                        
                        elif D_with_lasso and single_session == -2 and not D_with_PCA: # DEFAULT 8 sessions
                            path_save = path_begin + os.sep + 'results' + os.sep +'ONLY8SESSIONS_%s'%dynamic_type[:-4] +  os.sep + str(datetime2.now()).split()[0].replace('-','_')    
                            
                        elif D_with_PCA:
                            if single_session > -1:
                                path_save = path_begin + os.sep + 'results' + os.sep +'PCA_session_res_%s'%dynamic_type[:-4] +  os.sep + 'session_num_%d'%single_session + os.sep + str(datetime2.now()).split()[0].replace('-','_')

                            else:
                                path_save = path_begin + os.sep + 'results' + os.sep +'PCA_session_res_all_tog_%s'%dynamic_type[:-4] + os.sep + str(datetime2.now()).split()[0].replace('-','_')


                                
                                
                        elif not dynamics_prior:
                      
                            addi_prior = 'frob'+ str(params_infer_x_no_prior.get('lambda_frob')).replace('.','_') + 'smooth_iter' + str(params_infer_x_no_prior.get('lambda_frob')).replace('.','_') + 'smooth_t' + str(params_infer_x_no_prior.get('lambda_smooth_time')).replace('.','_') + 'decor' + str(params_infer_x_no_prior.get('lambda_decor')).replace('.','_')
                            if single_session > -1:
                                path_save = path_begin + os.sep + 'results' + os.sep +'ONLY_SIB_res_%s'%dynamic_type[:-4] + os.sep + addi_prior + os.sep + 'session_num_%d'%single_session + os.sep + str(datetime2.now()).split()[0].replace('-','_')

                            else:
                                path_save = path_begin + os.sep + 'results' + os.sep +'ONLY_SIB_ses_all_tog_%s'%dynamic_type[:-4] + os.sep + addi_prior + os.sep + str(datetime2.now()).split()[0].replace('-','_')

                            
                        elif single_session > -1 and D_with_lasso:
                            path_save = path_begin + os.sep + 'results' + os.sep +'Single_SPARSE_ses_%s'%dynamic_type[:-4] +  os.sep + 'ses_num_%d'%single_session + os.sep + str(datetime2.now()).split()[0].replace('-','_')
                        
                        elif single_session > -1:
                            path_save = path_begin + os.sep + 'results' + os.sep +'Single_ses_res_%s'%dynamic_type[:-4] +  os.sep + 'ses_num_%d'%single_session + os.sep + str(datetime2.now()).split()[0].replace('-','_')
                            
                        elif  D_with_lasso:
                            path_save = path_begin + os.sep + 'results' + os.sep +'D_SPARSE_res_%s'%dynamic_type[:-4] +  os.sep + str(datetime2.now()).split()[0].replace('-','_')

                        else:
                            path_save = path_begin + os.sep + 'results' + os.sep +'NO_D_SPARSE_res_%s'%dynamic_type[:-4] +  os.sep + str(datetime2.now()).split()[0].replace('-','_')

                                     
                        path_save_full = path_save + os.sep + 'code_' + str(datetime2.now()).replace(' ','_').replace('.','_').replace(':','_') 
                        if not os.path.exists(path_save_full):
                            os.makedirs(path_save_full)
                        addi_save = {**{'F_noise_level': noise_level, 'include_mask': include_mask, 'to_mix_F' : to_mix_F, 
                                        'to_hard_thres_c':to_hard_thres_c} ,**fix_dict}

                        coefficients, F, latent_dyn, error_reco_array, error_reco_array_med,D,additional_return = train_model_include_D(max_time, dt,  dynamic_type,
                                                                                                                                        num_subdyns = num_subs, 
                                                                                            data = data, step_f = step_f, GD_decay =  GD_decay, 
                                                                                            max_error = max_error, max_iter = max_iter, 
                                                                                            include_D = include_D, seed_f = seed_f, seed = seed,
                                                                                            normalize_eig = normalize_eig, error_order_max_display = 1,error_order_max = 1,
                                                                                            to_print = to_print, params = params_update_c ,  
                                                                                            type_norm = type_norm, to_save_mid = True, save_freq = save_freq,
                                                                                            path_save = path_save_full,
                                                                                            add_avg = add_avg,
                                                                                            wind_avg = wind_avg, 
                                                                                            latent_dim_per_region = latent_dim_per_region,
                                                                                            take_multiple_gd  = take_multiple_gd,
                                                                                            D_graph_driven = D_graph_driven ,
                                                                                            combine_session = combine_session,
                                                                                            saving_graphs = saving_graphs ,
                                                                                            saving_graph_freq = saving_graph_freq,
                                                                                            num_gradient_steps = num_gradient_steps,
                                                                                            sparse_f = sparse_f,
                                                                                            latent_dim = latent_dim, 
                                                                                            include_patch =  include_patch,
                                                                                            patch_size = patch_size, num_patch = num_patch, sparsity_on_f_max = sparsity_on_f_max,
                                                                                            increase_in_sparsity_f = increase_in_sparsity_f ,
                                                                                            include_mask  = include_mask ,
                                                                                            data_min = data_min,
                                                                                            data_max = data_max, 
                                                                                            multiply_data = multiply_data, nullify_big_winds = True, addition = addition,
                                                                                            weight_observation_eq = weight_observation_eq,
                                                                                            lambda_x  = lambda_x,
                                                                                            infer_x_c_together = infer_x_c_together,
                                                                                            type_x_infer = type_x_infer, null_D = null_D, update_D_based_on_one_trial = update_D_based_on_one_trial,
                                                                                            multiple_D  =  multiple_D,  D_with_lasso =  D_with_lasso, step_D=step_D, sparse_f_params = sparse_f_params,
                                                                                            params_update_x =  params_update_x , 
                                                                                            use_both_obs_and_latent = use_both_obs_and_latent, 
                                                                                            k_hard_thres_c = k_hard_thres_c, F = [],
                                                                                            noise_level = noise_level,
                                                                                            addi_save = addi_save,
                                                                                            parameters_f_wise_step = parameters_f_wise_step  ,
                                                                                            to_mix_F = to_mix_F,
                                                                                            to_hard_thres_c=to_hard_thres_c,
                                                                                            dynamics_prior = dynamics_prior, 
                                                                                            fix_x0 = fix_x0, params_D = params_D, single_session = single_session,
                                                                                            save_comparison_to_ground_truth=save_comparison_to_ground_truth,
                                                                                            normalize_F = normalize_F, l1_D = l1_D, 
                                                                                            norm_D_cols = norm_D_cols,
                                                                                            params_infer_x_no_prior = params_infer_x_no_prior,
                                                                                            PCA_type =  PCA_type,
                                                                                            D_with_PCA = D_with_PCA, 
                                                                                            all_regs_together = all_regs_together,
                                                                                            update_type_D = update_type_D, 
                                                                                            lambda_D = lambda_D,                                                                                             
                                                                                            **fix_dict                                                                                            
                                                                                            )             
                        if to_save: 
                            if name_auto:                                pass
                            else:                                        save_name = input('save_name')
                            save_dict = {'F':F, 'coefficients':coefficients,
                                         'latent_dyn': latent_dyn, 
                                         'max_time': max_time, 'dt':dt,'dyn_type':dynamic_type,
                                          'error_reco_array' :error_reco_array, 'D':D}          
                            save_file_dynamics(save_name, [ 'main_folder_results', dynamic_type, 'clean%s'%addi_name,update_c_type] + addition_save, save_dict , path_name = path_begin)
                            
        

    
    
    