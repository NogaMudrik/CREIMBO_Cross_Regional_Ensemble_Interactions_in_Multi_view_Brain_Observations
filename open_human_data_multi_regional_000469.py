# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 04:54:21 2024

@author: anonymous

notes:
    - dandi id 000469
    - original's data paper https://www.nature.com/articles/s41597-024-02943-8
    - Kyzar M, Kamiński J, Brzezicka A, Reed CM, Chung JM, 
    Mamelak AN, Rutishauser U. Dataset of human-single neuron activity during a Sternberg working memory task. Sci Data. 2024 Jan 18;11(1):89. doi: 10.1038/s41597-024-02943-8. PMID: 38238342; PMCID: PMC10796636.
    - dandi archive https://dandiarchive.org/dandiset/000469/0.240123.1806/files?location=sub-10&page=1
    
"""

"""
params_folder
"""
from main_CREIMBO import *

import glob

to_save = True
two_exameples  = False 
res = 0.03 
only_part_sessions = False # True

T_min = 42/res
T_cut = T_min +  5500


print('overall %.1f sec'%(T_cut*res))


if two_exameples and 'ALL_PHD_MATERIALS' in os.getcwd():
    path_files = r'E:\ALL_PHD_MATERIALS\CODES\dLDS-multi-regional\human_data_from_colab\subject10_sessions_to_take'
    

path_files =  r'E:\ALL_PHD_MATERIALS\CODES\dLDS-multi-regional\FROM_DANDI_SERVER\from_drive\save_res_from_opening-20240422T053449Z-001\save_res_from_opening\spike_times'

files = glob.glob(path_files + os.sep + '*.npy' )
files_neural_activity = [file for file in files if  'spike' in file ]


if (not two_exameples) and 'ALL_PHD_MATERIALS' in os.getcwd():    
    files_areas_path =  r'E:\ALL_PHD_MATERIALS\CODES\dLDS-multi-regional\FROM_DANDI_SERVER\from_drive\save_res_from_opening-20240422T053449Z-001\save_res_from_opening\regions_dict' 
    files_areas = glob.glob(files_areas_path + os.sep +'*.npy')

else:
    files_areas = np.array([file for file in files if  'regions' in file ])

print('finish finding  files')
names_files_neuro = {key:key.split(os.sep)[-1] for key in files_neural_activity}



sessions_neurons_firing_times_to_key = {key:key.replace('spike_times_values_dict_','').replace('.npy', '') 
                      for key in names_files_neuro.values()}


names_files_area = {key:key.split(os.sep)[-1] for key in files_areas}
regions_to_key = {key:key.replace('regions_dict_','').replace('.npy', '')  for key in names_files_area.values()}

if only_part_sessions:
    num_sessions = 8
    vals = list(sessions_neurons_firing_times_to_key.values())[:num_sessions]
    sessions_neurons_firing_times_to_key = {key:val for key, val in sessions_neurons_firing_times_to_key.items() if val in vals}
    regions_to_key = {key:val for key, val in regions_to_key.items() if val in vals}
    

session_to_filename = {val:key for key,val in sessions_neurons_firing_times_to_key.items()}



area_key_to_region = {val:key for key, val in regions_to_key.items()}


    
    
    
"""
find areas (neuron X area)
""" 
locs_keep = {}   
area_non_empty = {}
for file_area_full in files_areas:
    
    file_area = names_files_area[file_area_full]
    if file_area in regions_to_key:
        key = regions_to_key[file_area]
        locs_area = np.load(file_area_full, allow_pickle = True).item()
        locs_area = np.array([locs_area[val] for val in range(len(locs_area))])
        where_not_empty = np.where(locs_area != '')[0]
        locs_area_no_empty = locs_area[where_not_empty]
        
        locs_keep[key] = where_not_empty
        area_non_empty[key] = locs_area_no_empty
        
neural_dict = {}
neural_dict_only_with_region = {}
for neural_file in files_neural_activity:
    if neural_file in sessions_neurons_firing_times_to_key:
        file_name = neural_file.split(os.sep)[-1].replace('.npy', '')
        neural_res = np.load(neural_file, allow_pickle = True).item()
        neural_file_short = names_files_neuro[neural_file]
        key = sessions_neurons_firing_times_to_key[neural_file_short]
        neural_dict[key] = neural_res
        print('loaded')
        neural_dict_only_with_region[key] = {key_n:val for key_n, val in neural_res.items() if key_n in locs_keep[key]}

"""
load data - neural data
"""
sessions_neurons_firing_times = {}
sessions_neurons_firing_times_only_with_region = {}

for neural_file in files_neural_activity:
    
    file_name = neural_file.split(os.sep)[-1].replace('.npy', '')
    neural_res = np.load(neural_file, allow_pickle = True).item()
    neural_file_short = names_files_neuro[neural_file]
    if neural_file_short in sessions_neurons_firing_times_to_key:
        key = sessions_neurons_firing_times_to_key[neural_file_short]
        sessions_neurons_firing_times[key] = neural_res
        print('loaded')
        sessions_neurons_firing_times_only_with_region[key] = {key_n:val for key_n, val in neural_res.items() if key_n in locs_keep[key]}
        

"""
trials info
"""

start_stop_times = {}


"""
neural file - mats (3d). Calculate firing rate
"""
#TODO - CHANGE TO POISSON


Ts = {}
Ns = {} # this is a dict of nuber of neurons per session
firings_rates_gauss = {}
firings_rates = {}


areas_to_take = np.unique(lists2list([list(val) for val in area_non_empty.values()]))
for session, firings_times in sessions_neurons_firing_times.items():
    result = np.load(path_files+os.sep +  session_to_filename[session], allow_pickle = True).item() #firings_times #
    result = {neuron: val[(val/res > T_min) & (val/res <= T_cut)] - T_min*res for neuron, val in result.items()}
    firing_rate_mat, firing_rate_mat_gauss, old2new = from_spike_times_to_rate(result,
                                                                               type_convert = 'discrete',
                                 res = res, max_min_val = [], return_T = False, 
                                 T_max = np.inf, T_min = 0,  params_gauss = {})#
    
    firings_rates_gauss[session] = firing_rate_mat_gauss
    firings_rates[session] = firing_rate_mat
    
    Ts[session] = firing_rate_mat_gauss.shape[1]

    N = len(locs_keep[session] )
    Ns[session] = N

if to_save:
    save_name = 'human_all_sub_id_000469_%s_%.3f_onlytwo%s_part_%s'%(today,res, str(two_exameples), str(only_part_sessions))
    np.save( r'E:\ALL_PHD_MATERIALS\CODES\dLDS-multi-regional\FROM_DANDI_SERVER\save_res_from_opening' + os.sep +  'firings_rates_gauss_%s.npy'%save_name , 
            {'firing' : firings_rates_gauss, 'res':res, 'T_min':T_min , 'T_cut':T_cut})
print('find num_include_dict')    
area_non_empty_gen = {}
argsorts = {}
num_include_dict = {} 
for neural_key, areas in area_non_empty.items():
    num_include = 0
    gen_areas = []
    for area in areas:
        gen_area = area
        if gen_area not in areas_to_take:
            gen_area_add = 'zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz'
        else:
            gen_area_add = gen_area
            num_include += 1
        gen_areas.append(gen_area_add)
    argsorts[neural_key] = np.argsort(gen_areas)
    area_non_empty_gen[neural_key] = gen_areas
    num_include_dict[neural_key] = num_include


"""
reorder areas and neurons 
"""
"""
build_d mask one ensemble
"""
Ds = {}
counts_dicts = {}
def create_D_given_areas(areas, unique_ar_base= []):
    unique_ar, counts = np.unique(areas, return_counts = True)
    if checkEmptyList(unique_ar_base):
        unique_ar_base = unique_ar
        
    counts_dict =  {area:count for count, area in zip(counts, unique_ar)}      
    for area in unique_ar_base:
        if area not in unique_ar:
            print('area %s not in ar'%area)
            counts_dict[area] = 0
            
            
    areas_new = np.sort(areas)
    if np.array([ area != area_new for area, area_new in zip(areas, areas_new)]).any():
        raise ValueError('areas must be ordered')
    if np.array([ area != area_new for area, area_new in zip(unique_ar_base, np.sort(unique_ar_base))]).any():
         raise ValueError('areas must be ordered base')
    D = np.zeros( (len(areas_new), len(unique_ar_base) ))
    former_count = 0
    c = 0
    for area_unique in unique_ar_base:
        c+= 1
        count = counts_dict[area_unique]
   
        D[former_count : former_count + count, c-1] = c
        former_count += count
        
    return D, counts_dict, unique_ar
        
def limit_max_value(mat, max_value):    
    mat[mat > max_value] = max_value
    return mat
    
"""
ARGSORT NEURONS
"""
    
firings_rates_gauss_after_argsort = {}   
area_non_empty_gen_after_argsort = {} 


for session, gen_areas in area_non_empty_gen.items():
    # ARGSORT
    argsorts_cur = argsorts[session]
    num_include = num_include_dict[session]
    # CURRENT MAT
    firings_rates_gauss_after_argsort[session] = firings_rates_gauss[session][argsorts_cur][:num_include]
    
    # AREAS    
    area_non_empty_gen_after_argsort[session] = np.array(gen_areas)[argsorts_cur][:num_include]

"""
now cut the neurons further to consider only areas of interest
areas_of_interest = areas to take.
here we want all
"""
print('find neural_new_only_selected_areas')
no_taken = {}
areas_new_only_selected = {}
neural_new_only_selected_areas = {}
for session, gen_areas in area_non_empty_gen_after_argsort.items():
    areas = gen_areas
    areas_args = np.array([c for c,area in enumerate(areas) ])#if area in areas_to_take])
    areas_new = areas[areas_args]
    areas_new_only_selected[session] = areas_new 
    
    neural_activity = firings_rates_gauss_after_argsort[session]
    neural_activity_new = neural_activity[areas_args,:]
    neural_new_only_selected_areas[session] = neural_activity_new
    
    


print('find unique areas')
areas_unique = np.unique(lists2list([np.unique(val) for val in areas_new_only_selected.values()]))

for session, gen_areas in areas_new_only_selected.items():
    areas = gen_areas 
    D,counts_dict, _ =  create_D_given_areas(areas, areas_unique)
    Ds[session] = D
    counts_dicts[session] = counts_dict
is_norm = False    
neural_new_only_selected_areas_before_norm = neural_new_only_selected_areas.copy()
if is_norm:
    
    neural_new_only_selected_areas = {key: limit_max_value(val/(np.percentile(val, 98, 1).reshape((-1,1)) + 1e-9), max_value = 1.1) for key, val in neural_new_only_selected_areas_before_norm.items()}




    
"""
ORGANIZE OUT OF TRIALS. NO NEED HERE SINCE I DO NOT SEP. TO TRIALS.
"""
data_concat = {}


keys = list(neural_new_only_selected_areas.keys())
data_concat_small = {}
small_try_max_neurons = 180
small_try_max_time = 500




data_concat =  neural_new_only_selected_areas.copy()  

data_concat_small = {key:val[:, :small_try_max_time] for key,val in neural_new_only_selected_areas.items()}


  
areas_new_only_selected_small = {file: areas[:small_try_max_neurons] 
                                 for file, areas in areas_new_only_selected.items()}    

keys_to_nums = {key:num for num,key in enumerate(keys)}
num_to_keys = {num:key for num,key in enumerate(keys)}

# graphs 
print('building graphs')
graphs = {key:np.corrcoef(data_concat[key]) for key in keys}


graphs_small = {key:np.corrcoef(data_concat_small[key])  for key in keys}
D_masks =  {key: 1*(D != 0) for key, D in Ds.items()}
D_masks_small= {}
max_reg = 0
for key,D in Ds.items():
    D_cur = D[:small_try_max_neurons] 
    D_cur_sum = np.where(D_cur.sum(0) != 0)[0]
    max_reg  = np.max([D_cur_sum[-1] + 1, max_reg])

for key,D in Ds.items():
    D_cur = D[:small_try_max_neurons, :max_reg]
    D_masks_small[key] = 1*(D_cur !=0)
    
def from_D_mask_to_indices(D):
    return [np.where(D[:,col] != 0)[0] for col in range(D.shape[1])]

indices_regs = {key: from_D_mask_to_indices(D) for key,D in Ds.items()}    
indices_regs_small = {key: from_D_mask_to_indices(D) for key,D in D_masks_small.items()}           

data_active_magnified = {key:data*100 for key,data in data_concat.items()}
data_active_magnified_small = {key:data*100 for key,data in data_concat_small.items()}


full_name_save = 'human_apr_apr19_info_dict_%s.npy'%save_name    
cur_path = os.getcwd()    

new_start_end_dict = {}
if to_save:  
    np.save(r'human_apr_apr19_2_%s.npy'%save_name, {'new_start_end_dict':new_start_end_dict,'data_active'
                                                      :data_active_magnified ,'data':data_active_magnified ,
                                                      'labels': areas_new_only_selected , 'H_dict': graphs, 'unique_regions': np.array(areas_to_take).astype(str), 
                                         'D_masks':D_masks, 'keys_to_num':keys_to_nums ,
                                         'num_to_keys':num_to_keys, 'indices_regs':indices_regs, 'start_stop_times':start_stop_times})



    np.save(r'human_apr_apr19_2_%s_small.npy'%save_name, {'new_start_end_dict':new_start_end_dict,'data_active'
                                                      :data_active_magnified_small ,'data':data_active_magnified_small ,
                                                      'labels': areas_new_only_selected , 
                                                      'H_dict': graphs, 
                                                      'unique_regions': np.array(areas_to_take).astype(str), 
                                         'D_masks':D_masks, 'keys_to_num':keys_to_nums ,
                                         'num_to_keys':num_to_keys, 'indices_regs':indices_regs, 'start_stop_times':start_stop_times})


    save_dict = {
        'neural_new_only_selected_areas': neural_new_only_selected_areas,
        'areas_new_only_selected': areas_new_only_selected,
        'area_non_empty_gen': area_non_empty_gen,
        'area_non_empty_gen_after_argsort':area_non_empty_gen_after_argsort,
        'firings_rates_gauss': firings_rates_gauss,
        'firings_rates_gauss_after_argsort':firings_rates_gauss_after_argsort,
        'Ds': Ds,
        'counts_dicts': counts_dicts,
        'firings_rates': firings_rates,
        'neural_dict_only_with_region': neural_dict_only_with_region,
        'neural_dict': neural_dict,
        'Ns': Ns,
        'Ts': Ts,

        'start_stop_times': start_stop_times,
        'area_non_empty': area_non_empty,
        'locs_keep': locs_keep,
        'names_files_neuro': names_files_neuro,
        'sessions_neurons_firing_times_to_key': sessions_neurons_firing_times_to_key,
        'session_to_filename': session_to_filename,
        'names_files_area': names_files_area,
        'regions_to_key': regions_to_key,
        'area_key_to_region': area_key_to_region,
        'res':res,
      
        'areas_to_take':areas_to_take, 
        'params_gauss': {},
        'is_norm': is_norm,
        'T_cut':T_cut,
        'T_min':T_min,
        'neural_new_only_selected_areas_before_norm': neural_new_only_selected_areas_before_norm
    }
    np.save('human_apr_apr19_info_dict_%s.npy'%save_name, save_dict)
    
    small_dict = {
        'areas_new_only_selected': areas_new_only_selected,
        'area_non_empty_gen': area_non_empty_gen,
        'area_non_empty_gen_after_argsort':area_non_empty_gen_after_argsort,
        'Ds': Ds,
        'counts_dicts': counts_dicts,
        'res':res,
        'neural_dict': neural_dict,
        'Ns': Ns,
        'Ts': Ts,
        'T_cut':T_cut,
  
        'start_stop_times': start_stop_times,
        'area_non_empty': area_non_empty,
        'locs_keep': locs_keep,
        'names_files_neuro': names_files_neuro,
        'sessions_neurons_firing_times_to_key': sessions_neurons_firing_times_to_key,
        'session_to_filename': session_to_filename,
        'names_files_area': names_files_area,
        'regions_to_key': regions_to_key,
        'area_key_to_region': area_key_to_region,

        'areas_to_take':areas_to_take,
        'T_min':T_min,

        'is_norm': is_norm
    }
    np.save('human_apr_19_info_dict_small_%s.npy'%save_name, small_dict)
        
    
    
    
    
    
    
    
    
    
    
    
    
    




































path_hip = r'E:\ALL_PHD_MATERIALS\CODES\LOOKAHEAD_DYNAMICS\hippocampus_data_for_linocs_id_000469'
name_file = r'spike_times_values_dict.npy'

import os
import numpy as np
import seaborn as sns

#path_basic_lookahead = os.getcwd()
# MOVE A PATH
path_save = r'E:\ALL_PHD_MATERIALS\CODES\dLDS-multi-regional'

"""
each key in result is a neuron
"""
result = np.load(path_hip + os.sep + name_file, allow_pickle = True).item()


firing_rate_mat, firing_rate_mat_gauss, old2new = from_spike_times_to_rate(result,
                                                                           type_convert = 'discrete',
                             res =1, max_min_val = [], return_T = False, 
                             T_max = np.inf, T_min = 0,  params_gauss = {})

mean_firing_rates_mean = firing_rate_mat_gauss.mean(1).reshape((-1,1))

#firing_rate_mat_gauss_normalized = firing_rate_mat_gauss / (mean_firing_rates_mean  + 10-9)
firing_rate_mat_gauss_normalized =  firing_rate_mat_gauss/np.percentile(firing_rate_mat_gauss, 99, axis = 1).reshape((-1,1))


# heatmap
sns.heatmap(firing_rate_mat)

sns.heatmap(firing_rate_mat_gauss_normalized)

np.save(r'E:\ALL_PHD_MATERIALS\CODES\dLDS-multi-regional\rate_gaussian_000469_sub10_ses1_mat', firing_rate_mat_gauss_normalized)
np.save(path_save+ os.sep + 'rate_gaussian_000469_sub10_ses1_mat', firing_rate_mat_gauss_normalized)






"""
define variables for multi region
"""



























"""
save
"""