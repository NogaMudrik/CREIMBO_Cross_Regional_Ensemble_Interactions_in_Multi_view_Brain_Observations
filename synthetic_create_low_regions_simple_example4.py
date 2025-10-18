# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 20:04:34 2023

@author:anonymous

The options for synth_type include:
    'simple': Generates synthetic data with a simple structure.
    'simple2': Generates another type of synthetic data with a simple structure.
    'wide_low_neuron': Generates synthetic data with a wider range of neurons and lower neuron density.
    'wide_more_neuron': Generates synthetic data with a wider range of neurons and higher neuron density.
    'wide_more_neuron_multiple_trials': Generates synthetic data with a wider range of neurons and higher neuron density for multiple trials.
    'multiple_ensembles': Generates synthetic data with multiple ensembles of neurons.
    'multiple_ensembles_more_regions': Generates synthetic data with multiple ensembles of neurons and more regions.
    
    
    new '2024_05_15':
        'three_ensembles_four_regions' - multiple ensembles 
"""


from main_CREIMBO import *    


k_graph = 4;
with_graph = False
synth_type = 'three_ensembles_four_regions' 

if synth_type == 'simple':
    path_save_synth_data  = os.getcwd() + os.sep +  r'synth_%s'%synth_type
    path_save_figs_synth_data = path_save_synth_data + os.sep + r'figures_%s'%today
    
    
    w_noise = False
    cs, F, ys, Ds, xs, Ds_masks, labels, num_per_region_full = create_synth_data_all_sessions(M = 3,T = 400,
                                                                                              sigma = 2, wind = 35, num_regions = 3, 
                                       min_per_region = 4, max_per_region = 9,  period_min = 40,  period_max = 50, w_noise = w_noise,
                                       num_sessions  = 5)
    

    H_full = {}
    for session, data_active in ys.items():
       
             
        indices_regs = from_regions_to_indices(np.array(labels[session]) )
        H = make_kernel_3d(data_active, indices_regs, with_kNN = True, with_norm = True, k = k_graph)
        H_full[session] = {i:H[i] for i in range(len(H))}
    
    to_save = True
    
    if to_save:
        np.save(path_save_synth_data + os.sep  + r'synth_multi_high_ground_truth_noise_%s_%s.npy'%(str(w_noise), today), {'cs':cs, 'F':F, 'ys':ys, 'Ds':Ds, 'xs':xs, 'Ds_masks': Ds_masks, 'labels':labels, 
                                                                                        'num_per_region_full':num_per_region_full, 'H_dict':H_full})
        
elif synth_type == 'simple2':
    path_save_synth_data  = os.getcwd() + os.sep +  r'synth_%s'%synth_type
    path_save_figs_synth_data = path_save_synth_data + os.sep + r'figures_%s'%today
    
    if not os.path.exists(path_save_figs_synth_data ):
        os.makedirs(path_save_figs_synth_data)
    w_noise = False
    num_sessions  = 15
    cs, F, ys, Ds, xs, Ds_masks, labels, num_per_region_full = create_synth_data_all_sessions(M = 3,T = 400, sigma = 2, wind = 35, num_regions = 3, 
                                       min_per_region = 4, max_per_region = 9,  period_min = 40,  period_max = 50, w_noise = w_noise,
                                       num_sessions  = num_sessions)
    

    H_full = {}
    for session, data_active in ys.items():

             
        indices_regs = from_regions_to_indices(np.array(labels[session]) )
        H = make_kernel_3d(data_active, indices_regs, with_kNN = True, with_norm = True, k = k_graph)
        H_full[session] = {i:H[i] for i in range(len(H))}
    
    to_save = True
    
    if to_save:
        np.save(path_save_synth_data + os.sep + r'synth_multi_high_ground_truth_noise_%s_%s.npy'%(str(w_noise), today) , {'cs':cs, 'F':F, 'ys':ys, 'Ds':Ds, 'xs':xs, 'Ds_masks': Ds_masks, 'labels':labels, 
                                                                                        'num_per_region_full':num_per_region_full, 'H_dict':H_full})

        
elif synth_type == 'wide_low_neuron':        
    path_save_synth_data  = os.getcwd() + os.sep +  r'synth_%s'%synth_type
    path_save_figs_synth_data = path_save_synth_data + os.sep + r'figures_%s'%today
    if not os.path.exists(path_save_figs_synth_data ):
        os.makedirs(path_save_figs_synth_data )

    w_noise = False
    cs, F, ys, Ds, xs, Ds_masks, labels, num_per_region_full = create_synth_data_all_sessions(M = 3,T = 500, sigma = 2, wind = 20, num_regions = 3, 
                                       min_per_region = 4, max_per_region = 9,  period_min = 50,  period_max = 60, w_noise = w_noise,
                                       num_sessions  = 25)
    print('finished calculating!!')

    H_full = {}
    
    if with_graph:        
        for session, data_active in ys.items():
                 
            indices_regs = from_regions_to_indices(np.array(labels[session]) )
            H = make_kernel_3d(data_active, indices_regs, with_kNN = True, with_norm = True, k = k_graph)
            H_full[session] = {i:H[i] for i in range(len(H))}

    to_save = True

    if to_save:
        np.save(path_save_synth_data + os.sep + r'synth_multi_high_ground_truth_noise_%s_%s.npy'%(str(w_noise), today), {'cs':cs, 'F':F, 'ys':ys, 'Ds':Ds, 'xs':xs, 'Ds_masks': Ds_masks, 'labels':labels, 
                                                                                        'num_per_region_full':num_per_region_full, 'H_dict':H_full})
        
elif synth_type == 'wide_more_neuron':        
    path_save_synth_data  = os.getcwd() + os.sep +  r'synth_%s'%synth_type
    path_save_figs_synth_data = path_save_synth_data + os.sep + r'figures_%s'%today
    if not os.path.exists(path_save_figs_synth_data ):
        os.makedirs(path_save_figs_synth_data )

    w_noise = False
    cs, F, ys, Ds, xs, Ds_masks, labels, num_per_region_full = create_synth_data_all_sessions(M = 3,T = 500, sigma = 2, wind = 20, num_regions = 3, 
                                       min_per_region = 14, max_per_region = 29,  period_min = 50,  period_max = 160, w_noise = w_noise,
                                       num_sessions  = 25)
    print('finished calculating!!')

    H_full = {}
    
    if with_graph:        
        for session, data_active in ys.items():
                
            indices_regs = from_regions_to_indices(np.array(labels[session]) )
            H = make_kernel_3d(data_active, indices_regs, with_kNN = True, with_norm = True, k = k_graph)
            H_full[session] = {i:H[i] for i in range(len(H))}

    to_save = True

    if to_save:
        np.save(path_save_synth_data + os.sep + r'synth_multi_high_ground_truth_noise_%s_%s.npy'%(str(w_noise), today) , {'cs':cs, 'F':F, 'ys':ys, 'Ds':Ds, 'xs':xs, 'Ds_masks': Ds_masks, 'labels':labels, 
                                                                                        'num_per_region_full':num_per_region_full, 'H_dict':H_full})    
        
 

elif synth_type == 'wide_more_neuron_multiple_trials':        
    num_trials = 10
    path_save_synth_data  = os.getcwd() + os.sep +  r'synth_%s'%synth_type
    path_save_figs_synth_data = path_save_synth_data + os.sep + r'figures_%s'%today
    
    if not os.path.exists(path_save_figs_synth_data ):
        os.makedirs(path_save_figs_synth_data )

    w_noise = False
    cs, F, ys, Ds, xs, Ds_masks, labels, num_per_region_full = create_synth_data_all_sessions(M = 3,T = 500*num_trials, sigma = 2, wind = 20, num_regions = 3, 
                                       min_per_region = 14, max_per_region = 29,  period_min = 50,  period_max = 160, w_noise = w_noise,
                                       num_sessions  = 25)
    print('finished calculating!!')

    H_full = {}
    
    if with_graph:        
        for session, data_active in ys.items():
                  
            indices_regs = from_regions_to_indices(np.array(labels[session]) )
            H = make_kernel_3d(data_active, indices_regs, with_kNN = True, with_norm = True, k = k_graph)
            H_full[session] = {i:H[i] for i in range(len(H))}

    to_save = True

    if to_save:
        np.save(path_save_synth_data + os.sep + r'synth_multi_high_ground_truth_noise_%s_%s.npy'%(str(w_noise), today) , {'cs':cs, 'F':F, 'ys':ys, 'Ds':Ds, 'xs':xs, 'Ds_masks': Ds_masks, 'labels':labels, 
                                                                                        'num_per_region_full':num_per_region_full, 'H_dict':H_full})        

elif synth_type == 'wide_more_neuron_multiple_trials':        
    num_trials = 10
    path_save_synth_data  = os.getcwd() + os.sep +  r'synth_%s'%synth_type
    path_save_figs_synth_data = path_save_synth_data + os.sep + r'figures_%s'%today
    
    if not os.path.exists(path_save_figs_synth_data ):
        os.makedirs(path_save_figs_synth_data )

    w_noise = False
    cs, F, ys, Ds, xs, Ds_masks, labels, num_per_region_full = create_synth_data_all_sessions(M = 3,T = 500*num_trials, sigma = 2, wind = 20, num_regions = 3, 
                                       min_per_region = 14, max_per_region = 29,  period_min = 50,  period_max = 160, w_noise = w_noise,
                                       num_sessions  = 25)
    print('finished calculating!!')

    H_full = {}
    
    if with_graph:        
        for session, data_active in ys.items():
           
            indices_regs = from_regions_to_indices(np.array(labels[session]) )
            H = make_kernel_3d(data_active, indices_regs, with_kNN = True, with_norm = True, k = k_graph)
            H_full[session] = {i:H[i] for i in range(len(H))}

    to_save = True

    if to_save:
        np.save(path_save_synth_data + os.sep + r'synth_multi_high_ground_truth_noise_%s_%s.npy'%(str(w_noise), today) , {'cs':cs, 'F':F, 'ys':ys, 'Ds':Ds, 'xs':xs, 'Ds_masks': Ds_masks, 'labels':labels, 
                                                                                        'num_per_region_full':num_per_region_full, 'H_dict':H_full})   
elif synth_type == 'multiple_ensembles_small':        
    num_trials = 10
    path_save_synth_data  = os.getcwd() + os.sep +  r'synth_%s'%synth_type
    path_save_figs_synth_data = path_save_synth_data + os.sep + r'figures_%s'%today
    
    if not os.path.exists(path_save_figs_synth_data ):
        os.makedirs(path_save_figs_synth_data )

    w_noise = False
    num_ens_per_region = 2
    cs, F, ys, Ds, xs, Ds_masks, labels, num_per_region_full = create_synth_data_all_sessions(M = 2,T = 500*num_trials, sigma = 2, wind = 20, num_regions = 3, 
                                       min_per_region = 3, max_per_region = 7,  period_min = 50,  period_max = 160, w_noise = w_noise,
                                       num_sessions  = 5, num_ens_per_region = num_ens_per_region)
    print('finished calculating!!')

    H_full = {}
    
    if with_graph:        
        for session, data_active in ys.items():
                      
            indices_regs = from_regions_to_indices(np.array(labels[session]) )
            H = make_kernel_3d(data_active, indices_regs, with_kNN = True, with_norm = True, k = k_graph)
            H_full[session] = {i:H[i] for i in range(len(H))}

    to_save = True

    if to_save:
        np.save(path_save_synth_data + os.sep + r'synth_multi_high_ground_truth_noise_%s_%s.npy'%(str(w_noise), today) , {'cs':cs, 'F':F, 'ys':ys, 'Ds':Ds, 'xs':xs, 'Ds_masks': Ds_masks, 'labels':labels, 
                                                                                        'num_per_region_full':num_per_region_full, 'H_dict':H_full, 'latent_dim_per_region':num_ens_per_region})   
                
elif synth_type == 'multiple_ensembles':        
    num_trials = 10
    path_save_synth_data  = os.getcwd() + os.sep +  r'synth_%s'%synth_type
    path_save_figs_synth_data = path_save_synth_data + os.sep + r'figures_%s'%today
    
    if not os.path.exists(path_save_figs_synth_data ):
        os.makedirs(path_save_figs_synth_data )

    w_noise = False
    num_ens_per_region = 2
    cs, F, ys, Ds, xs, Ds_masks, labels, num_per_region_full = create_synth_data_all_sessions(M = 2,T = 500*num_trials, sigma = 2, wind = 20, num_regions = 3, 
                                       min_per_region = 14, max_per_region = 29,  period_min = 50,  period_max = 160, w_noise = w_noise,
                                       num_sessions  = 25, num_ens_per_region = num_ens_per_region)
    print('finished calculating!!')

    H_full = {}
    
    if with_graph:        
        for session, data_active in ys.items():
               
            indices_regs = from_regions_to_indices(np.array(labels[session]) )
            H = make_kernel_3d(data_active, indices_regs, with_kNN = True, with_norm = True, k = k_graph)
            H_full[session] = {i:H[i] for i in range(len(H))}

    to_save = True

    if to_save:
        np.save(path_save_synth_data + os.sep + r'synth_multi_high_ground_truth_noise_%s_%s.npy'%(str(w_noise), today) , {'cs':cs, 'F':F, 'ys':ys, 'Ds':Ds, 'xs':xs, 'Ds_masks': Ds_masks, 'labels':labels, 
                                                                                        'num_per_region_full':num_per_region_full, 'H_dict':H_full, 'latent_dim_per_region':num_ens_per_region})   
        
elif synth_type == 'three_ensembles_four_regions':

    path_save_synth_data  = os.getcwd() + os.sep +  r'synth_%s'%synth_type
    path_save_figs_synth_data = path_save_synth_data + os.sep + r'figures_%s'%today
    
    if not os.path.exists(path_save_figs_synth_data ):
        os.makedirs(path_save_figs_synth_data )

    w_noise = False
    num_ens_per_region = 3
    cs, F, ys, Ds, xs, Ds_masks, labels, num_per_region_full = create_synth_data_all_sessions(M = 3,T = 900, sigma = 2, wind = 10, num_regions = 4, 
                                       min_per_region = 4, max_per_region = 29,  period_min = 50,  period_max = 160, w_noise = w_noise,
                                       num_sessions  = 40, num_ens_per_region = num_ens_per_region, value_insert=1.007, perc0 = 80 )

    print('finished calculating!!')

    H_full = {}
    
    if with_graph:        
        for session, data_active in ys.items():                         
            indices_regs = from_regions_to_indices(np.array(labels[session]) )
            H = make_kernel_3d(data_active, indices_regs, with_kNN = True, with_norm = True, k = k_graph)
            H_full[session] = {i:H[i] for i in range(len(H))}

    to_save = True

    if to_save:
        np.save(path_save_synth_data + os.sep + r'three_ensbmeles_four_areas_%s_%s.npy'%(str(w_noise), today) , {'cs':cs, 'F':F, 'ys':ys, 'Ds':Ds, 'xs':xs, 'Ds_masks': Ds_masks, 'labels':labels, 
                                                                                        'num_per_region_full':num_per_region_full, 'H_dict':H_full, 'latent_dim_per_region':num_ens_per_region})   
elif synth_type == 'multiple_ensembles_more_regions':
    num_trials = 10
    path_save_synth_data  = os.getcwd() + os.sep +  r'synth_%s'%synth_type
    path_save_figs_synth_data = path_save_synth_data + os.sep + r'figures_%s'%today
    
    if not os.path.exists(path_save_figs_synth_data ):
        os.makedirs(path_save_figs_synth_data )

    w_noise = False
    num_ens_per_region = 2
    cs, F, ys, Ds, xs, Ds_masks, labels, num_per_region_full = create_synth_data_all_sessions(M = 3,T = 500*10, sigma = 2, wind = 20, num_regions = 3, 
                                       min_per_region = 14, max_per_region = 29,  period_min = 50,  period_max = 160, w_noise = w_noise,
                                       num_sessions  = 25, num_ens_per_region = num_ens_per_region)
    print('finished calculating!!')

    H_full = {}
    
    if with_graph:        
        for session, data_active in ys.items():                         
            indices_regs = from_regions_to_indices(np.array(labels[session]) )
            H = make_kernel_3d(data_active, indices_regs, with_kNN = True, with_norm = True, k = k_graph)
            H_full[session] = {i:H[i] for i in range(len(H))}

    to_save = True

    if to_save:
        np.save(path_save_synth_data + os.sep + r'synth_multi_high_ground_truth_noise_%s_%s.npy'%(str(w_noise), today) , {'cs':cs, 'F':F, 'ys':ys, 'Ds':Ds, 'xs':xs, 'Ds_masks': Ds_masks, 'labels':labels, 
                                                                                        'num_per_region_full':num_per_region_full, 'H_dict':H_full, 'latent_dim_per_region':num_ens_per_region})   

        
                
else:
    raise ValueError('data was not identified!')        
        
cs = {key:val for counter, (key,val) in enumerate(cs.items()) if counter <= 5}         
ys = {key:val for counter, (key,val) in enumerate(ys.items()) if counter <= 5}  
xs = {key:val for counter, (key,val) in enumerate(xs.items()) if counter <= 5}  
Ds = {key:val for counter, (key,val) in enumerate(Ds.items()) if counter <= 5}  
"""
plotting
"""    
to_plot =False
if to_plot:
    
    """
    plot D masks
    """
    fig, axs = plt.subplots(1, len(Ds_masks), sharex = True, sharey = True, figsize = (10*len(Ds_masks),8))
    [sns.heatmap(Ds_masks_i,  ax = axs[i], cbar = False) 
     for i, Ds_masks_i in Ds_masks.items()]
    shapes = [Ds_masks_i.shape for i, Ds_masks_i in Ds_masks.items()]
    max_len = np.max([shapes[i][0] for i in range(len(shapes))])
    [ax.set_ylim([0,max_len]) for ax in axs]
    [add_labels(ax, xlabel = 'regions', ylabel = '', zlabel = '', title = 'session %d'%(i+1)) 
     for i,ax in enumerate(axs)]
    add_labels(axs[0], xlabel = '', ylabel = 'Neurons', zlabel = '', title = '') 

    axs[0].set_ylabel('Neurons')
    axs[0].set_yticks(np.arange(0 , max_len) + 0.5)
    ticklabels = np.arange(1, max_len+1).astype(str)
    ticklabels[1:-1] = ''
    axs[0].set_yticklabels(ticklabels, fontsize = 20)
    plt.suptitle('D masks', fontsize = 30)
    fig.tight_layout()
    plt.savefig(path_save_figs_synth_data + os.sep + 'D_masks.png')

    
    """
    plot D 
    """
    fig, axs = plt.subplots(1, len(Ds_masks), sharex = True, sharey = True, figsize = (10*len(Ds_masks),8))
    [sns.heatmap(Ds_masks_i,  ax = axs[i], cbar = False) 
     for i, Ds_masks_i in Ds.items()]
    shapes = [Ds_masks_i.shape for i, Ds_masks_i in Ds_masks.items()]
    max_len = np.max([shapes[i][0] for i in range(len(shapes))])
    [ax.set_ylim([0,max_len]) for ax in axs]
    [add_labels(ax, xlabel = 'regions', ylabel = '', zlabel = '', title = 'session %d'%(i+1)) 
     for i,ax in enumerate(axs)]
    add_labels(axs[0], xlabel = '', ylabel = 'Neurons', zlabel = '', title = '') 

    axs[0].set_ylabel('Neurons')
    axs[0].set_yticks(np.arange(0 , max_len) + 0.5)
    ticklabels = np.arange(1, max_len+1).astype(str)
    ticklabels[1:-1] = ''
    axs[0].set_yticklabels(ticklabels, fontsize = 20)
    plt.suptitle('D', fontsize = 30)
    fig.tight_layout()
    plt.savefig(path_save_figs_synth_data + os.sep + 'D.png')

    
    """
    plot c
    """
    fig, axs = plt.subplots(len(cs), 1, sharex = True, sharey = True, figsize = (8,len(cs)*8))
    [sns.heatmap(Ds_masks_i,  ax = axs[i], cbar = False) 
     for i, Ds_masks_i in cs.items()]
    shapes = [Ds_masks_i.shape for i, Ds_masks_i in cs.items()]
    max_len = np.max([shapes[i][1] for i in range(len(shapes))])
   
    [add_labels(ax, ylabel = 'sub-dyn', xlabel = '', zlabel = '', title = 'session %d'%(i+1)) 
     for i,ax in enumerate(axs)]
    add_labels(axs[-1], ylabel = '', zlabel = '',  xlabel = 'Time', title = '') 

    
    axs[-1].set_xticks(np.arange(0 , max_len) + 0.5)
    ticklabels = np.arange(1, max_len+1).astype(str)
    ticklabels[1:-1] = ''
    axs[-1].set_xticklabels(ticklabels, fontsize = 20)
    plt.suptitle('c', fontsize = 30)
    fig.tight_layout()
    plt.savefig(path_save_figs_synth_data + os.sep + 'c_heat.png')
    #plt.savefig(path_save_figs_synth_data + os.sep + 'c_heat.svg')
    
    
    """
    plot c
    """
    fig, axs = plt.subplots(len(cs), 1, sharex = True, sharey = True, figsize = (8,len(cs)*8))
    [axs[i].plot(Ds_masks_i.T) 
     for i, Ds_masks_i in cs.items()]
    shapes = [Ds_masks_i.shape for i, Ds_masks_i in cs.items()]
    max_len = np.max([shapes[i][1] for i in range(len(shapes))])
   
    [add_labels(ax, ylabel = 'sub-dyn', xlabel = '', zlabel = '', title = 'session %d'%(i+1)) 
     for i,ax in enumerate(axs)]
    add_labels(axs[-1], ylabel = '', zlabel = '',  xlabel = 'Time', title = '') 

    
    axs[-1].set_xticks(np.arange(0 , max_len) + 0.5)
    ticklabels = np.arange(1, max_len+1).astype(str)
    ticklabels[1:-1] = ''
    axs[-1].set_xticklabels(ticklabels, fontsize = 20)
    plt.suptitle('c', fontsize = 30)
    fig.tight_layout()
    plt.savefig(path_save_figs_synth_data + os.sep + 'c.png')
    #plt.savefig(path_save_figs_synth_data + os.sep + 'c.svg')    
    
    """
    plot y
    """
    fig, axs = plt.subplots(len(ys), 1, sharex = True, sharey = True, figsize = (8,len(cs)*8))
    [sns.heatmap(Ds_masks_i,  ax = axs[i], cbar = False) 
     for i, Ds_masks_i in ys.items()]
    shapes = [Ds_masks_i.shape for i, Ds_masks_i in cs.items()]
    max_len = np.max([shapes[i][1] for i in range(len(shapes))])
   
    [add_labels(ax, ylabel = 'sub-dyn', xlabel = '', zlabel = '', title = 'session %d'%(i+1)) 
     for i,ax in enumerate(axs)]
    add_labels(axs[-1], ylabel = '', zlabel = '',  xlabel = 'Time', title = '') 

    
    axs[-1].set_xticks(np.arange(0 , max_len) + 0.5)
    ticklabels = np.arange(1, max_len+1).astype(str)
    ticklabels[1:-1] = ''
    axs[-1].set_xticklabels(ticklabels, fontsize = 20)
    plt.suptitle('y', fontsize = 30)
    fig.tight_layout()
    plt.savefig(path_save_figs_synth_data + os.sep + 'y.png')
    #plt.savefig(path_save_figs_synth_data + os.sep + 'y.svg')  
    
    """
    plot x
    """
    fig, axs = plt.subplots(len(ys), 1, sharex = True, sharey = True, figsize = (8,len(cs)*8))
    [sns.heatmap(Ds_masks_i,  ax = axs[i], cbar = False) 
     for i, Ds_masks_i in xs.items()]
    shapes = [Ds_masks_i.shape for i, Ds_masks_i in xs.items()]
    max_len = np.max([shapes[i][1] for i in range(len(shapes))])
   
    [add_labels(ax, ylabel = 'sub-dyn', xlabel = '', zlabel = '', title = 'session %d'%(i+1)) 
     for i,ax in enumerate(axs)]
    add_labels(axs[-1], ylabel = '', zlabel = '',  xlabel = 'Time', title = '') 

    
    axs[-1].set_xticks(np.arange(0 , max_len) + 0.5)
    ticklabels = np.arange(1, max_len+1).astype(str)
    ticklabels[1:-1] = ''
    axs[-1].set_xticklabels(ticklabels, fontsize = 20)
    plt.suptitle('x', fontsize = 30)
    fig.tight_layout()
    plt.savefig(path_save_figs_synth_data + os.sep + 'x.png')
    #plt.savefig(path_save_figs_synth_data + os.sep + 'x.svg')    
    
    """
    plot x 3d
    """
    fig, axs = plt.subplots(int(np.ceil(len(ys)/2)), 2, sharex = True, sharey = True, figsize = (8,len(cs)*8), subplot_kw={'projection': '3d'})
    axs = axs.flatten()
    [plot_3d(x_i,  ax = axs[i]) 
     for i, x_i in xs.items()]
    shapes = [x_i.shape for i, x_i in xs.items()]
    max_len = np.max([shapes[i][1] for i in range(len(shapes))])
   
    [add_labels(ax, ylabel = 'sub-dyn', xlabel = '', zlabel = '', title = 'session %d'%(i+1)) 
     for i,ax in enumerate(axs)]
    add_labels(axs[-1], ylabel = '', zlabel = '',  xlabel = 'Time', title = '') 

    

    

    plt.suptitle('x', fontsize = 30)
    fig.tight_layout()
    plt.savefig(path_save_figs_synth_data + os.sep + 'x3d.png')
  
    
    
    """
    plot F masks
    """
    fig, axs = plt.subplots(1, len(F), sharex = True, sharey = True, figsize = (20,7))
    [sns.heatmap(f_i,  ax = axs[i], square = True, cbar =False, annot = True) 
      for i, f_i in enumerate(F)]
    ylabels = ['post-region' if i == 0 else '' for i in range(len(F))]
    
    [add_labels(ax, ylabel = ylabels[i], xlabel = 'pre-region', title = '$f_%d$'%(i+1),zlabel = '' 
                  ) for i, ax in enumerate(axs)]
    
    plt.suptitle('$f_i$', fontsize = 30)
    fig.tight_layout()
    plt.savefig(path_save_figs_synth_data + os.sep + 'F.png')

    
    
    
    
    