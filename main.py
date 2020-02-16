from data_utils import kitti_stuff
from data_utils.fafe_stuff import FafeDetections, FafeMeasurements
from utils import environments
from utils import logger
from utils import plot_stuff
from pmbm import pmbm
from pmbm.config import Config
from utils import constants
import numpy as np
import time, datetime
import os
import sys
# sys.path.append('/home/zhanghm/mot/fafe')

# For CV model:
config_cv =   Config(config_name='CV',
                     motion_model='CV-Kitti',
                     poisson_states_model_name='uniform-CV',
                     filter_class='Linear',
                     measurement_var_xy=1e-1,
                     detection_probability=0.95,
                     survival_probability=0.9,
                     max_nof_global_hypos=100,
                     prune_threshold_global_hypo=-4,
                     prune_threshold_targets=-4,
                     prune_single_existence=1e-3,
                     clutter_intensity=1e-4,
                     gating_distance=3,
                     birth_gating_distance=3,
                     uniform_weight=1,
                     poisson_vx=10,
                     poisson_vy=10,
                     sigma_v=5)

# For CA model:
config_ca =   Config(config_name='CA',
                     motion_model='CA',
                     poisson_states_model_name='uniform-CA',
                     filter_class='Linear',
                     measurement_var_xy=1e-1,
                     detection_probability=0.95,
                     survival_probability=0.9,
                     max_nof_global_hypos=100,
                     prune_threshold_global_hypo=-4,
                     prune_threshold_targets=-4,
                     prune_single_existence=1e-3,
                     clutter_intensity=1e-4,
                     gating_distance=3,
                     birth_gating_distance=3,
                     uniform_weight=1,
                     poisson_vx=10,
                     poisson_vy=10,
                     sigma_v=5)

# For bicycle models: 
config_bc =  Config(config_name='BC',
                    motion_model='Bicycle',
                    poisson_states_model_name='uniform',
                    filter_class='UKF',
                    measurement_var_xy=1e-1,
                    measurement_var_psi=5e-1,
                    detection_probability=0.95,
                    survival_probability=0.9,
                    max_nof_global_hypos=100,
                    prune_threshold_global_hypo=-4,
                    prune_threshold_targets=-4,
                    prune_single_existence=1e-3,
                    clutter_intensity=5e-4,
                    gating_distance=3,
                    birth_gating_distance=2,
                    uniform_weight=1,
                    poisson_v=5,
                    poisson_d=2,
                    sigma_xy_bicycle=0.5,
                    sigma_phi=2.5,
                    sigma_v=2,
                    sigma_d=0.5)
    
# For mixed models
config_mix = Config(config_name='Mixed',
                    motion_model='Mixed',
                    poisson_states_model_name='uniform-mixed',
                    filter_class='Mixed',
                    measurement_var_xy=1e-1,
                    measurement_var_psi=5e-1, #1,
                    detection_probability=0.95,
                    survival_probability=0.9,
                    max_nof_global_hypos=100,
                    prune_threshold_global_hypo=-4,
                    prune_threshold_targets=-4,
                    prune_single_existence=1e-3,
                    clutter_intensity=1e-4,
                    gating_distance=3,
                    birth_gating_distance=3,
                    uniform_weight=1,
                    poisson_vx=10,#5,
                    poisson_vy=10,#5,
                    poisson_v=5,#10,
                    poisson_d=2,#5,
                    sigma_xy_bicycle=0.5,
                    sigma_v=2,#1,
                    sigma_d=0.5,#0.01,
                    sigma_phi=2.5)#1)   


config_cv_car =   Config(config_name='Car-CV',
                     classes_to_track=['Car', 'Van'],
                     motion_model='CV-Kitti',
                     poisson_states_model_name='uniform-CV',
                     filter_class='Linear',
                     measurement_var_xy=1e-1,
                     detection_probability=0.95,
                     survival_probability=0.9,
                     max_nof_global_hypos=100,
                     prune_threshold_global_hypo=-4,
                     prune_threshold_targets=-4,
                     prune_single_existence=1e-3,
                     clutter_intensity=1e-4,
                     gating_distance=3,
                     birth_gating_distance=3,
                     uniform_weight=1,
                     poisson_vx=10,
                     poisson_vy=10,
                     sigma_v=5)
config_cv_ped =   Config(config_name='Ped-CV',
                     classes_to_track=['Pedestrian'],
                     motion_model='CV-Kitti',
                     poisson_states_model_name='uniform-CV',
                     filter_class='Linear',
                     measurement_var_xy=1e-1,
                     detection_probability=0.95,
                     survival_probability=0.9,
                     max_nof_global_hypos=100,
                     prune_threshold_global_hypo=-4,
                     prune_threshold_targets=-4,
                     prune_single_existence=1e-3,
                     clutter_intensity=1e-4,
                     gating_distance=3,
                     birth_gating_distance=3,
                     uniform_weight=1,
                     poisson_vx=10,
                     poisson_vy=10,
                     sigma_v=5)

config_bc_car =  Config(config_name='Car-BC',
                    classes_to_track=['Car', 'Van'],
                    motion_model='Bicycle',
                    poisson_states_model_name='uniform',
                    filter_class='UKF',
                    measurement_var_xy=1e-1,
                    measurement_var_psi=5e-1,
                    detection_probability=0.95,
                    survival_probability=0.9,
                    max_nof_global_hypos=100,
                    prune_threshold_global_hypo=-4,
                    prune_threshold_targets=-4,
                    prune_single_existence=1e-3,
                    clutter_intensity=5e-4,
                    gating_distance=3,
                    birth_gating_distance=2,
                    uniform_weight=1,
                    poisson_v=5,
                    poisson_d=2,
                    sigma_xy_bicycle=0.5,
                    sigma_phi=2.5,
                    sigma_v=2,
                    sigma_d=0.5)
config_bc_ped =  Config(config_name='Ped-BC',
                    classes_to_track=['Pedestrian'],
                    motion_model='Bicycle',
                    poisson_states_model_name='uniform',
                    filter_class='UKF',
                    measurement_var_xy=1e-1,
                    measurement_var_psi=5e-1,
                    detection_probability=0.95,
                    survival_probability=0.9,
                    max_nof_global_hypos=100,
                    prune_threshold_global_hypo=-4,
                    prune_threshold_targets=-4,
                    prune_single_existence=1e-3,
                    clutter_intensity=5e-4,
                    gating_distance=3,
                    birth_gating_distance=2,
                    uniform_weight=1,
                    poisson_v=5,
                    poisson_d=2,
                    sigma_xy_bicycle=0.5,
                    sigma_phi=2.5,
                    sigma_v=2,
                    sigma_d=0.5)
config_cv_fafe =   Config(config_name='PMBM-CV',
                     classes_to_track=['Car', 'Van'],
                     motion_model='CV-Kitti',
                     poisson_states_model_name='uniform-CV',
                     filter_class='Linear',
                     measurement_var_xy=3e-1,
                     detection_probability=0.6,
                     survival_probability=0.9,
                     max_nof_global_hypos=100,
                     prune_threshold_global_hypo=-6,
                     prune_threshold_targets=-6,
                     prune_single_existence=1e-4,
                     clutter_intensity=1e-3,
                     gating_distance=4,
                     birth_gating_distance=4,
                     uniform_weight=0.1,
                     poisson_vx=10,
                     poisson_vy=10,
                     sigma_v=5)

config_cv_fafe1 =   Config(config_name='PMBM-CV-1',
                     classes_to_track=['Car', 'Van'],
                     motion_model='CV-Kitti',
                     poisson_states_model_name='uniform-CV',
                     filter_class='Linear',
                     measurement_var_xy=3e-1,
                     detection_probability=0.6,
                     survival_probability=0.9,
                     max_nof_global_hypos=100,
                     prune_threshold_global_hypo=-6,
                     prune_threshold_targets=-6,
                     prune_single_existence=1e-4,
                     clutter_intensity=1e-3,
                     gating_distance=3,
                     birth_gating_distance=4,
                     uniform_weight=10,
                     poisson_vx=10,
                     poisson_vy=10,
                     sigma_v=5)
config_cv_fafe2 =   Config(config_name='PMBM-CV-2',
                     classes_to_track=['Car', 'Van'],
                     motion_model='CV-Kitti',
                     poisson_states_model_name='uniform-CV',
                     filter_class='Linear',
                     measurement_var_xy=3e-1,
                     detection_probability=0.4,
                     survival_probability=0.9,
                     max_nof_global_hypos=100,
                     prune_threshold_global_hypo=-6,
                     prune_threshold_targets=-6,
                     prune_single_existence=1e-4,
                     clutter_intensity=1e-3,
                     gating_distance=3,
                     birth_gating_distance=4,
                     uniform_weight=10,
                     poisson_vx=10,
                     poisson_vy=10,
                     sigma_v=5)
config_cv_fafe3 =   Config(config_name='PMBM-CV-3',
                     classes_to_track=['Car', 'Van'],
                     motion_model='CV-Kitti',
                     poisson_states_model_name='uniform-CV',
                     filter_class='Linear',
                     measurement_var_xy=3e-1,
                     detection_probability=0.4,
                     survival_probability=0.9,
                     max_nof_global_hypos=100,
                     prune_threshold_global_hypo=-4,
                     prune_threshold_targets=-4,
                     prune_single_existence=1e-4,
                     clutter_intensity=1e-3,
                     gating_distance=3,
                     birth_gating_distance=4,
                     uniform_weight=10,
                     poisson_vx=10,
                     poisson_vy=10,
                     sigma_v=5,
                     sigma_cv=0.5)
config_cv_fafe4 =   Config(config_name='PMBM-CV-4',
                     classes_to_track=['Car', 'Van'],
                     motion_model='CV-Kitti',
                     poisson_states_model_name='uniform-CV',
                     filter_class='Linear',
                     measurement_var_xy=3e-1,
                     detection_probability=0.4,
                     survival_probability=0.9,
                     max_nof_global_hypos=100,
                     prune_threshold_global_hypo=-4,
                     prune_threshold_targets=-4,
                     prune_single_existence=1e-3,
                     clutter_intensity=1e-3,
                     gating_distance=3,
                     birth_gating_distance=4,
                     uniform_weight=10,
                     poisson_vx=10,
                     poisson_vy=10,
                     sigma_v=5,
                     sigma_cv=0.5)
config_cv_fafe5 =   Config(config_name='PMBM-CV-5',
                     classes_to_track=['Car', 'Van'],
                     motion_model='CV-Kitti',
                     poisson_states_model_name='uniform-CV',
                     filter_class='Linear',
                     measurement_var_xy=3e-1,
                     detection_probability=0.4,
                     survival_probability=0.9,
                     max_nof_global_hypos=100,
                     prune_threshold_global_hypo=-6,
                     prune_threshold_targets=-6,
                     prune_single_existence=1e-4,
                     clutter_intensity=1e-4,
                     gating_distance=3,
                     birth_gating_distance=4,
                     uniform_weight=10,
                     poisson_vx=10,
                     poisson_vy=10,
                     sigma_v=5,
                     sigma_cv=0.5)
config_cv_fafe6 =   Config(config_name='PMBM-CV-6',
                     classes_to_track=['Car', 'Van'],
                     motion_model='CV-Kitti',
                     poisson_states_model_name='uniform-CV',
                     filter_class='Linear',
                     measurement_var_xy=3e-1,
                     detection_probability=0.4,
                     survival_probability=0.9,
                     max_nof_global_hypos=100,
                     prune_threshold_global_hypo=-6,
                     prune_threshold_targets=-6,
                     prune_single_existence=1e-4,
                     clutter_intensity=1e-3,
                     gating_distance=3,
                     birth_gating_distance=4,
                     uniform_weight=10,
                     poisson_vx=10,
                     poisson_vy=10,
                     sigma_v=5,
                     sigma_cv=1)
config_cv_fafe7 =   Config(config_name='PMBM-CV-7',
                     classes_to_track=['Car', 'Van'],
                     motion_model='CV-Kitti',
                     poisson_states_model_name='uniform-CV',
                     filter_class='Linear',
                     measurement_var_xy=3e-1,
                     detection_probability=0.4,
                     survival_probability=0.9,
                     max_nof_global_hypos=100,
                     prune_threshold_global_hypo=-6,
                     prune_threshold_targets=-6,
                     prune_single_existence=1e-4,
                     clutter_intensity=1e-3,
                     gating_distance=3,
                     birth_gating_distance=4,
                     uniform_weight=10,
                     poisson_vx=10,
                     poisson_vy=10,
                     sigma_v=5,
                     sigma_cv=1.5)

max_number_of_timesteps = constants.LARGE
#max_number_of_timesteps = 19
sequence_indeces = 0 #np.arange(0,21)  #,12,14,16,20] # [12] #np.arange(0,21)  #[0,1,12,14,16,20]  # np.arange(0,21) ##, 5] #,1,2,3,4,5]

showroom_path = '/home/zhanghm/mot/fafe/showroom/weights_2019-05-03_14-01_epoch_110_fafe'

# 0 for CV, 1 for Bicycle, 2 for mixed, 'all' for running all
# model = 'compare-pmbm-fafe' #'all'
model = 'all' #'all'
model = 0

save_plots = False
show_stuff = False

plot_gospa_comparison = False # For comparing PMBM w. FaFeNet
use_fafenet_detections = False
use_fafenet_measurements = False
track_only_cars_vans = False # If only use car, van, misc as measurements to pmbm

if model == 0: 
    measurement_dimss = [2]
    configs = [config_cv]
if model == 1: 
    measurement_dimss = [3]
    configs = [config_bc]
elif model == 2:
    measurement_dimss = [3]
    configs = [config_mix]
elif model == 'all':
    measurement_dimss = [2,2,3,3]
    configs = [config_cv, config_ca, config_bc, config_mix]
elif model == 'kg-plot':
    measurement_dimss = [2,3]
    configs = [config_cv, config_mix]
elif model == 'diff-mix':
    measurement_dimss = [3,3,3]
    configs = [config_mix, config_mix2, config_mix3]
elif model == 'fafe':
    measurement_dimss = [2]
    configs = [config_fafe]
    use_fafenet_detections = True
elif model == 'ca':
    measurement_dimss = [2]
    configs = [config_ca]
elif model == 'ped-car':
    measurement_dimss = [2,2,3,3]
    configs = [config_cv_car, config_cv_ped, config_bc_car, config_bc_ped]
elif model == 'compare-pmbm-fafe':
    measurement_dimss = [2,2,2,2,2,2,2,2] # [2] #
    configs = [config_cv_fafe, config_cv_fafe1, config_cv_fafe2, config_cv_fafe3, config_cv_fafe4,
              config_cv_fafe5, config_cv_fafe6, config_cv_fafe7] # [config_cv_fafe4] #
    use_fafenet_measurements = True
    log_root = '/home/zhanghm/mot/fafe/showroom/showroom_all/showroom_bev_NN/logs/log-seq'
    sequence_indeces = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20]


log_root = '/home/zhanghm/mot/fafe/showroom/showroom_all/showroom_bev_NN/logs/log-seq'
sequence_indeces = [20]

time_str = datetime.datetime.now().strftime('%m-%d_%H%M')
for sequence_idx in sequence_indeces: 
    log_path = log_root + str(sequence_idx).zfill(4)
    print('Sequence: {}'.format(sequence_idx))
    try:
        root = '/home/zhanghm/Dataset/KITTI/tracking'
        kitti= kitti_stuff.Kitti(ROOT=root, split='training')
        kitti.lbls = kitti.load_labels(sequence_idx)
        kitti.imus = kitti.load_imu(sequence_idx)
        kitti.load_measurements(p_missed=0.05, p_clutter=0.02, sigma_xy=0.1, sigma_psi=0.1)
        if use_fafenet_detections:
            fd = FafeDetections(showroom_path, sequence=sequence_idx)
        if use_fafenet_measurements:
            fm = FafeMeasurements(log_path)
    except AssertionError:
        kitti = kitti_stuff.Kitti(ROOT='/home/zhanghm/Dataset/KITTI/tracking', split='training')
        kitti.lbls = kitti.load_labels(sequence_idx)
        kitti.imus = kitti.load_imu(sequence_idx)
        kitti.load_measurements(p_missed=0.05, p_clutter=0.02, sigma_xy=0.1, sigma_psi=0.1)
        root = '/home/zhanghm/Dataset/KITTI/tracking'   
    imud = kitti.load_imu(sequence_idx)
    
    for ix_config, config in enumerate(configs):
        print('\tConfig: {} ({})'.format(ix_config, config.name))
        measurement_dims = measurement_dimss[ix_config]
        
        if use_fafenet_detections:
            number_of_timesteps = min(fd.max_frame_idx, max_number_of_timesteps)
            fafe_gospa_filename = os.path.join(showroom_path, 'gospa_scores_' + str(sequence_idx).zfill(4) + '.txt')
        else:
            number_of_timesteps = min(kitti.max_frame_idx, max_number_of_timesteps)
            
        _pmbm = pmbm.PMBM(config)
        L = logger.Logger(sequence_idx=sequence_idx, config_idx=ix_config, filename=config.name)

        toc_list = []
        for frame_idx in range(number_of_timesteps):
            #if show_stuff: print('\t\tFrame: {}'.format(frame_idx))
            print("\t\tFrame:{}".format(frame_idx))
            tic = time.time()
            
            if use_fafenet_detections:
                measurements, classes = fd.get_fafe_detections(frame_idx, measurement_dims)
            elif use_fafenet_measurements:
                measurements, classes = fm.get_fafe_measurements(frame_idx, measurement_dims)
            else:
                measurements, classes = kitti.get_measurements(frame_idx, measurement_dims, classes_to_track=config.classes_to_track)
           
            _pmbm.run(measurements, classes, imud[frame_idx], frame_idx, verbose=True, verbose_time=False)
                       
            toc= time.time()-tic
            toc_list.append(toc)

            L.log_data(_pmbm, frame_id=frame_idx, total_time=toc, measurements=measurements, true_states=kitti.get_bev_states(frame_idx, classes_to_track=config.classes_to_track), verbose=False)
        
        data = logger.load_logging_data(filename=L.filename)
        gospa_sl = logger.calculate_GOSPA_score(data=data, gt_dims=2)
        mot_summary = logger.calculate_MOT(sequence_idx, root, data=data, classes_to_track=config.classes_to_track)
        predictions_gospa_scores, predictions_average_gospa = logger.prediction_stats(sequence_idx, config, kitti, data=data)
        
        L.log_stats(toc_list, gospa_sl, mot_summary, config, predictions_average_gospa)
        if show_stuff:
            plot_stuff.plot_time_score(data=data, score_list=gospa_sl, config_name=config.name+'_seq_'+str(sequence_idx).zfill(4)) 
            plot_stuff.plot_hypos_over_time(data=data, config_name=config.name+'_seq_'+str(sequence_idx).zfill(4))
            plot_stuff.plot_target_life(data=data, config_name=config.name+'_seq_'+str(sequence_idx).zfill(4))
        if plot_gospa_comparison:
            plot_stuff.compare_pmbm_fafe_gospas(pmbm_sl=gospa_sl, fafe_filename=fafe_gospa_filename, pmbm_pred_gospa=predictions_average_gospa)

        if save_plots:
            print('Saving {} tracking history plots...'.format(min(kitti.max_frame_idx, number_of_timesteps)))
            dir_name = config.name
            in_dir = '_'.join(('showroom/output_tracks', dir_name, time_str))
            if not os.path.exists(in_dir):
                os.mkdir(in_dir)
            path = os.path.join(in_dir, str(sequence_idx).zfill(4))
            for frame_idx in range(min(kitti.max_frame_idx, number_of_timesteps)):
                print('{},'.format(frame_idx),end='')
                plot_stuff.plot_tracking_history(path, sequence_idx=sequence_idx, data=data, kitti=kitti, final_frame_idx=frame_idx, disp='save', only_alive=True, show_cov=True, show_predictions=config.show_predictions,
                                                config_name=config.name, car_van_flag=track_only_cars_vans, fafe=False, num_conseq_frames=None)
            print('Saved successfully!')
            print('Making movie in path: {}]'.format(path))
            plot_stuff.make_movie_from_images(path)
            print('{}.mov complete'.format(path))
            

df, avg_df = plot_stuff.sequence_analysis(sortby='CfgName')
if avg_df is not None: print(avg_df.to_string())
df

# Save as latex tables
df2 = df.drop(columns=['Filter', 'PoissonModel', 'MotionModel', 'MostlyLost', '#Fragmentations', 'PredGOSPA'])
file = open("showroom/stats_table.txt", "w")

for seq in range(0, max(df['SeqId'].values) + 1):
    df3 = df2.loc[df2['SeqId'] == seq]
    df3 = df3.sort_values(by ='CfgName', ascending=True)
    _str = '\n\\begin{table}[] \n\centering'
    file.write(_str)
    file.write(df3.to_latex(index=False)) 
    _str = ' \caption{Results for sequence ' + str(seq) + '} \n \label{tab:avg-pmbm-df}\n\end{table}\n'
    file.write(_str)
file.close()

if avg_df is not None:
    file = open("showroom/average_stats_table.txt", "w")
    file.write(avg_df.to_latex(index=True)) 
    file.close()

if True:
    from utils import plot_stuff
    stats_path = '/home/zhanghm/mot/conventional-MOT/showroom/all_sequences/cv-ca-bc-mx/logs' + '/stats'
    df, avg_df = plot_stuff.sequence_analysis(filenames_prefix=stats_path, sortby='CfgName')    