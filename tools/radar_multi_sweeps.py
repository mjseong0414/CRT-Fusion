import mmcv
import pickle
import numpy as np
import torch
from nuscenes.utils.data_classes import RadarPointCloud

point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3]
invalid_states=list(range(18))
dynprop_states=list(range(8))
ambig_states=list(range(5))


def _load_points(pts_filename):
        raw_points = RadarPointCloud.from_file(
            pts_filename,
            invalid_states = invalid_states,
            dynprop_states = dynprop_states,
            ambig_states = ambig_states).points.T
        # x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
        raw_points = raw_points.astype(np.float32)
        return raw_points
        
    

def radar_multi_sweeps(info_path):
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
    
    new_infos = dict()
    new_infos['infos'] = list()
    new_infos['metadata'] = infos['metadata']
    num_infos = len(infos['infos'])
    for i, info in enumerate(infos['infos']):
        print(f'{((i+1)/num_infos) * 100}')
        num_sweeps = len(info['radar_sweeps'])
        sweep_indexes = {}
        sweep_points_list = []
        for idx in range(num_sweeps):
            sweep_pts_file_name = info['radar_sweeps'][idx]
            if sweep_pts_file_name.split('/')[3].startswith('samples'):
                sweep_index = 10
            elif sweep_pts_file_name.split('/')[3].startswith('sweeps'):
                unique_part = sweep_pts_file_name.split('/')[4]
                sweep_index = sweep_indexes.get(unique_part, 0)
                sweep_indexes[unique_part] = sweep_index + 1
            
            points_sweep = _load_points(sweep_pts_file_name)
            sample_or_sweep = sweep_pts_file_name.split('/')[3]
            points_sweep[:, :2] += points_sweep[:, 8:10] * info['radar_sweeps_time_gap'][idx] ################ radial velocity
            
            ### load sensor2lidar_rot/tran ###
            R = info['radar_sweeps_r2l_rot'][idx]
            T = info['radar_sweeps_r2l_trans'][idx]
            
            # current lidar coordinate radar points
            points_sweep[:, :3] = points_sweep[:, :3] @ R
            points_sweep[:, 8:10] = points_sweep[:, 8:10] @ R[:2, :2]
            points_sweep[:, :3] += T
            
            sweep_index_np = np.ones((points_sweep.shape[0], 1)) * sweep_index
            points_sweep = np.concatenate((points_sweep, sweep_index_np), axis=1)
            sweep_points_list.append(points_sweep)
        
        points = np.concatenate(sweep_points_list)
        pts_mask = (points[:,0]<=51.2) & (points[:,0]>=-51.2) & (points[:,1]<=51.2) & (points[:,1]>=-51.2)
        points = points[pts_mask]
        
        radar_ms_filename = info['radar_sweeps'][0].split('/')[-1].replace('RADAR_FRONT', 'RADAR').split('.')[0] + '.npy'
        np.save(f'./data/nuscenes/crtfusion_radar_ms/{radar_ms_filename}', points)
        info['radar_ms_filename'] = 'crtfusion_radar_ms/' + radar_ms_filename
        
        new_infos['infos'].append(info)
    
    return new_infos
        
        

if __name__=='__main__':
    train_info_path = './data/nuscenes/nuscenes_infos_crtfusion_train.pkl'
    val_info_path = './data/nuscenes/nuscenes_infos_crtfusion_val.pkl'
    # test_info_path = './data/nuscenes/nuscenes_infos_crtfusion_test.pkl'
    
    train_infos = radar_multi_sweeps(train_info_path)
    val_infos = radar_multi_sweeps(val_info_path)
    # test_infos = radar_multi_sweeps(test_info_path)
    
    mmcv.dump(train_infos, './data/nuscenes/nuscenes_infos_crtfusion_train.pkl')
    mmcv.dump(val_infos, './data/nuscenes/nuscenes_infos_crtfusion_val.pkl')
    # mmcv.dump(test_infos, './data/nuscenes/nuscenes_infos_crtfusion_test.pkl')
    