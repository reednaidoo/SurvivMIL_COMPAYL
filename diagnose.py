from __future__ import print_function
import argparse
import os
import pdb
import numpy as np
import pandas as pd
import yaml
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap
from wsi_core.batch_process_utils import initialize_df
import torch
import torch.nn as nn

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params


def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict



def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params
def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GraphCAM')
    parser.add_argument('--save_exp_code', type=str, default='test')
    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--path_file', type=str, default='tcga_lung_files/test_0.txt', help='')
    parser.add_argument('--path_WSI', type=str, default='', help='')
    parser.add_argument('--attns_folder', type=str, default= '/home/rnaidoo_l/Documents/Code/SMPeds_outcome/patient_attentions', help='Where are the attentions and coords stored')
    parser.add_argument('--config_file', type=str, default='heatmap_config.yaml', help='')
    parser.add_argument('--testdata', type=str, default='/home/rnaidoo_l/Documents/Code/SMPeds_outcome/datasets/Outcome/SMPeds_Outcome_fold_0.csv', help='')



    args = parser.parse_args()
    vis_folder= args.attns_folder
    folddat = pd.read_csv(args.testdata)


    config_path = 'heatmap_config.yaml'
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n' + key)
            for value_key, value_value in value.items():
                print(value_key + " : " + str(value_value))
        else:
            print('\n' + key + " : " + str(value))



    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])




    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1],
                                                                                  patch_args.overlap,
                                                                                  step_size[0], step_size[1]))
    preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False,
                      'keep_ids': 'none', 'exclude_ids': 'none'}
    def_filter_params = {'a_t': 50.0, 'a_h': 8.0, 'max_n_holes': 10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]

    # slides = sorted(os.listdir(data_args.data_dir))
    
    slides = []
    patref = folddat[folddat['Splits'] == 'test']['0'].to_list()
    for i in patref:
        slides.append(i.split('/')[-1].replace('.pt', '.svs'))

    slides = [slide for slide in slides if data_args.slide_ext in slide]
    df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                           use_heatmap_args=True)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    os.makedirs(exp_args.raw_save_dir, exist_ok=True)


    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size,
                         'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                         'use_center_shift': heatmap_args.use_center_shift}
    
    attentions_scores = []


    # for i in os.listdir(data_args.data_dir):
    for i in slides:
        slide_name = i 
        if data_args.slide_ext not in slide_name:
            slide_name += data_args.slide_ext
        print('\nprocessing: ', slide_name)

        slide_id = slide_name.replace(data_args.slide_ext, '')

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        if os.path.exists(os.path.join(r_slide_save_dir, '{}_scores.png'.format(slide_id))):
            continue

        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)


        seg_params = def_seg_params.copy()

        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))

        print('Initializing WSI object')

        
        wsi_object = initialize_wsi(slide_path, seg_params=seg_params,
                                    filter_params=filter_params)

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

        vis_patch_size = tuple(
            (np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        vis_params['line_thickness'] = 250
        mask = wsi_object.visWSI(**vis_params, number_contours=False,annot_display=True)
        mask.save(mask_path)



        patient_ref = slide_id.replace('.', '_')
        patient_ref = patient_ref.replace(' ', '_')   

        coords_path = os.path.join(vis_folder, '{}/coords/coords.pt'.format(patient_ref))
        attention_scores_path = os.path.join(vis_folder, '{}/attentions/attention_scores.pt'.format(patient_ref))
        risk_scores_path = os.path.join(vis_folder, '{}/risk_score/risk_score.pt'.format(patient_ref))
        risk_level_path = os.path.join(vis_folder, '{}/risk_level/risk_level.pt'.format(patient_ref))

        if not os.path.exists(coords_path) or not os.path.exists(attention_scores_path):
            print(f"Skipping {slide_id} because coords or attention_scores file does not exist.")
            continue


        coords_tensor = torch.load(coords_path)
        coords_array = coords_tensor.numpy().reshape(-1, 2)  # Reshape to ensure we have pairs
        coords_list = [(int(x), int(y)) for x, y in coords_array]
        
        attention_scores = torch.load(attention_scores_path)

        
        
        coords = np.array(coords_list)
        scores = np.array(attention_scores)


        if data_args.process_list is not None:
            process_stack.to_csv('/home/admin_ofourkioti/PycharmProjects/Histo_tree_data/TCGA_LUNG/heatmaps/{}.csv'.format(data_args.process_list.replace('.csv', '')),
                                 index=False)
        else:
            process_stack.to_csv('/home/rnaidoo_l/Documents/Code/SMPeds_outcome/patient_attentions/{}.csv'.format(exp_args.save_exp_code), index=False)


        wsi_kwargs = {'patch_size': patch_size, 'step_size': step_size,
                     'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                    'use_center_shift': heatmap_args.use_center_shift}

        heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)

        

        # Constructing the heatmap:

        heatmap_1 = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap,
                                  alpha=heatmap_args.alpha,
                                  use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
                                  thresh=-1, patch_size=vis_patch_size, convert_to_percentiles= True)

        heatmap_1.save(os.path.join(r_slide_save_dir, '{}_att.tiff'.format(slide_id)), format='TIFF')



        # Saving the top patches: 

        print(f'Sampling top patches for {slide_id}')

        patch_dir = os.path.join(r_slide_save_dir, 'patches')
        os.makedirs(patch_dir, exist_ok=True)

        top_k = 30  
        sorted_indices = np.argsort(-scores)  
        top_indices = sorted_indices[:top_k]  
        for idx in top_indices:
            s_coord = coords[idx]
            s_score = scores[idx]
            print(f'coord: {s_coord} score: {s_score:.3f}')
            patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
            patch.save(os.path.join(patch_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))


        # Saving the predictions of this patient:

        risk_scores = torch.load(risk_scores_path)
        risk_level = torch.load(risk_level_path)

        # Convert risk level to string
        risk_level_str = str(risk_level)

        print('RISK LEVEL:', risk_level_str)

        # Create a list to store the information
        info_list = []

        # Iterate over the scores and levels
        for score in risk_scores:
            info = f"Score: {score:.3f}, Risk Diagnosis: {risk_level_str}"
            info_list.append(info)

        # Save the information to a text file
        output_file = os.path.join(r_slide_save_dir, 'Patient_Diagnosis.txt')
        with open(output_file, 'w') as f:
            f.write('\n'.join(info_list))

        print(f"Information saved to {output_file}")