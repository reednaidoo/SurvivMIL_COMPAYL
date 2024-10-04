import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

from file_utils import save_hdf5
from dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from encoder_file import get_encoder


import os
import logging

import timm
import torch
import torch.nn as nn
from torchvision import transforms
from huggingface_hub import login, hf_hub_download
#from .models.resnet50_trunc import resnet50_trunc_imagenet

def get_norm_constants(which_img_norm: str = 'imagenet'):
    constants_zoo = {
        'imagenet': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
        'ctranspath': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
        'openai_clip':{'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711)},
        'uniform': {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
    }
    constants = constants_zoo[which_img_norm]
    return constants.get('mean'), constants.get('std')


def get_eval_transforms(
        which_img_norm: str = 'imagenet', 
        img_resize: int = 224, 
        center_crop: bool = False
):
    r"""
    Gets the image transformation for normalizing images before feature extraction.
    Args:
        - which_img_norm (str): transformation type
    Return:
        - eval_transform (torchvision.Transform): PyTorch transformation function for images.
    """
    
    eval_transform = []

    if img_resize > 0:
        eval_transform.append(transforms.Resize(img_resize))

        if center_crop:
            eval_transform.append(transforms.CenterCrop(img_resize))

    mean, std = get_norm_constants(which_img_norm)

    eval_transform.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose(eval_transform)
    return eval_transform


def get_encoder(
        enc_name='vit_large_patch16_224.dinov2.uni_mass100k', 
        checkpoint='pytorch_model.bin',
        which_img_norm='imagenet', 
        img_resize=224, 
        center_crop=True, 
        test_batch=0, 
        device=None,
        #assets_dir=os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), '../../assets/ckpts'),
        assets_dir = '/home/rnaidoo_l/assets/ckpts', 
        kwargs={},
):
    r"""
    Get image encoder with pretrained weights and the their normalization.

    Args:
        - enc_name (str): Name of the encoder (finds folder that is named <enc_name>, which the model checkpoint assumed to be in this folder)
        - checkpoint (str): Name of the checkpoint file (including extension)
        - assets_dir (str): Path to where checkpoints are saved.

    Return:
        - model (torch.nn): PyTorch model used as image encoder.
        - eval_transforms (torchvision.transforms): PyTorch transformation function for images.
    """

    enc_name_presets = {
        'resnet50_trunc': ('resnet50.supervised.trunc_in1k_transfer', None, 'imagenet'),
        'uni': ('vit_large_patch16_224.dinov2.uni_mass100k', 'pytorch_model.bin', 'imagenet'),
    }
    
    if enc_name in enc_name_presets.keys():
        enc_name, checkpoint, which_img_norm = enc_name_presets[enc_name]
    
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ### ResNet50 Truncated Encoder, Dim=1024, Pretrained on ImageNet
    if enc_name == 'resnet50trunc.supervised.in1k_transfer':
        model = resnet50_trunc_imagenet()
        assert which_img_norm == 'imagenet'

    ### UNI
    elif enc_name == 'vit_large_patch16_224.dinov2.uni_mass100k':
        ckpt_dir = os.path.join(assets_dir, enc_name)
        ckpt_path = os.path.join(assets_dir, enc_name, checkpoint)
        print('ckpt_path:', ckpt_path)
        assert which_img_norm == 'imagenet'
        if not os.path.isfile(ckpt_path):
            from huggingface_hub import login, hf_hub_download
            login() # login with your User Access Token, found at https://huggingface.co/settings/tokens
            os.makedirs(ckpt_dir, exist_ok=True)
            hf_hub_download('MahmoodLab/UNI', filename="pytorch_model.bin", local_dir=ckpt_dir, force_download=True)

        uni_kwargs = {
            'model_name': 'vit_large_patch16_224',
            'img_size': 224, 
            'patch_size': 16, 
            'init_values': 1e-5, 
            'num_classes': 0, 
            'dynamic_img_size': True
        }
        model = timm.create_model(**uni_kwargs)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    else:
        return None, None

    eval_transform = get_eval_transforms(
        which_img_norm=which_img_norm, 
        img_resize=img_resize, 
        center_crop=center_crop
    )

    logging.info(f'Missing Keys: {missing_keys}')
    logging.info(f'Unexpected Keys: {unexpected_keys}')
    logging.info(str(model))
    
    # Send to GPU + turning on eval
    model.eval()
    model.to(device)

    # Test Batch
    logging.info(f"Transform Type: {eval_transform}")
    if test_batch:
        imgs = torch.rand((2, 3, 224, 224), device=device)
        with torch.no_grad():
            features = model(imgs)
        logging.info(
            f'Test batch successful, feature dimension: {features.size(1)}')
        del imgs, features

    return model, eval_transform




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default= os.getcwd() + '/raw_features')
parser.add_argument('--data_slide_dir', type=str, default='/neuroblastoma_batch1')
parser.add_argument('--slide_ext', type=str, default= '.h5') # changed to h5 because of the nature of the reffile 
parser.add_argument('--csv_path', type=str, default='reffile.csv')
parser.add_argument('--feat_dir', type=str, default='/feats')
parser.add_argument('--model_name', type=str, default='uni_v1', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	#model, img_transforms = get_encoder(args.model_name, img_resize=args.target_patch_size)
	model, img_transforms = get_encoder(img_resize=args.target_patch_size)		
      
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id + '.svs')
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
                                     wsi=wsi, 
									 img_transforms=img_transforms)

		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
		output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))