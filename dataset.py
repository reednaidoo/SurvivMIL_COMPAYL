import pandas as pd
import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import random
from pathlib import Path
import tifffile as tfl
from sklearn.utils import shuffle
import h5py



class histodata(Dataset):
    """
    Dataset class for the SMPeds dataset
    """

    def __init__(
        self,
        h5_path=None,
        csv_path=None,
        state=None,
        shuffle=False,
        one_vs_target='high',
        concat = False,
    ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.h5_path = h5_path
        self.csv_path = csv_path
        self.shuffle = shuffle
        self.one_vs_target = one_vs_target
        self.concat = concat

        self.slide_data = pd.read_csv(self.csv_path)


        # Split dataset based on 'Splits' column
        self.state = state
        self.pwn = np.unique(
            self.slide_data[
                (self.slide_data['Splits'] == self.state)
            ]['0']
            .reset_index(drop=True)
            .values 
        )
        # print("Setting up histo data")  




        # Shuffle the dataset if required
        if self.shuffle:
            random.shuffle(self.pwn)

    def __getitem__(self, idx):
        slide_id = self.pwn[idx]
        full_path = slide_id
        features = torch.load(full_path).double() 

        patient = full_path.split('/')[-1].split('.pt')[0]
        test_path = full_path.replace("pt_files", "h5_files")
        test_path = test_path.replace(".pt", ".h5")

        


        # Alternative loading used for testing
        with h5py.File(test_path, 'r') as f:
            test_features = f['features'][()]  
            test_coords = f['coords'][()]
            
            test_features = torch.tensor(test_features).double()
            test_coords = torch.tensor(test_coords).double()


        labels_csv = pd.read_csv(self.csv_path)

        # Defining the survival analysis response variables
        label = labels_csv.loc[labels_csv['0'] == full_path, 'label'].values
        censorship = 1 - labels_csv.loc[labels_csv['0'] == full_path, 'outcome'].values 
        time_to_event = labels_csv.loc[labels_csv['0'] == full_path, 'survival_days'].values


        if self.concat == False:
            # defining the electronic health records features
            mycn_label = labels_csv.loc[labels_csv['0'] == full_path, 'Mycn Status'].values
            age_label = labels_csv.loc[labels_csv['0'] == full_path, 'patient_age_at_biopsy_months'].values
            ehr = torch.tensor([mycn_label, age_label], dtype=torch.float32)
            # ----> shuffle within each split
            if self.shuffle:
                index = [x for x in range(features.shape[0])]
                random.shuffle(index)
                features = features[index]

            if self.state == 'test':
                return (test_features, ehr, test_coords), (label, time_to_event, censorship), (patient)
            elif self.state == 'train': 
                return (features, ehr), (label, time_to_event, censorship)

        
        elif self.concat == True: 
            mycn_label = labels_csv.loc[labels_csv['0'] == full_path, 'Mycn Status'].values
            age_label = labels_csv.loc[labels_csv['0'] == full_path, 'patient_age_at_biopsy_months'].values
            mycn_label = torch.tensor(mycn_label).squeeze()
            age_label = torch.tensor(age_label).squeeze()
            ehr = torch.tensor([mycn_label, age_label], dtype=torch.float32) # returned in the dataloader for model/batch purposes, but not used
            
            # Create the mycn and age tensors
            mycn_tensor = torch.tensor([mycn_label] * features.shape[1], dtype=torch.float32).unsqueeze(0)
            age_tensor = torch.tensor([age_label] * features.shape[1], dtype=torch.float32).unsqueeze(0)

            # Concatenate features with mycn_tensor and age_tensor along the first dimension
            combined_features = torch.cat((features, mycn_tensor, age_tensor), dim=0)
            # ----> shuffle within each split
            if self.shuffle:
                index = torch.randperm(combined_features.shape[0])
                combined_features = combined_features[index]

            # still returning ehr as a separate tensor, but not considered in predictions if multimodal = wsi
            # return (combined_features,ehr), (label, time_to_event, censorship)
            if self.state == 'test':
                return (test_features, ehr, test_coords), (label, time_to_event, censorship) 
            elif self.state == 'train': 
                return (features, ehr), (label, time_to_event, censorship)


    def __len__(self):
        return len(self.pwn)