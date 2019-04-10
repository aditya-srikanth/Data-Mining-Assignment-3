import numpy as np 
import os
import pickle
import pandas as pd

def process_dataset(file_path,file_name):
    if not os.path.isdir(file_name+"processed_data"):
        with open(file_path,"rb") as f:
            data = pd.read_csv(f).values
            features = data[:,:-1]
            labels = data[:,-1]

        
    else:
        with open(file_name+"processed_data/features.txt") as f:
            features = pickle.load(f)
        with open(file_name+"processed_data/labels.txt") as f:
            labels = pickle.load(f)
    
        return features,labels
process_dataset()