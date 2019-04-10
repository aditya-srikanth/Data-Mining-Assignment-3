import numpy as np 
import os
import pickle
import pandas as pd

def process_dataset(file_path,file_name):
    if not os.path.isdir("processed_data"):
        with open(file_path+'/'+file_name,"rb") as f:
            data = pd.read_csv(f).values
        features = data[:,:-1]
        labels = np.reshape(data[:,-1],(data.shape[0],1))
        
        # feature scaling
        features = (features - np.mean(features,axis=0))/np.std(features,axis=0)
        
        os.mkdir("processed_data")

        with open("./processed_data/features.txt",'wb') as f:
            pickle.dump(features,f)
        with open("./processed_data/labels.txt",'wb') as f:
            pickle.dump(labels,f)

        
    else:
        with open("processed_data/features.txt",'rb') as f:
            features = pickle.load(f)
        with open("processed_data/labels.txt",'rb') as f:
            labels = pickle.load(f)
        
    return features,labels