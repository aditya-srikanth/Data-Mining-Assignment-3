import numpy as np 
import os 
from threading import Thread
from multiprocessing import Pool,cpu_count
from pprint import pprint

from preprocess_dataset import process_dataset

class DBSCAN:
    def __init__(self,file_path,file_name,eps,minpts):
        self.core_points = []
        self.core_point_labels = []
        self.core_points_index = []
        self.n_threads = 5
        self.eps = eps
        self.minpts = minpts
        self.features,self.labels = process_dataset(file_path,file_name)
        # self.features,self.labels = self.features[:10000,:],self.labels[:10000]

    def set_thread_count(self,n_threads):
        self.n_threads = n_threads

    def find_core_points(self,indices):
        start_index = indices[0]
        end_index = indices[1]
        sample = self.features[start_index:end_index+1,:]
        print('entered thread start index: ',start_index,' end index: ',end_index,' sample shape ',sample.shape)
        distances = []
        core_points = []
        for point_index in range(sample.shape[0]):
            if point_index % 10 == 0:
                print('pointbreak at ', point_index + start_index)
            point = sample[point_index,:]
            distances = np.sqrt(np.sum((self.features - point)**2,axis=1))
            # print(np.argwhere(distances <= self.eps).shape[0])
            # print('distances: ',distances)
            if np.argwhere(distances <= self.eps).shape[0] >= self.minpts:
                core_points.append(start_index + point_index)
             
        return core_points
    
    def find_border_points(self,indices):
        start_index = indices[0]
        end_index = indices[1]
        print('entered thread',start_index,end_index)
        distances = []
        
        sample = self.features[start_index:end_index+1,:]
        
        border_points = []

        for point_index in range(sample.shape[0]):
            # print('pointbreak ')
            point = sample[point_index,:]
            distances = np.sqrt(np.sum((self.features - point)**2,axis=1))
            # print(np.argwhere(distances <= self.eps).shape[0])
            # print('distances: ',distances)
            if np.intersect1d(np.argwhere(distances <= self.eps),self.core_points_index):
                border_points.append(start_index + point_index)

        return border_points

    def fit(self):
        # get the number of training instances
        N = self.features.shape[0]
        # get the size of each input that the thread will take
        size = (N)//self.n_threads
        print('number of processors available: ',cpu_count())
        # create the thread pool
        thread_pool_for_core_points = Pool(processes = self.n_threads)
        # thread_pool_for_border_points = Pool(processes = self.n_threads)

        indices = []
        for start_index in range(0,N,size):
            if start_index + size < N:
                end_index = start_index + size - 1
                print('processsing: ',start_index,end_index)
                indices.append((start_index,end_index))
            else:
                end_index = start_index + (N - start_index) - 1
                print('final iteration: ',start_index,end_index)
                indices.append((start_index,end_index))
        
        results = thread_pool_for_core_points.map(self.find_core_points,indices)

        thread_pool_for_core_points.close()
        thread_pool_for_core_points.join()
        
        

        for result in results:
            for index in result:
                self.core_points_index.append(index)
        
        self.core_points = self.features[index,:]
        self.core_point_labels = self.labels[index,:]
        
        print(self.core_points.shape)

        # np.delete(self.features,self.core_points_index,axis=0)
        # np.delete(self.labels,self.core_points_index,axis=0)


        # self.core_points = np.array(self.core_points)

if __name__ == "__main__":
    test = DBSCAN("creditcardfraud","creditcard.csv",10,200)
    test.fit()