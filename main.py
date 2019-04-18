import numpy as np 
import os 
from threading import Thread
from multiprocessing import Pool,cpu_count
from pprint import pprint

from preprocess_dataset import process_dataset

class DBSCAN:
    def __init__(self,file_path,file_name,eps=1,minpts=1):
        self.core_points = []
        self.core_point_labels = []
        self.core_points_index = []
        self.border_points_index = []
        self.border_points = []
        self.border_point_labels = []
        self.nearest_neighbours = {}
        self.n_threads = 5
        self.eps = eps
        self.minpts = minpts
        self.features,self.labels = process_dataset(file_path,file_name)
        self.features,self.labels = self.features[:2000,:],self.labels[:2000]
        self.clusters = []

    def set_thread_count(self,n_threads):
        self.n_threads = n_threads

    def find_core_points(self,indices):
        start_index = indices[0]
        end_index = indices[1]
        sample = self.features[start_index:end_index+1,:]
        print('entered thread start index: ',start_index,' end index: ',end_index,' sample shape ',sample.shape)
        distances = []
        core_points = []
        nearest_neighbours = {}
        for point_index in range(sample.shape[0]):
            point = sample[point_index,:]
            distances = np.sqrt(np.sum((self.features - point)**2,axis=1))
            candidates = np.argwhere(distances <= self.eps)
            if candidates.shape[0] > self.minpts:
                core_points.append(start_index + point_index)
                nearest_neighbours[start_index+point_index] = set(list(candidates.flatten()))
             
        return core_points,nearest_neighbours
    
    def find_border_points(self,indices):
        start_index = indices[0]
        end_index = indices[1]
        print('entered thread',start_index,end_index)
        distances = []
        
        sample = self.features[start_index:end_index+1,:]
        
        border_points = []

        for point_index in range(sample.shape[0]):
            point = sample[point_index,:]
            distances = np.sqrt(np.sum((self.features - point)**2,axis=1))

            candidates = np.argwhere(distances <= self.eps) + start_index

            # if it is not a core point and is in the vicinity of a core point
            if np.intersect1d(candidates,self.core_points_index).shape[0] >= 1 and not (start_index + point_index) in self.core_points_index:
                border_points.append(start_index + point_index)

        return border_points

    def fit(self):
        # get the number of training instances
        N = self.features.shape[0]
        # get the size of each input that the thread will take
        size = (N)//self.n_threads
        print('number of processors available: ',cpu_count())

        # create the thread pool for core and border points
        thread_pool_for_core_points = Pool(processes = self.n_threads)
        thread_pool_for_border_points = Pool(processes = self.n_threads)

        core_indices = []
        for start_index in range(0,N,size):
            if start_index + size < N:
                end_index = start_index + size - 1
                # print('processsing: ',start_index,end_index)
                core_indices.append((start_index,end_index))
            else:
                end_index = start_index + (N - start_index) - 1
                # print('final iteration: ',start_index,end_index)
                core_indices.append((start_index,end_index))
        
        core_results = thread_pool_for_core_points.map(self.find_core_points,core_indices)

        thread_pool_for_core_points.close()
        thread_pool_for_core_points.join()

        for result in core_results:
            self.nearest_neighbours.update(result[1])
            for index in result[0]:
                self.core_points_index.append(index)
        
        self.core_points = self.features[self.core_points_index,:]
        self.core_point_labels = list(self.labels[self.core_points_index,:])
        
        border_indices = []
        for start_index in range(0,N,size):
            if start_index + size < N:
                end_index = start_index + size - 1
                border_indices.append((start_index,end_index))
            else:
                end_index = start_index + (N - start_index) - 1
                border_indices.append((start_index,end_index))
        
        border_results = thread_pool_for_border_points.map(self.find_border_points,border_indices)

        thread_pool_for_border_points.close()
        thread_pool_for_border_points.join()


        for result in border_results:
            for index in result:
                self.border_points_index.append(index)
        

        self.border_points = self.features[self.border_points_index,:]
        self.border_point_labels = list(self.labels[self.border_points_index,:])
        
        self.core_points = np.array(self.core_points)
        self.border_points = np.array(self.border_points)

        cluster_points = self.nearest_neighbours
        # print(self.nearest_neighbours)
        visited = set()
        for key,value in self.nearest_neighbours.items():
            visited.add(key)
            for temp in value:
                visited.add(temp)

        visited = {item:False for item in list(visited)}
        # print(visited,'al;skdfj')
        clusters = []
        # print(clusters,)
        # generate clusters
        for point in cluster_points.keys():
            if visited[point]:
                continue
            queue = []
            cluster = set()
            queue.append(point)
            while len(queue) > 0:
                node = queue.pop()
                if not visited[node]:
                    cluster.add(node)
                    queue += list(self.nearest_neighbours[point])
                    # print('node: ',node)
                    visited[node] = True
            print('cluster: ',cluster)
            if len(cluster) > 0:
                clusters.append(cluster)
        
        print(len(clusters))
        print(clusters)
        self.clusters = clusters

if __name__ == "__main__":
    test = DBSCAN("creditcardfraud","creditcard.csv",eps=2,minpts=20)
    test.fit()