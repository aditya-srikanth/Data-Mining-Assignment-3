import numpy as np 
import os 
from multiprocessing import Pool
from multiprocessing import cpu_count
from multiprocessing import get_context
import SharedArray as sa
from pprint import pprint
import gc

import visualization 
from preprocess_dataset import process_dataset

def find_core_points(indices):
        features = sa.attach("shm://features")
        print('features: \n',features.shape)
        minpts = 10
        eps = 1
        start_index = indices[0]
        end_index = indices[1]
        sample = features[start_index:end_index+1,:]
        print('entered thread start index: ',start_index,' end index: ',end_index,' sample shape ',sample.shape)
        distances = []
        core_points = []
        nearest_neighbours = {}
        for point_index in range(sample.shape[0]):
            if point_index % 1000 == 0:
                print('collected garbage')
                gc.collect()
            point = sample[point_index,:]
            distances = np.sqrt(np.sum((features - point)**2,axis=1))
            candidates = np.argwhere(distances <= eps)
            if candidates.shape[0] > minpts:
                core_points.append(start_index + point_index)
                nearest_neighbours[start_index+point_index] = set(list(candidates.flatten()))
        print('exiting: ',start_index,end_index,'\n\n')
        sa.delete("shm://features")
        return core_points,nearest_neighbours


class DBSCAN:
    def __init__(self,file_path,file_name,eps=1,minpts=1):
        self.core_points = []
        self.core_point_labels = []
        self.core_points_index = []
        self.border_points_index = []
        self.border_points = []
        self.border_point_labels = []
        self.noise_points = []
        self.nearest_neighbours = {}
        self.n_threads = cpu_count()
        self.eps = eps
        self.minpts = minpts
        self.features,self.labels = process_dataset(file_path,file_name)
        self.features,self.labels = self.features[:200000,:],self.labels[:200000]
        sa.delete("shm://features") ## TODO send this down
        self.shared_memory = sa.create("shm://features",self.features.shape)

        # copy the array into the shared memory
        for row_index in range(self.features.shape[0]):
            for point_index in range(self.features.shape[1]):
                self.shared_memory[row_index,point_index] = self.features[row_index,point_index]
        self.clusters = []

    def set_thread_count(self,n_threads):
        self.n_threads = n_threads

    # def find_core_points(self,indices,eps,minpts):
    #     features = sa.attach("shm://features")
    #     print('features: \n',features.shape)

    #     start_index = indices[0]
    #     end_index = indices[1]
    #     sample = features[start_index:end_index+1,:]
    #     print('entered thread start index: ',start_index,' end index: ',end_index,' sample shape ',sample.shape)
    #     distances = []
    #     core_points = []
    #     nearest_neighbours = {}
    #     for point_index in range(sample.shape[0]):
    #         if point_index % 100 == 0:
    #             print('index: ',point_index)
    #             gc.collect()
    #             print('collected garbage')
    #         point = sample[point_index,:]
    #         distances = np.sqrt(np.sum((features - point)**2,axis=1))
    #         candidates = np.argwhere(distances <= eps)
    #         if candidates.shape[0] > minpts:
    #             core_points.append(start_index + point_index)
    #             nearest_neighbours[start_index+point_index] = set(list(candidates.flatten()))
    #     print('exiting: ',start_index,end_index,'\n\n')
    #     return core_points,nearest_neighbours
    
    def find_border_points(self,indices):
        features = sa.attach("shm://features")
        start_index = indices[0]
        end_index = indices[1]
        print('entered thread',start_index,end_index)
        distances = []
        
        sample = features[start_index:end_index+1,:]
        
        border_points = []

        for point_index in range(sample.shape[0]):
            if point_index % 1000 == 0:
                gc.collect()
                print("collecting")
            point = sample[point_index,:]
            distances = np.sqrt(np.sum((features - point)**2,axis=1))

            candidates = np.argwhere(distances <= self.eps) + start_index

            # if it is not a core point and is in the vicinity of a core point
            if np.intersect1d(candidates,self.core_points_index).shape[0] >= 1 and not (start_index + point_index) in self.core_points_index:
                border_points.append(start_index + point_index)
        print('exiting border finding: ',start_index,end_index,end_index - start_index + 1,'\n\n')
        return border_points

    def fit(self):
        # get the number of training instances
        N = self.features.shape[0]
        # get the size of each input that the thread will take
        size = (N)//self.n_threads
        print('number of processors available: ',cpu_count())

        with get_context("spawn").Pool(processes=self.n_threads,maxtasksperchild=1) as thread_pool_for_core_points:
            # create the thread pool for core and border points
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
                    # core_indices.append((eps,minpts))
            
            core_results = thread_pool_for_core_points.map(find_core_points,core_indices)

            thread_pool_for_core_points.close()
            thread_pool_for_core_points.join()

        for result in core_results:
            self.nearest_neighbours.update(result[1])
            for index in result[0]:
                self.core_points_index.append(index)
        
        self.core_points = self.features[self.core_points_index,:]
        self.core_point_labels = list(self.labels[self.core_points_index,:])
        
        with get_context("spawn").Pool(processes=self.n_threads,maxtasksperchild=1) as thread_pool_for_border_points:
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
            if len(cluster) > 0:
                clusters.append(cluster)
        
        print('number of clusters: ',len(clusters))
        # print('clusters are:')
        # for cluster in clusters:
        #     print(cluster)
        self.clusters = clusters
        cluster_points = set(self.border_points_index + self.core_points_index)
        all_points = set([i for i in range(self.features.shape[0])])
        self.noise_index = all_points.difference(cluster_points)
        self.noise_points = [self.features[i,:] for i in self.noise_index]
        return self.noise_points
    
    
    def plot(self):
        visuals = visualization.Visualization()
        for point in self.noise_points:
            visuals.OUTLIERS.append(visuals.dimension_reduction(point))
        for point in self.core_points:
            visuals.NON_OUTLIERS.append(visuals.dimension_reduction(point))
        for point in self.border_points:
            visuals.NON_OUTLIERS.append(visuals.dimension_reduction(point)) 
        visuals.outlier_plot()           

if __name__ == "__main__":
    test = DBSCAN("creditcardfraud","creditcard.csv",eps=10,minpts=10)
    outliers = test.fit()
    print('outliers are: ',len(outliers))
    with open('noise.txt','w') as f:
        for noise_point in outliers:
            print(noise_point,file=f)
    # test.plot()