import numpy as np 
import os 
from multiprocessing import Pool
from multiprocessing import cpu_count
from multiprocessing import get_context
import SharedArray as sa
import gc
import time as time
from sklearn.metrics import accuracy_score 

import visualization 
from preprocess_dataset import process_dataset


def find_core_points(indices):
        params = sa.attach('shm://params')
        eps = params[0]
        minpts = params[1]
        features = sa.attach("shm://features")
        start_index = indices[0]
        end_index = indices[1]
        sample = features[start_index:end_index+1,:]
        print('entered thread for core points, start index: ',start_index,' end index: ',end_index,' sample shape ',sample.shape)
        distances = []
        core_points = []
        # nearest_neighbours = {}
        for point_index in range(sample.shape[0]):
            if point_index % 1000 == 0:
                print(start_index,' collected garbage',' at index: ',point_index,' remaining indices: ',sample.shape[0] - point_index)
                gc.collect()
            point = sample[point_index,:]
            distances = np.sqrt(np.sum((features - point)**2,axis=1))
            candidates = np.argwhere(distances <= eps)
            if candidates.shape[0] > minpts:
                core_points.append(start_index + point_index)
                # nearest_neighbours[start_index+point_index] = set(list(candidates.flatten()))
        print('core point search complete, exiting: ',start_index,end_index,'\n\n')
        features = None 
        start_index = None 
        end_index = None 
        sample = None
        gc.collect()
        # return core_points,nearest_neighbours
        # return core_points,nearest_neighbours
        return (core_points,{})
 


def find_border_points(indices):
        params = sa.attach('shm://params')
        eps = params[0]
        minpts = params[1]
        features = sa.attach("shm://features")
        core_points_index = sa.attach("shm://core_points")
        start_index = indices[0]
        end_index = indices[1]
        print('finding border points entered thread',start_index,end_index)
        distances = []
        
        sample = features[start_index:end_index+1,:]
        
        border_points = []

        for point_index in range(sample.shape[0]):
            if point_index % 1000 == 0:
                gc.collect()
                print(start_index," collecting garbage, at index: ",point_index,' remaining indices: ',sample.shape[0] - point_index)
            point = sample[point_index,:]
            distances = np.sqrt(np.sum((features - point)**2,axis=1))
            candidates = np.argwhere(distances <= eps) + start_index
            # if it is not a core point and is in the vicinity of a core point
            if np.intersect1d(candidates,core_points_index).shape[0] >= 1 and not (start_index + point_index) in core_points_index:
                border_points.append(start_index + point_index)
            del distances
        print('exiting border finding: ',start_index,end_index,end_index - start_index + 1,'\n\n')
        
        return border_points

class DBSCAN:
    eps = 0
    minpts = 0
    def __init__(self,file_path,file_name):
        print('class',DBSCAN.eps,DBSCAN.minpts)
        self.core_points = []
        self.core_point_labels = []
        self.core_points_index = []
        self.border_points_index = []
        self.border_points = []
        self.border_point_labels = []
        self.noise_points = []
        # self.nearest_neighbours = {}      # use for small values, space complexity is O(n^2)
        self.n_threads = cpu_count()
        self.features = []
        self.labels = []
        self.features,self.labels = process_dataset(file_path,file_name)        # limit the size of the dataset
        size = 10000
        self.features,self.labels = self.features[:size,:],self.labels[:size]
        print('features: \n',self.features.shape)
        try:
            sa.delete("shm://features")
        except Exception as e:
            print('file does not exist')
        self.shared_memory = sa.create("shm://features",self.features.shape)

        # copy the array into the shared memory
        for row_index in range(self.features.shape[0]):
            for point_index in range(self.features.shape[1]):
                self.shared_memory[row_index,point_index] = self.features[row_index,point_index]
        self.clusters = []

    def set_thread_count(self,n_threads):
        self.n_threads = n_threads

    def fit(self):
        start_time = time.time()
        # get the number of training instances
        N = self.features.shape[0]
        self.n_threads = min(self.n_threads,N)
        # get the size of each input that the thread will take
        size = (N)//self.n_threads
        print('number of processors available: ',cpu_count())
        core_point_start_time = time.time()
        with get_context("spawn").Pool(processes=self.n_threads,maxtasksperchild=1) as thread_pool_for_core_points:
            # create the thread pool for core and border points
            core_indices = []
            for start_index in range(0,N,size):
                if start_index + size < N:
                    end_index = start_index + size - 1
                    core_indices.append((start_index,end_index))
                else:
                    end_index = start_index + (N - start_index) - 1
                    core_indices.append((start_index,end_index))
            
            core_results = thread_pool_for_core_points.map(find_core_points,core_indices)

            thread_pool_for_core_points.close()
            thread_pool_for_core_points.join()
        core_point_end_time = time.time()

        print('core points evaluated\n')

        for result in core_results:
            # self.nearest_neighbours.update(result[1]) # uncomment this for memory intensive dfs of clusters
            for index in result[0]:
                self.core_points_index.append(index)
        
        self.core_points = self.features[self.core_points_index,:]
        self.core_point_labels = list(self.labels[self.core_points_index,:])

        try:
            sa.delete('shm://core_points')
        except Exception as e:
            print('core points shared memory does not exist')
        core_points_array = sa.create('shm://core_points',len(self.core_points_index))
        for point_index in range(len(self.core_points_index)):
            core_points_array[point_index] = self.core_points_index[point_index]

        border_point_start_time = time.time()
        with get_context("spawn").Pool(processes=self.n_threads,maxtasksperchild=1) as thread_pool_for_border_points:
            border_indices = []
            for start_index in range(0,N,size):
                if start_index + size < N:
                    end_index = start_index + size - 1
                    border_indices.append((start_index,end_index))
                else:
                    end_index = start_index + (N - start_index) - 1
                    border_indices.append((start_index,end_index))
            
            border_results = thread_pool_for_border_points.map(find_border_points,border_indices)

            thread_pool_for_border_points.close()
            thread_pool_for_border_points.join()

        border_point_end_time = time.time()
        for result in border_results:
            for index in result:
                self.border_points_index.append(index)
        
        self.border_points = self.features[self.border_points_index,:]
        self.border_point_labels = list(self.labels[self.border_points_index,:])
        
        self.core_points = np.array(self.core_points)
        self.border_points = np.array(self.border_points)

        ##### VERY MEMORY INTENSIVE OPERATION, DOES DFS TO EVALUATE CLUSTERS ########
        # cluster_points = self.nearest_neighbours
        # visited = set()
        # for key,value in self.nearest_neighbours.items():
        #     visited.add(key)
        #     for temp in value:
        #         visited.add(temp)

        # visited = {item:False for item in list(visited)}
        # clusters = []
        # # generate clusters
        # for point in cluster_points.keys():
        #     if visited[point]:
        #         continue
        #     queue = []
        #     cluster = set()
        #     queue.append(point)
        #     while len(queue) > 0:
        #         node = queue.pop()
        #         if not visited[node]:
        #             cluster.add(node)
        #             queue += list(self.nearest_neighbours[point])
        #             # print('node: ',node)
        #             visited[node] = True
        #     if len(cluster) > 0:
        #         clusters.append(cluster)
        
        # print('number of clusters: ',len(clusters))
        # print('clusters are:')
        # for cluster in clusters:
        #     print(cluster)
        # self.clusters = clusters

        cluster_points = set(self.border_points_index + self.core_points_index)
        all_points = set([i for i in range(self.features.shape[0])])
        self.noise_index = all_points.difference(cluster_points)
        self.noise_points = [self.features[i,:] for i in self.noise_index]
        print("stats:\ntime for core: ",core_point_end_time - core_point_start_time," border point time: "\
            ,border_point_end_time - border_point_start_time, "total time: ",time.time() - start_time)
        print("deleting shared memory")
        sa.delete("shm://features")
        sa.delete("shm://core_points")
        return self.noise_points,core_point_end_time - core_point_start_time,border_point_end_time - border_point_start_time,time.time() - start_time
    
    
    def plot(self):
        params = sa.attach('shm://params')
        eps = params[0]
        minpts = params[1]
        visuals = visualization.Visualization()
        for point in self.noise_points:
            visuals.OUTLIERS.append(visuals.dimension_reduction(point))
        for point in self.core_points:
            visuals.NON_OUTLIERS.append(visuals.dimension_reduction(point))
        for point in self.border_points:
            visuals.NON_OUTLIERS.append(visuals.dimension_reduction(point))
        visuals.outlier_plot_numpy(save_path="./dbscan_plots/eps_"+str(eps)+"_minpts_"+str(minpts))           

    def print_accuracy_score(self,redirect = None):
        accuracy = 0
        for point in self.noise_index:
            if self.labels[point] == 1:
                accuracy += 1
        cluster_points = self.core_points_index + self.border_points_index
        for point in cluster_points:
            if self.labels[point] == 0:
                accuracy += 1
        if redirect == None:
            print('accuracy: \n',accuracy / self.features.shape[0] * 100,"%")
        else:
            print('accuracy: \n',accuracy / self.features.shape[0] * 100,"%")
            print('accuracy: \n',accuracy / self.features.shape[0] * 100,"%",file=f)
            
if __name__ == "__main__":
    # eps = int(input('enter eps\n'))
    # minpts = int(input('enter minpts\n'))
    try:
        sa.delete("shm://params")
    except Exception as e:
        print('params to be created')
    params = sa.create("shm://params",(2,))
    with open('stats.txt','w') as f:
        print("stats: minpts, eps time for core points, time for border points, total time\n",file=f,flush=True)
        for eps in range(0,20,5):
            for minpts in range(0,20,5):
                params[0] = eps 
                params[1] = minpts
                print('params',params)
                test = DBSCAN("creditcardfraud","creditcard.csv")
                outliers,core_time,border_time,total_time = test.fit()
                print('number of outliers are: ',len(outliers))
                print(str(minpts) +',' + str(eps) + ',' + str(core_time)+','+str(border_time)+','+str(total_time)+'\n',file=f,flush=True)
                print('accuracy at minpts: ',minpts,' and eps = ',eps)
                test.print_accuracy_score(f)
                test.plot() 
    sa.delete('shm://params')