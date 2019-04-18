import numpy as np
import pandas as pd
import matplotlib as plt
import os
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from time import time

dataset=pd.read_csv('/home/user/Projects/Prateek_Projects/creditcardfraud/creditcard.csv')
#print(dataset.head())
data=dataset.values
data=data[:5000,:]
X=data[:,:30]
y=data[:,30]
#print(X[0])


scaler = StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled=scaler.transform(X_scaled)

eps=80
minPts=100

#Calculating distanc between two points
def distances(point1, point2):
	return np.sum(np.square(point1-point2))

#for i in range(10,30):
#	print(distances(X_scaled[i], X_scaled[88]))


def label(points):
	p=1
	'''
	Each point is labelled as a core point, border point or a Noise point

	Pointer: A dictionary which contains the keys as index and the value as a tuple (corresponding point, points in that neighbourhood)
	Pointer[index]=(point, neighbourhood)

	core_pt: A dictionary which takes in an index key to gice the corresponding value, a tuple of the corresponding point and whether it has been visited or not
	core_pt[index]=[point, visited, index]

	border_pt: A dictionary which takes in an index key to gice the corresponding value, a tuple of the corresponding point and whether it has been visited or not
	border_pt[index]=(point, visited)

	noise_pt: A dictionary which takes in an index key to gice the corresponding value, a tuple of the corresponding point and whether it has been visited or not
	noise_pt[index]=(point, visited)

	neighbourhood: A list of tuples of the corresponding neighbourhood points, their indexes and whether they have been visited or not
	neighbourhood=[(neighbourhood point, index, visited), ...]
	'''
	core_pt=OrderedDict()
	border_pt=OrderedDict()	
	noise_pt=OrderedDict()

	pointer=OrderedDict()
	visited=OrderedDict()
	visited={k: 1 for k in range(0,2000)}

	count=0
	neighbourhood=[]

	for index, point in enumerate(points):
		#print(i)
		for j, distance_pt in enumerate(points):#if distance_pt!=point
			#print(j)

			if distances(point, distance_pt) <eps:
				count+=1
				neighbourhood.append([distance_pt, j, 0]) #(neighbourhood point, index, visited)

		#print(count)
		#Create a pointer from point to neighbourhood
		pointer[index]=[point, neighbourhood]
		#print(count)
		#print(len(neighbourhood))
		neighbourhood=[]
		if count>minPts:
			core_pt[index]=[point,0, index]
			visited[index]=0
			print(p)
			p=p+1

		count=0
	print(len(visited))

	for index in core_pt.keys():
		for neighbourhood_pts in pointer[index][1]:
			if neighbourhood_pts[1] not in core_pt.keys() and visited[neighbourhood_pts[1]]==1: #not in visited.keys():
				border_pt[neighbourhood_pts[1]]=[neighbourhood_pts[0] ,0, index]
				visited[neighbourhood_pts[1]]=0
				print(p)
				p=p+1

	"""
	for index, point in enumerate(points):
		#print(i)
		if index not in visited.keys():
			no_core_pts=0
			visited[index]=1
			for j in core_pt.keys():#if distance_pt!=point
				print('yo')
				#print(j)	
				if distances(point, core_pt[j][0]) <eps:
					border_pt[index]=[point,0, index]
					visited[index]=1
					break
					no_core_pts+=1
					#neighbourhood.append([distance_pt, j, 0]) #(neighbourhood point, index, visited)

			"""
			#if count<minPts and no_core_pts>0:
			#	border_pt[index]=[point,0, index]
			#elif index not in core_pt.keys() and no_core_pts==0:
			#	noise_pt[index]=[point,0, index]
	"""
			#else:
			#	print(index, "nooo")
			count=0
	"""

	return core_pt, border_pt, noise_pt, pointer, visited

core_point, border_point, noise_point, Pointer, Visited =label(X_scaled)
print(len(core_point), len(border_point), len(noise_point), len(Pointer))
y_pred=list(Visited.values())

print(accuracy_score(y,y_pred))
#For given eps and minPts, the algorithm prints (1727 218 55)
#print('Core point= ',X_scaled[77], 'Pointer= ',pointer[77][0])

"""
def Density_Edges(core_pt, border_pt):
	final_Cluster=[]
	Cluster=[]
	#visited=0
	temp=0
	'''
	'''
	first_index=list(core_pt)[0]
	first_point=core_pt[first_index]
	core_pt_index=first_point
	'''
	'''
	#for vector in pointer[first_index][1][0]:
	#	if pointer[first_index][1][1] in 
	
	#while visited<len(X_scaled):

	while True:
		core_pt_index=choose_core_pt(core_pt)
		temp=0
		if core_pt_index==-1:
			return final_Cluster
		Cluster.append(core_pt[core_pt_index][0])
		#core_pt_index is the key index in core_pt
		#Add the core_pt into the cluster and make it visited
		temp_index=core_pt_index
		core_pt[core_pt_index][1]=1
		#Finding a core point in its neighbourhood
		neighbourhood_pts=Pointer[core_pt_index][1]
		# Check if core pt exists in neighbourhood and it has not been visited
		for core_neighbourhood_pts in neighbourhood_pts:
			neighbourhood_index=core_neighbourhood_pts[1]# Stores index of the corresponding point

			if neighbourhood_index in core_pt.keys(): # Is the neighbouring point a core point?
				if core_pt[neighbourhood_index][1]==0:# Has the neighbouring point been visited
					core_pt_index=neighbourhood_index
					core_pt[core_pt_index][1]=1 # The neighbouring point becomes the new core point
					temp=1
					break

		if temp==0:
			final_Cluster.append(Cluster)
			Cluster=[]

	return 0

#print(Density_Edges(core_point, border_point))
def choose_core_pt(core_pts):
	for k in core_pts:
		if core_pts[k][1]==0:
			core_pts[k][1]=1
			return k
	return -1


final=Density_Edges(core_point, border_point)

sum=0
for single_cluster in final:
	print(len(single_cluster))
	sum+=len(single_cluster)

print(sum)
"""