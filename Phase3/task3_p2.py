import pandas as pd
import numpy as np
import math
#from sklearn.decomposition import TruncatedSVD
from scipy import linalg
import sys
import time
import pickle
from scipy.sparse.linalg import svds, eigs
from sklearn.decomposition import TruncatedSVD
from scipy import stats
#import statistics
from collections import Counter

def flip(b):
	if b==0:
		return 1
	else:
		return 0

fp=open("movie_tag_matrix.pickle","rb")
movie_tag_matrix = pickle.load(fp)
fp.close()

ml_movies=pd.read_csv('Data/mlmovies.csv')
movie_list=ml_movies['movieid'].tolist()
genome_tags=pd.read_csv('Data/genome-tags.csv')
tag_list=genome_tags['tagId'].tolist()
name_list=ml_movies['moviename'].tolist()

u, s, vt = svds(movie_tag_matrix, k=500)
#svd = TruncatedSVD(n_components=500)
#svd.fit(movie_tag_matrix) 
#u=svd.fit_transform(movie_tag_matrix)
#print(u)

u=np.dot(u,np.diag(s))
#print(u.shape)
#print(u.max())
#print(u.min())
print 'Enter the number of layers (L):'
L=int(input())

print 'Enter the number of hash functions per layer (k):'
k=int(input())

print 'Enter set of movie vectors:(type 0 after last vector)'
movie_vectors=[]
while True:
	vec=int(input())
	if(vec==0):
		break
	else:
		movie_vectors.append(int(vec))

Layers=[]
Layers_hash_table=[]
#print(n_buckets)
for i in range(0,L):
	hash_family=np.zeros((500,k))
	for j in range(0,k):
		mu,sigma=0,1
		hash_family[:,j]=np.random.normal(mu,sigma,500)
	#print(hash_family)
	Layers.append(hash_family)
	hash_values=np.zeros((len(movie_list),k))
	for j in range(0,len(movie_list)):
		for a in range(0,k):
			hash_value=np.dot(u[j],hash_family[:,a])
			if(hash_value>=0):
				hash_values[j][a]=1
			else:
				hash_values[j][a]=0
			hash_values=hash_values.astype(int)
	#print(hash_values)
	single_layer_dict={}
	for j in range(0,len(movie_list)):
		hashindex=''
		for a in range(0,len(hash_values[j])):
			hashindex=hashindex+str(hash_values[j,a])
		if(hashindex in single_layer_dict.keys()):
			single_layer_dict[hashindex].append(movie_list[j])
		else:
			l1=[movie_list[j]]
			single_layer_dict[hashindex]=l1
	#print(single_layer_dict)
	#print(len(single_layer_dict.keys()))
	Layers_hash_table.append(single_layer_dict)

print 'Enter movieid to be searched:' 
key_id=int(input())
key_index=movie_list.index(key_id)
key_point=u[key_index]
key_r=0
while key_r<1:
	print 'Enter no. of closest movie to be returned (r):'
	key_r=int(input())

retrieved_buckets=[]
query_points=[]
for i in range(0,L):
	hash_family=Layers[i]
	hash_values=np.dot(key_point,hash_family)
	for j in range(0,k):
		if(hash_values[j]>=0):
			hash_values[j]=1
		else:
			hash_values[j]=0
	hash_values=hash_values.astype(int)
	#print(hash_values)
	retrieved_buckets.append(hash_values)
	hash_table=Layers_hash_table[i]
	key_bucket=''
	for j in range(0,k):
		key_bucket=key_bucket+str(hash_values[j])
	l1=(hash_table[key_bucket])
	query_points=query_points+l1
#print(set(query_points))
#print(len(query_points))
#print(len(set(query_points)))
query_set=set(query_points)
query_set=list(query_set)

if(key_r<len(query_set)):
	distances=[]
	names=[]
	for i in query_set:
		index=movie_list.index(i)
		point_vector=u[index]
		point_distance=math.sqrt(np.sum((point_vector - key_point)**2))
		distances.append(point_distance)
		names.append(name_list[index])
	results=pd.DataFrame(columns=['id','name','distance'])
	results['id']=query_set
	results['name']=names
	results['distance']=distances
else:
	m=0
	for i in range(0,L):
		hvalue=retrieved_buckets[i]
		hvalue[m]=flip(hvalue[m])
		key_bucket=''
		for j in range(0,k):
			key_bucket=key_bucket+str(hash_values[j])
		l1=(hash_table[key_bucket])
		query_set=list(set(query_set+l1))
		query_points=query_points+l1
		if key_r < len(query_set):
			break
	distances=[]
	names=[]
	for i in query_set:
		index=movie_list.index(i)
		point_vector=u[index]
		point_distance=math.sqrt(np.sum((point_vector - key_point)**2))
		distances.append(point_distance)
		names.append(name_list[index])
	results=pd.DataFrame(columns=['id','name','distance'])
	results['id']=query_set
	results['name']=names
	results['distance']=distances





results=results.sort_values(by='distance', ascending=True)
results=results[(-results.id.isin([key_id]))]
results.set_index('id',inplace=True)
print results.head(n=key_r)
		
print 'No. of movies considered:'
print len(query_points)

print 'No. of unique movies considered:'
print len(query_set)









