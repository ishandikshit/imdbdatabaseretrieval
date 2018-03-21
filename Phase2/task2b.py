# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:06:35 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
import math
from sklearn.decomposition import TruncatedSVD
from scipy import linalg
from sklearn.cluster import KMeans

#READING ALL REQUIRED DATA FILES
movie_actor=pd.read_csv('Data/movie-actor.csv')
#print(movie_actor)
imdb_actor_info=pd.read_csv('Data/imdb-actor-info.csv')
actor_list=imdb_actor_info['id'].tolist()
name_list=imdb_actor_info['name'].tolist()  #FORMING A LIST OF ACTORS NAMES
#print(actor_list,len(actor_list))
movie_actor=movie_actor.drop('actor_movie_rank',axis=1)
#print(movie_actor)

#MAKING A DICTIONARY FOR FINDING ALL MOVIES OF AN ACTOR
actor_dict={k: list(v) for k,v in movie_actor.groupby('actorid')['movieid']}
#print(actor_dict)
co_co=np.zeros((len(actor_list),len(actor_list)))

#FORMING THE CO-ACTOR MATRIX BASED ON MOVIES IN WHICH ACTORS APPEAR TOGETHER 
for i in range(0,len(actor_list)):
    for j in range(0,len(actor_list)):
        actor1=set(actor_dict[actor_list[i]])
        actor2=set(actor_dict[actor_list[j]])
        co_co[i,j]=len(set.intersection(actor1,actor2))
#print(co_co)

#SCIKIT IMPLEMENTATION
svd = TruncatedSVD(n_components=3,algorithm='arpack')
svd=svd.fit(co_co)  #FITS MODEL TO THE SVD OBJECT
dec=svd.fit_transform(co_co)    #PERFORMS REDUCTION USING THE MODEL
#print(dec)
latent_semantics=pd.DataFrame(columns=['actorid','name','ls1','ls2','ls3'])
latent_semantics['actorid']=actor_list
latent_semantics['name']=name_list
latent_semantics['ls1']=dec[:,0]
latent_semantics['ls2']=dec[:,1]
latent_semantics['ls3']=dec[:,2]

#REPORTING THE LATENT SEMANTICS
print('FIRST LATENT SEMANTIC')
ls1=latent_semantics.sort_values(by='ls1',ascending=False)
print(ls1[['actorid','name','ls1']])
print('SECOND LATENT SEMANTIC')
ls2=latent_semantics.sort_values(by='ls2',ascending=False)
print(ls2[['actorid','name','ls2']])
print('THIRD LATENT SEMANTIC')
ls3=latent_semantics.sort_values(by='ls3',ascending=False)
print(ls3[['actorid','name','ls3']])
kmeans = KMeans(n_clusters=3, random_state=0).fit(dec)

#GROUPING THE OBJECTS ACCORDING TO LATENT SEMANTICS
g1=[]
g2=[]
g3=[]
#print(len(kmeans.labels_))
#print(kmeans.labels_)
for i in range(0,len(kmeans.labels_)-1):
    if(kmeans.labels_[i]==0):
        g1.append(name_list[i])
    elif(kmeans.labels_[i]==1):
        g2.append(name_list[i])
    elif(kmeans.labels_[i]==2):
        g3.append(name_list[i])
print('GROUP1:')
print(g1)
print('GROUP2:')
print(g2)
print('GROUP3:')
print(g3)
