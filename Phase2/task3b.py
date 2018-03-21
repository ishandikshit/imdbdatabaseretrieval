# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:12:42 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
import math
from sklearn.decomposition import TruncatedSVD
from scipy import linalg
import sys

#READING THE SEED ACTOR LIST
seed_size=len(sys.argv)-1
seed=[]
for i in range(0,seed_size):
    seed.append(int(sys.argv[i+1]))
    
#READING THE REQUIRED DATA FILES
movie_actor=pd.read_csv('Data/movie-actor.csv')
#print(movie_actor)
imdb_actor_info=pd.read_csv('Data/imdb-actor-info.csv')
actor_list=imdb_actor_info['id'].tolist()   #FORMING A LIST OF ACTORS
#print(actor_list,len(actor_list))
name_list=imdb_actor_info['name'].tolist()  #FORMING A LIST OF ACTORS NAMES
#FORMING THE TELEPORTATION SEED VECTOR
seed_vector=np.zeros(len(actor_list))
for i in range(0,len(actor_list)):
    for j in range(0,seed_size):
        if(actor_list[i]==seed[j]):
            seed_vector[i]=1
            continue
seed_vector=seed_vector/seed_size    #ASSIGNING PROBABILITY TO SEED VECTOR      

movie_actor=movie_actor.drop('actor_movie_rank',axis=1)
#print(movie_actor)
actor_dict={k: list(v) for k,v in movie_actor.groupby('actorid')['movieid']}
#print(actor_dict)
co_co=np.zeros((len(actor_list),len(actor_list)))

#FORMING THE CO-ACTOR-CO-ACTOR MATRIX BASED ON MOVIES THAT ACTORS HAVE ACTED TOGETHER
for i in range(0,len(actor_list)):
    for j in range(0,len(actor_list)):
        actor1=set(actor_dict[actor_list[i]])
        actor2=set(actor_dict[actor_list[j]])
        co_co[i,j]=len(set.intersection(actor1,actor2))
#print(co_co)

#NORMALIZING THE COLUMNS OF GRAPH TO MAKE TRANSITION MATRIX
transition=co_co/co_co.sum(axis=0)
#print(transition)

#INITIALIZING PARAMETERS
alpha=0.85
maxerr=0.001
size=co_co.shape[0]
pr0,pr1=np.zeros(size),np.ones(size)

#ITERATIVELY COMPUTING PAGERANKS
while np.sum(np.abs(pr1-pr0)) > maxerr:
    pr0=pr1
    pr1=alpha*np.dot(transition,pr1)+(1-alpha)*seed_vector

#print(pr1)
pageranks=pd.DataFrame(columns=['actorid','pagerank','name'])
pageranks['actorid']=actor_list
pageranks['pagerank']=pr1
pageranks['name']=name_list
#print(pageranks)
pageranks=pageranks.sort_values(by='pagerank', ascending=False)
#print(pageranks)

#FINDING THE TOP 10 RELATED ACTORS
related_actors=pageranks[(-pageranks.actorid.isin(seed))]
print(related_actors.head(n=10))