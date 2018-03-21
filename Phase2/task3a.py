import pandas as pd
import numpy as np
import sys
import task2a as t2a

#READING SEED ACTOR LIST
seed_size=len(sys.argv)-1
seed = []
for i in range(0,seed_size):
    seed.append(int(sys.argv[i+1]))

#READING THE REQUIRED DATA FILES
movie_actor=pd.read_csv('Data/movie-actor.csv')
imdb_actor_info=pd.read_csv('Data/imdb-actor-info.csv')
name_dict={k: list(v) for k,v in imdb_actor_info.groupby('id')['name']}

#RETRIEVING THE ACTOR-ACTOR SIMILARITY MATRIX FROM TASK 2A
a = raw_input("Please input 1-5 to choose the similarity/distance measure to use for finding similarity:\n"
              "1: Manhattan Distance"
              "\n2: Cosine Similarity"
              "\n3: Cosine Distance"
              "\n4: Mahalanobis Distance"
              "\n5: Euclidean Distance"
              "\nNote: Anything besides these will automatically consider Cosine Similarity\n"
              "Enter here: ")
try:
    a = int(a)
except TypeError:
    a = 2
except ValueError:
    a = 2
finally:
    if a > 5:
        a = 2

b = raw_input("\nDo you want to ignore actors with whom no tags were associated? "
              "\n0 to ignore, anything else to include: ")

if b == 0 or b == '0':
    b = False
else:
    b = True
name_list=[]
actor_list,co_co=t2a.get_actor_actor_similarity_matrix(a, b)
for i in actor_list:
    name_list.append(name_dict[i])
    

#FORMING THE TELEPORTATION VECTOR OF SEEDS
seed_vector=np.zeros(len(actor_list))
for i in range(0,len(actor_list)):
    for j in range(0,seed_size):
        if(actor_list[i]==seed[j]):
            seed_vector[i]=1
            continue

seed_vector=seed_vector/seed_size


movie_actor=movie_actor.drop('actor_movie_rank',axis=1)
actor_dict={k: list(v) for k,v in movie_actor.groupby('actorid')['movieid']}

#CONVERTING GRAPH INTO TRANSITION MATRIX
co_co = np.asarray(co_co)
transition=co_co/co_co.sum(axis=0)
alpha=0.85
maxerr=0.001
size=co_co.shape[0]
pr0,pr1=np.zeros(size),np.ones(size)

#ITERATIVELY FINDING PAGERANKS
while np.sum(np.abs(pr1-pr0)) > maxerr:
    pr0=pr1
    pr1=alpha*np.dot(transition,pr1)+(1-alpha)*seed_vector

pageranks=pd.DataFrame(columns=['actorid','pagerank','name'])
pageranks['actorid']=actor_list
pageranks['pagerank']=pr1
pageranks['name']=name_list
pageranks=pageranks.sort_values(by='pagerank', ascending=False)

#FINDING 10 MOST RELATED ACTORS
related_actors=pageranks[(-pageranks.actorid.isin(seed))]
print(related_actors.head(n=10))



