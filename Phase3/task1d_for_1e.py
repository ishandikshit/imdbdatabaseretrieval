import pandas as pd
import numpy as np
import math
#from sklearn.decomposition import TruncatedSVD
from scipy import linalg
import sys
import time
import pickle

def get_output(userid):
	mlratings=pd.read_csv('Data/mlratings.csv')
	user_movies=mlratings[mlratings.userid==int(userid)]
	seed=user_movies['movieid'].tolist()
	#READING THE SEED ACTOR LIST
	'''
	seed_size=len(sys.argv)-1
	for i in range(0,seed_size):
	    seed.append(int(sys.argv[i+1]))
	'''
	seed_size=len(seed)
	fp=open("movie_tag_matrix.pickle","rb")
	movie_tag_matrix = pickle.load(fp)
	fp.close()
	ml_movies=pd.read_csv('Data/mlmovies.csv')
	movie_list=ml_movies['movieid'].tolist()
	genome_tags=pd.read_csv('Data/genome-tags.csv')
	tag_list=genome_tags['tagId'].tolist()
	

	movie_movie_similarity_matrix=np.dot(movie_tag_matrix,movie_tag_matrix.T)
	seed_vector=np.zeros(len(movie_list))
	for i in range(0,len(movie_list)):
	    for j in range(0,seed_size):
	        if(movie_list[i]==seed[j]):
	            seed_vector[i]=1
	            continue
	seed_vector=seed_vector/seed_size 
	#INITIALIZING PARAMETERS
	alpha=0.85
	maxerr=0.001
	size=movie_movie_similarity_matrix.shape[0]
	pr0,pr1=np.zeros(size),np.ones(size)

	#NORMALIZING THE COLUMNS OF GRAPH TO MAKE TRANSITION MATRIX
	#transition=movie_movie_similarity_matrix/movie_movie_similarity_matrix.sum(axis=0)
	transition=np.zeros((len(movie_list),len(movie_list)))
	for i in range(0,len(movie_list)):
		sum=np.sum(movie_movie_similarity_matrix[:,i])
		if sum !=0:
			transition[:,i]=movie_movie_similarity_matrix[:,i]/sum


	#ITERATIVELY COMPUTING PAGERANKS
	while np.sum(np.abs(pr1-pr0)) > maxerr:
	    pr0=pr1
	    pr1=alpha*np.dot(transition,pr1)+(1-alpha)*seed_vector
	name_list=ml_movies['moviename'].tolist()
	pageranks=pd.DataFrame(columns=['movieid','pagerank','name'])
	pageranks['movieid']=movie_list
	pageranks['pagerank']=pr1
	pageranks['name']=name_list
	#print(pageranks)
	pageranks=pageranks.sort_values(by='pagerank', ascending=False)
	#print(pageranks)

	#FINDING THE TOP 10 RELATED ACTORS
	related_movies=pageranks[(-pageranks.movieid.isin(seed))]
	#Dataframe returned has columns : 'movieid','pagerank','name'
	return related_movies

#Command to call: give user id as input
print(get_output(1))