import pandas as pd
import numpy as np
import math
#from sklearn.decomposition import TruncatedSVD
from scipy import linalg
import sys
import time
import pickle

def get_output(userid,relevant,irrelevant):
	ml_tag=pd.read_csv('Data/mltags.csv')
	movie_tag=pd.DataFrame(columns=['movieid','tagid','timeweight'])

	ts_list=ml_tag['timestamp'].tolist()
	time_ranks=[]

	#converting timestamp to integer value
	for x in ts_list:
		 date_time = x
		 pattern = '%Y-%m-%d %H:%M:%S'
		 epoch = int(time.mktime(time.strptime(date_time, pattern)))
		 time_ranks.append(epoch)
	       
	#print(time_ranks)
	time_weights=[]
	#converting integer value of timestamp to time weight
	for x in range(0,len(time_ranks)):
	        if(len(time_ranks)==1):
	            time_weights.append(1)
	            break;
	        time_weights.append((time_ranks[x]-min(time_ranks))/(max(time_ranks)-min(time_ranks)))

	movie_tag['movieid']=ml_tag['movieid']
	movie_tag['tagid']=ml_tag['tagid']
	movie_tag['timeweight']=time_weights
	tagged_movies=movie_tag['movieid'].tolist()
	tagged_movies_tags=movie_tag['tagid'].tolist()

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

	#DICTIONARY FOR MOVIE AND TAGS
	tf_dict={k: list(v) for k,v in movie_tag.groupby('movieid')['tagid']}
	idf_dict={k: list(v) for k,v in movie_tag.groupby('tagid')['movieid']}
	

	R=len(relevant)
	N=len(relevant+irrelevant)
	ri=[]
	ni=[]
	selected_dict={}
	rel_dict={}
	top5_id=relevant+irrelevant

	relavance=[]
	for i in relevant:
		relavance.append(1)
	#print(relavance)
	#print(len(irrelevant))
	for i in range(0,len(irrelevant)):
		relavance.append(-1)
	print len(relavance)
	rel_dict={}
	if R==0:
		irrelevant=top5_id

	#print(len(top5_id),len(relavance))
	#print(relavance)

	for i in range(0,len(top5_id)):
		rel_dict[top5_id[i]]=relavance[i]
	#print(R)
	tagset=set()
	#print(relavance)
	for x in top5_id:
		tags_list=tf_dict[x]
		selected_dict[x]=tags_list
		tagset=tagset.union(set(tags_list))

	tags_top5=list(tagset)

	#print(selected_dict)
	#print(tagset)

	ri=[]
	ni=[]
	for i in range(0,len(tags_top5)):
		ri.append(0)
		ni.append(0)

	for i in range(0,len(tags_top5)):
		for m in top5_id:
			l1=selected_dict[m]
			rval=rel_dict[m]
			if (tags_top5[i] in l1):
				ni[i]=ni[i]+1
				if(rval==1):
					ri[i]=ri[i]+1
	#print(ni,ri)
	#print(ni)

	pr_feedback=[]

	for i in range(0,len(tags_top5)):
		try:
			numerator=float((ri[i]))/(R-ri[i])
			denominator=float((ni[i]-ri[i]))/(N-R-ni[i]+ri[i])
			pr=math.log((numerator/denominator),2)
		except:
			approx=float(ni[i])/N
			numerator=(ri[i]+approx)/(R+1)
			denominator=(ni[i]-ri[i]+approx)/(N-R+1)
			pr=math.log((numerator/denominator),2)

		pr_feedback.append(pr)

	for i in range(0,len(pr_feedback)):
		pr_feedback[i]=(pr_feedback[i]-min(pr_feedback))/max(pr_feedback)


	#print(pr_feedback)



	pr_dict={}
	'''
	for i in range(0,len(pr_feedback)):
		pr_dict[tags_top5[i]]=pr_feedback[i]

	print(pr_dict)
	'''
	'''
	for i in range(0,len(tag_list)):
		for j in pr_dict.keys():
			if tag_list[i]==j:
				print('found')
				movie_tag_matrix[:,i]*=pr_dict[j]
	'''
	for i in range(0,len(tags_top5)):
		for j in range(0,len(tag_list)):
			if tag_list[j]==tags_top5[i]:
				#print('found')
				#print(tag_list[j])
				movie_tag_matrix[:,i]*=pr_feedback[i]




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

get_output(1,[1684,2341,2863,2949],[7309,10021,6598,65,543,1070])

#Command to call: give user id as input
#print(get_output(1))