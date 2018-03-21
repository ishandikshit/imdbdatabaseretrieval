import dataframes as df
from Data import reader as dr
import numpy as np
import math
import sys
import pandas as pd

from sklearn import decomposition
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity


genre = ''
model = ''
if len(sys.argv) > 1:
    genre = sys.argv[3]
    model = sys.argv[2]
if model == 'lda':
    model='tf'
if model in ('svd', 'pca'):
    model = 'tf'


# tf = count of the given tag for the genre / total number of tags for that genre
# tfweighted = epoch * tf in the movie in which the tag has occured
# now we have multiple tags repeated for the same actor but different timestamps
# should we take average of the values of tf? makes more sense to take average if one tag is repeated more than once

# idf = log(total number of actors/number of actors where this tag has appeared)

#PHASE-3
def get_movie_vector(movieid, model):
    output = ''
    movies = df.movies_data.loc[df.movies_data['movieid'] == movieid]
    tags = df.tags_data.loc[df.tags_data['movieid'].isin(movies['movieid'])]
    movies_tags = movies.merge(tags, on='movieid')
    total_tags_for_movie = len(movies_tags)
    movies_tags = movies_tags.merge(movies_tags.groupby('tagid').size().reset_index())
    movies_tags['tf'] = (movies_tags[0]/total_tags_for_movie) * movies_tags['epoch']
    tags_names = df.genome_data.loc[df.genome_data['tagid'].isin(movies_tags['tagid'])]
    movies_tags = movies_tags.merge(tags_names, on='tagid', how='left')

    # del movies_tags[0]
    # del movies_tags['epoch']

    #phase2
    structured_output = {}
    movies_tags = movies_tags.groupby('tag').mean().reset_index()
    # print movies_tags
    if model == 'tf':
        movies_tags = movies_tags.sort_values('tf', ascending=False)
        # print movies_tags
        for data in range(len(movies_tags)):
            output += " <{} {}>".format(movies_tags['tag'].iloc[data], movies_tags[model].iloc[data])
            structured_output[movies_tags['tag'].iloc[data]] = movies_tags[model].iloc[data]
        # print structured_output
        return structured_output

    elif model == 'tfidf':
        movies_tags = movies_tags.groupby('tag').mean().reset_index()

        total_movies = len(df.movies_data.groupby('movieid').size())

        # number of genres where that particular tag appears
        tag_genre_count = df.tags_data.loc[df.tags_data['tag'].isin(movies_tags['tag'])].merge(df.movies_data, on=['movieid'], how='left').groupby(['tag', 'genre']).size().reset_index().groupby('tagid').size().reset_index()
        # print "Tags : "+str(tag_genre_count)+" "+str(movies_tags)+"-----END--------"
        movies_tags = tag_genre_count.merge(movies_tags)
        movies_tags['count_genres_tagid'] = tag_genre_count[0]
        del movies_tags[0]
        if len(movies_tags) < 1:
            return None
        movies_tags['idf'] = np.vectorize(math.log)(total_genres/movies_tags['count_genres_tagid'])
        movies_tags['tfidf'] = movies_tags['idf']*movies_tags['tf']
        movies_tags = movies_tags.sort_values('tfidf', ascending=False)
        tags_names = df.genome_data.loc[df.genome_data['tagid'].isin(movies_tags['tagid'])]
        movies_tags = movies_tags.merge(tags_names, on='tagid', how='left')
        # print movies_tags
        for data in range(len(movies_tags)):
            output += " <{} {}>".format(movies_tags['tag'].iloc[data], movies_tags[model].iloc[data])
            structured_output[movies_tags['tag'].iloc[data]] = movies_tags[model].iloc[data]
        return structured_output


def get_genre_vector(genre, model):
    output = ''
    movies = df.movies_data.loc[df.movies_data['genre'] == genre]
    tags = df.tags_data.loc[df.tags_data['movieid'].isin(movies['movieid'])]
    movies_tags = movies.merge(tags, on='movieid')
    del movies_tags['moviename']
    del movies_tags['userid']
    del movies_tags['movieid']
    del movies_tags['genre']
    total_tags_for_genre = len(movies_tags)
    movies_tags = movies_tags.merge(movies_tags.groupby('tagid').size().reset_index())

    movies_tags['tf'] = (movies_tags[0]/total_tags_for_genre) * movies_tags['epoch']
    tags_names = df.genome_data.loc[df.genome_data['tagid'].isin(movies_tags['tagid'])]
    movies_tags = movies_tags.merge(tags_names, on='tagid', how='left')
    del movies_tags[0]
    del movies_tags['epoch']

    #phase2
    structured_output = {}
    movies_tags = movies_tags.groupby('tag').mean().reset_index()
    # print movies_tags
    if model == 'tf':
        movies_tags = movies_tags.sort_values('tf', ascending=False)
        # print movies_tags
        for data in range(len(movies_tags)):
            output += " <{} {}>".format(movies_tags['tag'].iloc[data], movies_tags[model].iloc[data])
            structured_output[movies_tags['tag'].iloc[data]] = movies_tags[model].iloc[data]
        return structured_output
    elif model == 'tfidf':
        movies_tags = movies_tags.groupby('tagid').mean().reset_index()

        total_genres = len(df.movies_data.groupby('genre').size())

        # number of genres where that particular tag appears
        tag_genre_count = df.tags_data.loc[df.tags_data['tagid'].isin(movies_tags['tagid'])].merge(df.movies_data, on=['movieid'], how='left').groupby(['tagid', 'genre']).size().reset_index().groupby('tagid').size().reset_index()
        # print "Tags : "+str(tag_genre_count)+" "+str(movies_tags)+"-----END--------"
        movies_tags = tag_genre_count.merge(movies_tags)
        movies_tags['count_genres_tagid'] = tag_genre_count[0]
        del movies_tags[0]
        if len(movies_tags) < 1:
            return None
        movies_tags['idf'] = np.vectorize(math.log)(total_genres/movies_tags['count_genres_tagid'])
        movies_tags['tfidf'] = movies_tags['idf']*movies_tags['tf']
        movies_tags = movies_tags.sort_values('tfidf', ascending=False)
        tags_names = df.genome_data.loc[df.genome_data['tagid'].isin(movies_tags['tagid'])]
        movies_tags = movies_tags.merge(tags_names, on='tagid', how='left')
        # print movies_tags
        for data in range(len(movies_tags)):
            output += " <{} {}>".format(movies_tags['tag'].iloc[data], movies_tags[model].iloc[data])
            structured_output[movies_tags['tag'].iloc[data]] = movies_tags[model].iloc[data]
        return structured_output


# Method to get genre vectors in actor space
# Parameters - Genre, Model
def get_genre_vector_actor(genre, model):
    output = ''
    movies = df.movies_data.\
loc[df.movies_data['genre'] == genre]
    actors = df.movie_actors_data[df.movie_actors_data['movieid'].isin(movies['movieid'])]
    # tags = df.tags_data.loc[df.tags_data['movieid'].isin(movies['movieid'])]
    movies_actors = movies.merge(actors, on='movieid')
    del movies_actors['moviename']
    # del movies_actors['userid']
    del movies_actors['movieid']
    del movies_actors['genre']
    total_tags_for_genre = len(movies_actors)
    movies_actors = movies_actors.merge(movies_actors.groupby('actorid').size().reset_index())
    movies_actors['tf'] = (movies_actors[0]/total_tags_for_genre) * movies_actors['actor_movie_rank']
    actor_names = df.imdb_actor_data.loc[df.imdb_actor_data['actorid'].isin(movies_actors['actorid'])]

    # print movies_actors
    movies_actors = movies_actors.merge(actor_names, on='actorid', how='left')
    # print movies_actors
    del movies_actors[0]
    del movies_actors['actor_movie_rank']

    #phase2
    structured_output = {}
    movies_actors = movies_actors.groupby('name').mean().reset_index()
    if model == 'tf':
        movies_actors = movies_actors.sort_values('tf', ascending=False)
        for data in range(len(movies_actors)):
            output += " <{} {}>".format(movies_actors['name'].iloc[data], movies_actors[model].iloc[data])
            structured_output[movies_actors['name'].iloc[data]] = movies_actors[model].iloc[data]
        return structured_output
    elif model == 'tfidf':
        movies_actors = movies_actors.groupby('actorid').mean().reset_index()
        total_genres = len(df.movies_data.groupby('genre').size())
        actor_genre_count = df.movie_actors_data.loc[df.movie_actors_data['actorid'].isin(movies_actors['actorid'])].merge(df.movies_data, on=['movieid'], how='left').groupby(['actorid', 'genre']).size().reset_index().groupby('actorid').size().reset_index()
        movies_actors = actor_genre_count.merge(movies_actors)
        movies_actors['count_genres_actorid'] = actor_genre_count[0]
        del movies_actors[0]
        if len(movies_actors) < 1:
            return None
        movies_actors['idf'] = np.vectorize(math.log)(total_genres/movies_actors['count_genres_actorid'])
        movies_actors['tfidf'] = movies_actors['idf']*movies_actors['tf']
        movies_actors = movies_actors.sort_values('tfidf', ascending=False)
        actor_names = df.imdb_actor_data.loc[df.imdb_actor_data['actorid'].isin(movies_actors['actorid'])]
        movies_actors = movies_actors.merge(actor_names, on='actorid', how='left')
        for data in range(len(movies_actors)):
            output += " <{} {}>".format(movies_actors['name'].iloc[data], movies_actors[model].iloc[data])
            structured_output[movies_actors['name'].iloc[data]] = movies_actors[model].iloc[data]
        return structured_output


def for_all_genres(model):
    s = ''
    for index, row in df.movies_data.groupby('genre').size().reset_index().iterrows():
        t = (str(row['genre']))
        s += ''.join(t)
        s += '\n'
    print s
    return s

def for_all_actors(model):
    s = ''
    for index, row in df.imdb_actor_data.groupby('name').size().reset_index().iterrows():
        t = (str(row['name']))
        s += ''.join(t)
        s += '\n'
    print s
    return s

#get all the tags in the dataset
def get_total_genres():
    a =[]
    ml_movies = df.movies_data
    genres = ml_movies.genre
    for genre in genres:
        s = genre.split("|")
        a=a+s

    a= list(set(a))

    return a


#get all the movies in the dataset
def get_total_movies():
    a =[]
    ml_movies = df.movies_data
    movies = ml_movies.movieid
    # for genre in genres:
    #     s = genre.split("|")
    #     a=a+s

    a= list(set(movies))

    return a

#get all the actors in the dataset
def get_total_actors():
    a =[]
    imdb_actors = df.imdb_actor_data
    actors = imdb_actors.name
    for actor in actors:
        s = actor.split("|")
        a=a+s

    a= list(set(a))

    return a

# Method to print the top words in each latent semantic
def print_top_words(model, feature_names, n_top_words):
    print "\nMost contributing features per latent semantic: "
    for topic_idx, topic in enumerate(model.components_):
        message = "Latent semantic "+str(topic_idx+1)+": "
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def get_similarity_matrix(movie_lda):

    coo = coo_matrix(movie_lda)
    sim = cosine_similarity(coo.tocsr(), dense_output=False)
    return sim

def get_movies_watched_by_user(user_id):
    list1 = []
    # tags_record = __user_movie_table[__user_movie_table['userid'] == user_id]
    # list1 = tags_record.movieid.unique().tolist()

    rating_record = df.ratings_data[df.ratings_data['userid'] == user_id]
    list2 = rating_record.movieid.unique().tolist()

    return list(set(list1) | set(list2))

def get_feedback_paper(R, N, ri, ni):
    pr_feedback=0
    print R, N, ri, ni

    # for i in range(0,len(tags_top5)):
    try:
        numerator=(ri)/(R-ri)
        denominator=(ni-ri)/(N-R-ni+ri)
        pr=abs(math.log((numerator/denominator),2))
    except:
        numerator=(ri+(ni/N+1))/(R+1)
        denominator=(ni-ri+(ni/N+1))/(N-R+1)+1
        pr=abs(math.log((numerator/denominator),2))

    pr_feedback=pr
    return pr
    # for i in range(0,len(pr_feedback)):
    pr_feedback=(pr_feedback-min(pr_feedback))/max(pr_feedback)

    return pr_feedback

if genre == '' or model == '':
    pass
else:

    #PHASE 3 START

    #PHASE 3 END

    a = []
    R=0.0
    ri=[]
    N=5.0
    ni = []
    df_genre = pd.DataFrame()
    df_movie = pd.DataFrame()
    df_actor = pd.DataFrame()
    i=0

    import os.path
    cache=False
    if os.path.exists("Data/movie_tag_matrix2.csv"):
        # df_movie = pd.read_csv("Data/movie_tag_matrix.csv")
        cache = True

    for movieid in get_total_movies():
        a.append(movieid)
        if not cache:
            vector = get_movie_vector(movieid, model)
            print i
            df_movie = df_movie.append(vector, ignore_index=True)
            # if i==200:
            #     break
            i=i+1
    # print df_movie
    if not cache:
        df_movie.to_csv("Data/movie_tag_matrix2.csv", sep=",")
    if cache:
        from numpy import genfromtxt
        df_movie = pd.read_csv('Data/movie_tag_matrix2.csv')

    df_movie = df_movie.fillna(0)

    lda_movie = LatentDirichletAllocation(n_components=4, max_iter=10, learning_method='online', learning_offset=50., random_state=0)

    disliked_movies=[]
    while(True):
        movie_lda = lda_movie.fit_transform(df_movie)
        
        user_id=int(sys.argv[3])
        movie_list = get_movies_watched_by_user(user_id)
        # print "Movies Watched By User\n-----------------------"
        for mov in movie_list:
            movs = df.movies_data.loc[df.movies_data['movieid'] == mov]
            # print movs['moviename'].values[0]
        sim_matrix = get_similarity_matrix(movie_lda)
        
        index = None
        top=[]
        sim = []
        for movie in movie_list:
            if int(movie) in a:
                index = a.index(int(movie))
            temp = df.ratings_data.loc[(df.ratings_data['movieid']==movie) & (df.ratings_data['userid']==user_id)]
            time_watched = temp['timestamp'].values
            sorted_matrix = sim_matrix[index].data.argsort()[::-1]
            sim.append(sim_matrix[index, sorted_matrix[0]])
            for i in range(0,2):
                top.append(sorted_matrix[i])

        print "Movies Recommended:\n------------------ "
        t = []
        # print "Disliked Movies: "
        # print disliked_movies
        outp_movies = []
        print sim
        itr=0
        for mov in top:
            movs = df.movies_data.loc[df.movies_data['movieid'] == a[mov]]
            if movs['movieid'].values[0] not in disliked_movies:
                t.append({'id': movs['movieid'].values[0], 'name': movs['moviename'].values[0], 'distance': 0.1})
            itr+=1
        # print sim_matrix
        outp_movies=t[:5]
        print outp_movies

        #take input
        user_feedback = raw_input("\nPlease enter the index of movies you dont like separated by space\n")
        user_feedback = user_feedback.split()
        user_feedback = map(int, user_feedback)
        R=5-len(user_feedback)

        for movie in user_feedback:
            disliked_movies.append(movie)
            sorted_matrix = sim_matrix[index].data.argsort()[::-1]
            for i in range(1, 6):
                disliked_movies.append(a[sorted_matrix[i]])

        ni=[]
        ri=[]
        for movie in outp_movies:
            if movie['id'] not in user_feedback:
                ri=ri+list((get_movie_vector(movie['id'], model)).keys())
            ni=ni+list((get_movie_vector(movie['id'], model)).keys())
        n_ri = len(ri)
        n_ni = len(ni)
        calculated_feedback = get_feedback_paper(R, N, n_ri, n_ni)
        for tag in ri:
            df_movie[tag] = calculated_feedback
        #remove all these disliked movies from initial list

