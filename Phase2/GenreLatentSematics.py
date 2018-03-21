import dataframes as df
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


genre = ''
model = ''
if len(sys.argv) > 1:
    genre = sys.argv[3]
    model = sys.argv[2]
if model == 'lda':
    model='tf'
if model in ('svd', 'pca'):
    model = 'tfidf'


# tf = count of the given tag for the genre / total number of tags for that genre
# tfweighted = epoch * tf in the movie in which the tag has occured
# now we have multiple tags repeated for the same actor but different timestamps
# should we take average of the values of tf? makes more sense to take average if one tag is repeated more than once

# idf = log(total number of actors/number of actors where this tag has appeared)


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
    if model == 'tf':
        movies_tags = movies_tags.sort_values('tf', ascending=False)
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
        for data in range(len(movies_tags)):
            output += " <{} {}>".format(movies_tags['tag'].iloc[data], movies_tags[model].iloc[data])
            structured_output[movies_tags['tag'].iloc[data]] = movies_tags[model].iloc[data]
        return structured_output


# Method to get genre vectors in actor space
# Parameters - Genre, Model
def get_genre_vector_actor(genre, model):
    output = ''
    movies = df.movies_data.loc[df.movies_data['genre'] == genre]
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

        # number of genres where that particular tag appears
        actor_genre_count = df.movie_actors_data.loc[df.movie_actors_data['actorid'].isin(movies_actors['actorid'])].merge(df.movies_data, on=['movieid'], how='left').groupby(['actorid', 'genre']).size().reset_index().groupby('actorid').size().reset_index()
        # tag_genre_count = df.tags_data.loc[df.tags_data['tagid'].isin(movies_tags['tagid'])].merge(df.movies_data, on=['movieid'], how='left').groupby(['tagid', 'genre']).size().reset_index().groupby('tagid').size().reset_index()
        # print "Actors : "+str(actor_genre_count)+" "+str(movies_actors)+"-----END--------"
        movies_actors = actor_genre_count.merge(movies_actors)
        # movies_tags = tag_genre_count.merge(movies_tags)
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


if genre == '' or model == '':
    """
    f = open('output/allgenres_tf.txt', 'w')
    f.write(for_all_genres('tf'))
    f.close()

    f = open('output/allgenres_tfidf.txt', 'w')
    f.write(for_all_genres('tfidf'))
    f.close()
    """
    pass
else:
    a = []
    df_genre = pd.DataFrame()
    df_actor = pd.DataFrame()
    if sys.argv[1]=='tag':
        for genre in get_total_genres():
            a.append(genre)
            vector = get_genre_vector(genre, model)
            df_genre = df_genre.append(vector, ignore_index=True)

    elif sys.argv[1]=='actor':
        for genre in get_total_genres():
            a.append(genre)
            vector2 = get_genre_vector_actor(genre, model)
            df_actor = df_actor.append(vector2, ignore_index=True)
    df_genre = df_genre.fillna(0)
    df_actor = df_actor.fillna(0)
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()

    plt.cla()

    # Loading PCA model for tag space
    pca_genre = decomposition.PCA(n_components=4)
    
    # Loading SVD model for tag space
    svd_genre = TruncatedSVD(n_components=4, n_iter=7, random_state=42)
    
    # Loading LDA model for tag space
    lda_genre = LatentDirichletAllocation(n_components=4, max_iter=10, learning_method='online', learning_offset=50., random_state=0)


    # Loading PCA model for actor space
    pca_actor = decomposition.PCA(n_components=4)
    # Loading SVD model for actor space
    svd_actor = TruncatedSVD(n_components=4, n_iter=7, random_state=42)
    # Loading LDA model for actor space
    lda_actor = LatentDirichletAllocation(n_components=4, max_iter=10, learning_method='online', learning_offset=50., random_state=0)

    if sys.argv[1]=='tag':
        genre_svd = svd_genre.fit_transform(df_genre)
        genre_pca = pca_genre.fit_transform(df_genre)
        genre_lda = lda_genre.fit_transform(df_genre)

    elif sys.argv[1]=='actor':
        actor_svd = svd_actor.fit_transform(df_actor)
        actor_pca = pca_actor.fit_transform(df_actor)
        actor_lda = lda_actor.fit_transform(df_actor)

    
    index=None
    if sys.argv[3] in a:
        index = a.index(sys.argv[3])
    if index==None:
        print "GENRE NOT FOUND!!"
        exit()
    if sys.argv[2] =='lda':
        if sys.argv[1]=='tag':
            print '------TAG LDA-----'
            print genre_lda[index]
            print_top_words(lda_genre, list(df_genre), 4)
        elif sys.argv[1] == 'actor':
            print '------ACTOR LDA-----'
            print actor_lda[index]
            print_top_words(lda_actor, list(df_actor), 4)

    elif sys.argv[2] == 'pca':
        if sys.argv[1] == 'actor':
            print '------ACTOR PCA-----'
            print actor_pca[index]
            print_top_words(pca_actor, list(df_actor), 4)
        elif sys.argv[1]=='tag':
            print '------TAG PCA-----'
            print genre_pca[index]
            print_top_words(pca_genre, list(df_genre), 4)
    elif sys.argv[2] == 'svd':
        if sys.argv[1] == 'actor':
            print '------ACTOR SVD-----'
            print actor_svd[index]
            print_top_words(svd_actor, list(df_actor), 4)
        elif sys.argv[1]=='tag':
            print '------TAG SVD-----'
            print genre_svd[index]
            print_top_words(svd_genre, list(df_genre), 4)
