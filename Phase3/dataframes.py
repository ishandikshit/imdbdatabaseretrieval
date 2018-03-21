import time
import numpy as np
from pandas import Series
from Data import reader as dr

GENOME_TAGS = "Data/genome-tags.csv"
IMDB_ACTOR_INFO = "Data/imdb-actor-info.csv"
ML_MOVIES = "Data/mlmovies.csv"
ML_RATINGS = "Data/mlratings.csv"
ML_TAGS = "Data/mltags.csv"
ML_USERS = "Data/mlusers.csv"
MOVIE_ACTORS = "Data/movie-actor.csv"


genome_data = dr.read_data(GENOME_TAGS)
imdb_actor_data = dr.read_data(IMDB_ACTOR_INFO)
movies_data = dr.read_data(ML_MOVIES)
ratings_data = dr.read_data(ML_RATINGS)
tags_data = dr.read_data(ML_TAGS)
users_data = dr.read_data(ML_USERS)
movie_actors_data = dr.read_data(MOVIE_ACTORS)
genome_data['tagid'] = genome_data['tagId']
del genome_data['tagId']
imdb_actor_data = imdb_actor_data.rename(columns={'id': 'actorid'})

'''
print "genome data \n", genome_data.head()
print "imdb actor data \n", imdb_actor_data.head()
print "movies data \n", movies_data.head()
print "ratings data \n", ratings_data.head()
print "tags data \n", tags_data.head()
print "users data \n", users_data.head()
print "movie actor data \n", movie_actors_data.head()
'''


def epoch_time(date_time):
    pattern = '%Y-%m-%d %H:%M:%S'
    return int(time.mktime(time.strptime(date_time, pattern)))



# print tags_data.sort_values(['movieid','tagid'])
tags_data['epoch'] = np.vectorize(epoch_time)(tags_data['timestamp'])
# del tags_data['timestamp']
max_epoch = max(tags_data['epoch'])
min_epoch = min(tags_data['epoch'])
tags_data['epoch'] = ((tags_data['epoch'] - min_epoch) / (max_epoch - min_epoch)) + 1


genres = movies_data['genres'].str.split('|').apply(Series, 1).stack()
genres.index = genres.index.droplevel(-1)
genres.name = 'genre'
del movies_data['genres']
movies_data = movies_data.join(genres)
