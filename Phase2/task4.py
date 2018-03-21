from __future__ import division
import pandas as pd
import dataframes as df
import task1d as ans
import task1c as asp
from plotly.figure_factory._distplot import scipy
from scipy import spatial
import sys
import os

__current_directory = os.getcwd()
__output_folder = __current_directory + '/Output'

__actorTable = pd.merge(df.movie_actors_data, df.movies_data, on='movieid')
__tagTable = pd.merge(df.tags_data, df.genome_data, on='tagid')
__actorTagTable = pd.merge(__actorTable, __tagTable, on='movieid')

actor_model_file_name = __output_folder + '/MovieModelTFIDF.csv'
if os.path.isfile(actor_model_file_name):
    __movieModelTable = pd.read_csv(actor_model_file_name)
else:
    ans.process_movie_model()
    __movieModelTable = pd.read_csv(actor_model_file_name)

__user_movie_table = pd.merge(df.movies_data, df.tags_data, on='movieid')


def __get_movies_watch_by_user__(user_id):
    tags_record = __user_movie_table[__user_movie_table['userid'] == user_id]
    list1 = tags_record.movieid.unique().tolist()

    rating_record = df.ratings_data[df.ratings_data['userid'] == user_id]
    list2 = rating_record.movieid.unique().tolist()

    return list(set(list1) | set(list2))


# Print top 5 movies to watch for a user
def get_similar_movie(movie_list):
    # Get movie tag matrix
    movie_matrix = ans.__get_movie_tag_matrix__()
    movies = movie_matrix['row_values']

    # Get actor tag matrix
    actor_matrix = asp.__get_actor_tag_matrix__()
    actors = actor_matrix['row_values']

    # Do PCA on these two matrix
    pca_m = ans.__step_by_step_pca__(movie_matrix['matrix'], None)
    pca_a = asp.__step_by_step_pca__(actor_matrix['matrix'], None)

    similar_movies = {}
    movie_genres = []

    # Iterate over each movie id given as input
    for movie_id in movie_list:
        # get genres of movie id
        movie_genre = df.movies_data[df.movies_data['movieid'] == movie_id]['genre'].tolist()
        for genre in movie_genre:
            if genre not in movie_genres:
                movie_genres.append(genre)

        # Calculate Euclidean distance between given input movie point and all other movie point in reduced dimension
        Y_new = pca_m['Y']
        dist = {}
        for mid in movies:
            if mid != movie_id:
                dist[mid] = scipy.spatial.distance.euclidean(Y_new[movies.index(movie_id)], Y_new[movies.index(mid)])
                # dist[mid] = 1 - spatial.distance.cosine(Y_new[movies.index(movie_id)], Y_new[movies.index(mid)])

        # Retrieve top 10 related movies and maintain it in a dictionary
        movie_count = 0
        for key, value in sorted(dist.items(), key=lambda x: x[1]):
            if movie_count < 10:
                if key not in similar_movies:
                    similar_movies[key] = 1
                else:
                    similar_movies[key] += 1
                movie_count += 1

        # Get the actor with lowest rank for the given input movie id
        dist = {}
        actor_df = __actorTable[__actorTable['movieid'] == movie_id]
        actor_id = actor_df.ix[actor_df['actor_movie_rank'].idxmin()]['actorid']

        # Calculate Euclidean distance between this actor point and all other actor point in reduced dimension
        Y_new = pca_a['Y']
        for aid in actors:
            if aid != actor_id:
                dist[aid] = scipy.spatial.distance.euclidean(Y_new[actors.index(actor_id)], Y_new[actors.index(aid)])
                # dist[aid] = 1 - spatial.distance.cosine(Y_new[actors.index(actor_id)], Y_new[actors.index(aid)])

        # Retrieve top 10 related actors and maintain it in a dictionary
        actor_count = 0
        for key, value in sorted(dist.items(), key=lambda x: x[1]):
            if actor_count < 10:
                similar_actor_id = key
                movie_df = __actorTable[__actorTable['actorid'] == similar_actor_id]
                movie_id = movie_df.ix[movie_df['actor_movie_rank'].idxmin()]['movieid']
                if movie_id not in similar_movies:
                    similar_movies[movie_id] = 1
                else:
                    similar_movies[movie_id] += 1
                actor_count += 1

    final_data = []

    # Combine these 2 movie set and rank them based on count, rating and year
    for movie_id, count in similar_movies.items():
        # check if genre is valid and if so keep it
        valid_genre = False
        if movie_id not in movie_list:
            movie_genre = df.movies_data[df.movies_data['movieid'] == movie_id]['genre'].tolist()
            for genre in movie_genre:
                if genre in movie_genres:
                    valid_genre = True
            if valid_genre:
                movie_name = __actorTable[__actorTable['movieid'] == movie_id].iloc[0]['moviename']
                rating = df.ratings_data[df.ratings_data['movieid'] == movie_id].groupby('rating')['rating'].count().idxmax()
                year = __actorTable[__actorTable['movieid'] == movie_id].iloc[0]['year']
                # tags = ' | '.join(__actorTagTable[__actorTagTable['movieid'] == movie_id]['tag'].unique().tolist())
                # final_data.append({'movieid': movie_id, 'moviename': movie_name, 'count': count, 'rating': rating, 'year': year, 'tags': tags})
                final_data.append(
                    {'movieid': movie_id, 'moviename': movie_name, 'count': count, 'rating': rating, 'year': year })

    new_df = pd.DataFrame(final_data).sort_values(['count', 'rating', 'year'], ascending=[False, False, False])

    # If there is enough data retrieve top 5 rows and print
    if len(new_df) > 6:
        final = new_df[:5].reset_index()
        del final['index']
        print(final[['movieid', 'moviename', 'count', 'rating', 'year']])
    else:
        # get all genres from input movie and retrieve top rated movies in this genre to fill the rows
        final_m_data_list = []
        for genre in movie_genres:
            movies = df.movies_data[df.movies_data['genre'].map(lambda genres: genre in genres)]
            m_data = {}
            for movie_id in movies['movieid'].tolist():
                max_rating = df.ratings_data[df.ratings_data['movieid'] == movie_id].groupby('rating')['rating'].count().idxmax()
                m_data[movie_id] = max_rating
            t=[]
            t = sorted(m_data.items(), key=lambda x: x[1], reverse=True)
            temp = filter(lambda x: m_data[x] == t[0][1], m_data.keys())
            for t in temp:
                if t not in final_m_data_list:
                    final_m_data_list.append(t)
        ff = []
        for movie_id in final_m_data_list:
            rating = df.ratings_data[df.ratings_data['movieid'] == movie_id].groupby('rating')[
                'rating'].count().idxmax()
            year = __actorTable[__actorTable['movieid'] == movie_id].iloc[0]['year']
            movie_name = __actorTable[__actorTable['movieid'] == movie_id].iloc[0]['moviename']
            # tags = ' | '.join(__actorTagTable[__actorTagTable['movieid'] == movie_id]['tag'].unique().tolist())
            # ff.append({'movieid': movie_id, 'moviename': movie_name, 'rating': rating, 'year': year, 'count': 1, tags: tags})
            ff.append(
                {'movieid': movie_id, 'moviename': movie_name, 'rating': rating, 'year': year, 'count': 1})
        final_df = new_df.append(ff, ignore_index=True).sort_values(['count', 'rating', 'year'], ascending=[False, False, False])
        if len(final_df) > 5:
            final_df = final_df[:5].reset_index()
            del final_df['index']
            print(final_df[['movieid', 'moviename', 'count', 'rating', 'year']])
        else:
            final_df = final_df.reset_index()
            del final_df['index']
            print(final_df[['movieid', 'moviename', 'count', 'rating', 'year']])


inp = raw_input("\nEnter either 1 or 2\n1. Movie Id \n2. User Id\n")
if inp == '1':
    m_inp = raw_input('\nEnter list of movie Ids\n')
    m_list = []
    for x in m_inp.split():
        m_list.append(int(x))
    get_similar_movie(m_list)
else:
    u_inp = raw_input("\nEnter user id\n")
    get_similar_movie(__get_movies_watch_by_user__(int(u_inp)))
