from __future__ import division
import pandas as pd
import numpy as np
import dataframes as df
# import task1d as ans
import scipy.spatial as scp
import scipy.sparse
import math
from scipy.sparse import coo_matrix
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity


def __get_tf_idf_info__(movie_id):
    # given actor's data
    movies_actors = df.movie_actors_data.loc[df.movie_actors_data['movieid'] == movie_id]

    # tags data for given actor
    tags = df.tags_data
    tags = tags.loc[tags['movieid'].isin(movies_actors['movieid'])]
    del tags['userid']

    # actor tags merged data
    actors_tags = tags.merge(movies_actors, on='movieid', how='left')
    del actors_tags['actorid']
    del actors_tags['movieid']

    if len(actors_tags) == 0:
        return

    # calculate tag count for the given actor
    tagid_count_in_corpus = pd.DataFrame({'total_appearances': actors_tags.groupby(['tagid']).size()}).reset_index()
    movie_tags_counts = actors_tags.merge(tagid_count_in_corpus, on='tagid', how='left')

    # weighted tf calulcation
    movie_tags_counts['tf'] = movie_tags_counts['total_appearances'] / len(movie_tags_counts)
    movie_tags_counts['tf'] *= movie_tags_counts['epoch'] / movie_tags_counts['actor_movie_rank']

    del movie_tags_counts['actor_movie_rank']
    del movie_tags_counts['total_appearances']
    del movie_tags_counts['epoch']

    movie_tags_counts = movie_tags_counts.groupby('tagid').mean().reset_index()
    movie_tags_counts = movie_tags_counts.sort_values('tf', ascending=[False])
    tags_names = df.genome_data.loc[df.genome_data['tagid'].isin(movie_tags_counts['tagid'])]
    movie_tags_counts = movie_tags_counts.merge(tags_names, on='tagid', how='left')

    # now idf = total number of actors / actors for which this tag has appeared
    total_movies = len(__actorTable['movieid'].unique())

    tags = df.tags_data.loc[df.tags_data['tagid'].isin(movie_tags_counts['tagid'])]

    tag_actor_count = tags.merge(df.movie_actors_data, on=['movieid'], how='left') \
        .groupby(['tagid', 'actorid']).size().reset_index() \
        .groupby('tagid').size().reset_index()

    # print tag_actor_count
    movie_tags_counts = movie_tags_counts.merge(tag_actor_count, on='tagid')
    movie_tags_counts['idf'] = np.vectorize(math.log)(total_movies / movie_tags_counts[0])
    movie_tags_counts['tfidf'] = movie_tags_counts['tf'] * movie_tags_counts['idf']
    del movie_tags_counts[0]
    del movie_tags_counts['tf']
    del movie_tags_counts['idf']
    del movie_tags_counts['tagid']
    actors_tags_counts = movie_tags_counts.sort_values('tfidf', ascending=[False])

    all_tag_list = []
    for data in range(len(actors_tags_counts)):
        tag = actors_tags_counts['tag'].iloc[data]
        tf_idf = actors_tags_counts['tfidf'].iloc[data]
        all_tag_list.append({'movieid': movie_id, 'tag': tag, 'tfidfweight': tf_idf})
        # print("<{} {}>".format(tag, tf_idf))

    return all_tag_list


def process_movie_model():
    tf_idf_list = []
    for movieid in __actorTable['movieid'].unique().tolist():
        tf_idf_data = __get_tf_idf_info__(movieid)
        if tf_idf_data is not None:
            for entry in tf_idf_data:
                entry_dict = {}
                for key, value in entry.iteritems():
                    entry_dict[key] = value
                tf_idf_list.append(entry_dict)

    tf_idf_data_frame = pd.DataFrame(tf_idf_list)

    tf_idf_data_frame.to_csv(__output_folder + '/MovieModelTFIDF.csv', index=False, encoding='utf-8')


def __get_movies_watch_by_user__(user_id):
    tags_record = __user_movie_table[__user_movie_table['userid'] == user_id]
    list1 = tags_record.movieid.unique().tolist()

    rating_record = df.ratings_data[df.ratings_data['userid'] == user_id]
    list2 = rating_record.movieid.unique().tolist()

    return list(set(list1) | set(list2))


# Returns the input matrix with movies as objects and tags as features
def __get_movie_tag_matrix__():
    tags = df.genome_data['tag'].unique().tolist()
    movies = df.movies_data['movieid'].unique().tolist()
    row_len = len(movies)
    col_len = len(tags)
    matrix = np.zeros(shape=(row_len, col_len), dtype=np.float32)
    for row in __movieModelTable.as_matrix(['movieid', 'tag', 'tfidfweight']):
        if row[1] in tags:
            matrix[movies.index(row[0]), tags.index(row[1])] = float(row[2])

    return {'row_values': movies, 'column_values': tags, 'matrix': matrix}


def get_similarity_matrix(matrix):
    file_path = __output_folder + '/similarity.npz'
    if os.path.isfile(file_path):
        return scipy.sparse.load_npz(__output_folder + '/similarity.npz')
    else:
        X_std = StandardScaler().fit_transform(matrix)
        sklearn_pca = sklearnPCA()
        Y_sklearn = sklearn_pca.fit_transform(X_std)
        coo = coo_matrix(Y_sklearn)

        sim = cosine_similarity(coo.tocsr(), dense_output=False)
        c_mat = coo_matrix(sim, dtype=np.float16)
        scipy.sparse.save_npz(__output_folder + '/similarity.npz', c_mat)
        return sim


# Print top 5 movies to watch for a user
def get_similar_movie(movie_list, user_id):
    similar_movies = {}
    sorted_movie = {}
    # Iterate over each movie id given as input
    for movie_id in movie_list:
        rating = df.ratings_data[(df.ratings_data['userid'] == user_id) & (df.ratings_data['movieid'] == movie_id)]
        if len(rating) <= 0:
            rating = __user_movie_table[
                (__user_movie_table['userid'] == user_id) & (__user_movie_table['movieid'] == movie_id)]
        sorted_movie[movie_id] = rating.iloc[0].timestamp
        row = movies.index(movie_id)
        sim = similarity_csr[row].data.argsort()[::-1]
        sim_mov = np.array(list(np.array(movies)[sim[1:]]))
        sim_mov_not_watched = sim_mov[np.isin(sim_mov, movie_list, invert=True)]
        similar_movies[movie_id] = sim_mov_not_watched[:10]

    candidates = []
    for key, value in sorted(sorted_movie.items(), key=lambda x: x[1], reverse=True):
        candidates.append(key)

    recommended_movie = []
    index = 0
    for movie in candidates:
        if len(recommended_movie) < 5:
            sim_movie_id = similar_movies[movie][0]
            sim_value = similarity_csr[movies.index(candidates[index]), movies.index(sim_movie_id)]
            index += 1
            is_valid_reccomendation = True
            for x in recommended_movie:
                if x['movieid'] == sim_movie_id:
                    is_valid_reccomendation = False
            if is_valid_reccomendation:
                movie_name = __actorTable[__actorTable['movieid'] == sim_movie_id].iloc[0]['moviename']
                rating = df.ratings_data[df.ratings_data['movieid'] == sim_movie_id].groupby('rating')[
                    'rating'].count().idxmax()
                year = __actorTable[__actorTable['movieid'] == sim_movie_id].iloc[0]['year']
                recommended_movie.append(
                    {'movieid': sim_movie_id, 'moviename': movie_name, 'rating': rating, 'year': year, 'similarity': sim_value})

    if len(recommended_movie) < 5:
        for movie in candidates:
            index = 0
            for sim_movie_id in similar_movies[movie][1:]:
                if len(recommended_movie) < 5:
                    sim_value = similarity_csr[movies.index(candidates[index]), movies.index(sim_movie_id)]
                    index += 1
                    movie_name = __actorTable[__actorTable['movieid'] == sim_movie_id].iloc[0]['moviename']
                    rating = df.ratings_data[df.ratings_data['movieid'] == sim_movie_id].groupby('rating')[
                        'rating'].count().idxmax()
                    year = __actorTable[__actorTable['movieid'] == sim_movie_id].iloc[0]['year']
                    recommended_movie.append(
                        {'movieid': sim_movie_id, 'moviename': movie_name, 'rating': rating, 'year': year,
                         'similarity': sim_value})

    data_frame = pd.DataFrame(recommended_movie).sort_values(['similarity'], ascending=[False])
    data_frame.index = np.arange(1, len(data_frame) + 1)
    print(data_frame)

    feedback = raw_input("\nPlease enter the index of movies you dont like separated by space\n")
    user_feedback = map(int, feedback.split())
    for x in range(1, 6):
        if x in user_feedback:
            disliked_movie_id = data_frame['movieid'].iloc[int(x) - 1]
            disliked_movie_index = movies.index(disliked_movie_id)
            all_other_movie_index = similarity_csr[disliked_movie_index].data.argsort()[::-1][1:]
            similar_movies_to_disliked = all_other_movie_index[:10]
            for other_index in all_other_movie_index:
                similarity_csr[other_index, disliked_movie_index] = similarity_csr[
                                                                        other_index, disliked_movie_index] * 0.2
                similarity_csr[disliked_movie_index, other_index] = similarity_csr[
                                                                        disliked_movie_index, other_index] * 0.2
                for sim_index in similar_movies_to_disliked:
                    similarity_csr[sim_index, disliked_movie_index] = similarity_csr[
                                                                          sim_index, disliked_movie_index] * 0.2
                    similarity_csr[disliked_movie_index, sim_index] = similarity_csr[
                                                                          disliked_movie_index, sim_index] * 0.2
        else:
            liked_movie_id = data_frame['movieid'].iloc[int(x) - 1]
            liked_movie_index = movies.index(liked_movie_id)
            all_other_movie_index = similarity_csr[liked_movie_index].data.argsort()[::-1][1:]
            similar_movies_to_liked = all_other_movie_index[:10]
            for other_index in all_other_movie_index:
                similarity_csr[other_index, liked_movie_index] = similarity_csr[
                                                                        other_index, liked_movie_index] * 1.5
                similarity_csr[liked_movie_index, other_index] = similarity_csr[
                                                                        liked_movie_index, other_index] * 1.5
                for sim_index in similar_movies_to_liked:
                    similarity_csr[sim_index, liked_movie_index] = similarity_csr[
                                                                          sim_index, liked_movie_index] * 1.5
                    similarity_csr[liked_movie_index, sim_index] = similarity_csr[
                                                                          liked_movie_index, sim_index] * 1.5

    get_similar_movie(movie_list, user_id)

__current_directory = os.getcwd()
__output_folder = __current_directory + '/Output'

__actorTable = pd.merge(df.movie_actors_data, df.movies_data, on='movieid')
__tagTable = pd.merge(df.tags_data, df.genome_data, on='tagid')
__actorTagTable = pd.merge(__actorTable, __tagTable, on='movieid')

movie_model_file_name = __output_folder + '/MovieModelTFIDF.csv'
if os.path.isfile(movie_model_file_name):
    __movieModelTable = pd.read_csv(movie_model_file_name)
else:
    process_movie_model()
    __movieModelTable = pd.read_csv(movie_model_file_name)

__user_movie_table = pd.merge(df.movies_data, df.tags_data, on='movieid')
u_inp = raw_input("\nEnter user id\n")
# Get movie tag matrix
movie_matrix = __get_movie_tag_matrix__()
movies = movie_matrix['row_values']
# Get Similarity
similarity_coo = get_similarity_matrix(movie_matrix['matrix'])
similarity_csr = similarity_coo.tocsr()
get_similar_movie(__get_movies_watch_by_user__(int(u_inp)), int(u_inp))