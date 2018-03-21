from __future__ import division
import numpy as np
import pandas as pd
import dataframes as df
from plotly.figure_factory._distplot import scipy
import math
import sys
import os
import random


def __compute_tf_tag_weight__(movie_id):
    records = __actorTagTable[__actorTagTable['movieid'] == movie_id]
    tag_list = {}
    count = len(records)
    for index, row in records.iterrows():
        if row['tag'] not in tag_list.keys():
            tag_list[row['tag']] = row.weight
        else:
            tag_list[row['tag']] += row.weight

    total_tag_weight = sum(tag_list.values())
    for tag, weight in tag_list.iteritems():
        tf_weight = (weight / total_tag_weight)
        tag_list[tag] = tf_weight / count

    return tag_list


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


def __get_movie_tag_vector__(movie_id):
    movie_tag_vector = {}
    tags = df.genome_data['tag'].unique().tolist()
    for tag in tags:
        record = __movieModelTable[(__movieModelTable['tag'] == tag) & (__movieModelTable['movieid'] == movie_id)]
        tf_idf_value = 0
        if len(record) > 0:
            tf_idf_value = record.iloc[0]['tfidfweight']
        movie_tag_vector[tag] = tf_idf_value

    return movie_tag_vector


# Apply SVD on input matrix
def __step_by_step_svd__(matrix, dimension_count):
    total_dimensions = len(matrix[0])
    u, s, v = np.linalg.svd(matrix)
    eig_vecs = v.T
    eig_vals = s
    eig_vals_copy = eig_vals
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    if dimension_count is None:
        tot = sum(eig_vals)
        var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        index = 0
        for exp_var in cum_var_exp:
            if exp_var.real > 90:
                break
            index += 1

        dimension_count = index

    top_latent_index = []
    for i in range(0, dimension_count):
        top_latent_index.append(eig_vals_copy.tolist().index(eig_pairs[i][0]))

    params = []
    for x in range(0, total_dimensions):
        temp = []
        for y in range(0, dimension_count):
            eig_vecs_array = eig_pairs[y][1].reshape(total_dimensions, 1)
            temp.append(eig_vecs_array[x][0])
        params.append(temp)

    return {"top_latent_index": top_latent_index, "Y": matrix.dot(params)}


# Apply PCA on input matrix
def __step_by_step_pca__(matrix, dimension_count):
    # Find co variance on the input matrix
    cov_mat = np.cov(matrix.T)
    total_dimensions = len(matrix[0])
    # Eigen Decompose the co variance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_vals_copy = eig_vals
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    # Sort Eigen pairs based on Eigenvalue
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Based on explained variance decide on no of dimensions to keep
    # If no of dimensions are not specified threshold is set to 90 (capture 90% of original variance)
    if dimension_count is None:
        tot = sum(eig_vals)
        var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        index = 0
        for exp_var in cum_var_exp:
            if exp_var.real > 90:
                break
            index += 1

        dimension_count = index

    top_latent_index = []
    for i in range(0, dimension_count):
        top_latent_index.append(eig_vals_copy.tolist().index(eig_pairs[i][0]))

    # Retrieve the top latent semantics
    params = []
    for x in range(0, total_dimensions):
        temp = []
        for y in range(0, dimension_count):
            eig_vecs_array = eig_pairs[y][1].reshape(total_dimensions, 1)
            temp.append(eig_vecs_array[x][0])
        params.append(temp)

    return {"top_latent_index": top_latent_index, "Y": matrix.dot(params)}


# Returns the input matrix with movies as objects and tags as features
def __get_movie_tag_matrix__():
    matrix = []
    tags = df.genome_data['tag'].unique().tolist()
    random.shuffle(tags)
    movies = __movieModelTable['movieid'].unique().tolist()
    for movie_id in movies:
        movie_tag_vector = __get_movie_tag_vector__(movie_id)
        matrix.append(movie_tag_vector.values())

    return {'row_values': movies, 'column_values': tags, 'matrix': np.array(matrix)}


# Returns similar actor who have not acted on the given movie
def __process_actors_not_similarity__(pca, movies, tags, movie_id):
    Y_new = pca['Y']
    top_latents = pca['top_latent_index']
    dist = {}

    print('\nReducing no of dimensions from ' + str(len(tags)) + ' to ' + str(len(top_latents)))

    movie_name = __actorTable[__actorTable['movieid'] == movie_id].iloc[0]['moviename']
    print('\nTop 10 related actors who have not acted in ' + str(movie_name) + '\n')

    for mid in movies:
        if mid != movie_id:
            dist[mid] = scipy.spatial.distance.euclidean(Y_new[movies.index(movie_id)], Y_new[movies.index(mid)])

    count = 0
    related_actors = []
    actors_in_given_movie = __actorTable[__actorTable['movieid'] == movie_id]['actorid'].unique().tolist()
    for key, value in sorted(dist.items(), key=lambda x: x[1]):
        if len(related_actors) < 10:
            actor_df = __actorTable[__actorTable['movieid'] == key]
            actor_id = actor_df.ix[actor_df['actor_movie_rank'].idxmin()]['actorid']
            if actor_id not in actors_in_given_movie and actor_id not in related_actors:
                related_actors.append(actor_id)
                actor_name = df.imdb_actor_data[df.imdb_actor_data['actorid'] == actor_id]['name'].iloc[0]
                print(str(actor_name) + ' (' + str(actor_id) + ') ' + str(value))
        else:
            break
        count += 1


def get_movie_similarity(movie_id):
    matrix = __get_movie_tag_matrix__()
    movies = matrix['row_values']
    tags = matrix['column_values']

    pca = __step_by_step_pca__(matrix['matrix'], None)
    __process_actors_not_similarity__(pca, movies, tags, movie_id)

    pca = __step_by_step_pca__(matrix['matrix'], 5)
    __process_actors_not_similarity__(pca, movies, tags, movie_id)

movie_id = ''
if len(sys.argv) > 1:
    movie_id = int(sys.argv[1])

__current_directory = os.getcwd()
__output_folder = __current_directory + '/Output'

__actorTable = pd.merge(df.movie_actors_data, df.movies_data, on='movieid')
__tagTable = pd.merge(df.tags_data, df.genome_data, on='tagid')
__actorTagTable = pd.merge(__actorTable, __tagTable, on='movieid')

actor_model_file_name = __output_folder + '/MovieModelTFIDF.csv'
if os.path.isfile(actor_model_file_name):
    __movieModelTable = pd.read_csv(actor_model_file_name)
else:
    process_movie_model()
    __movieModelTable = pd.read_csv(actor_model_file_name)

if len(sys.argv) > 1 and 'task1d.py' == os.path.basename(sys.argv[0]):
    get_movie_similarity(movie_id)
