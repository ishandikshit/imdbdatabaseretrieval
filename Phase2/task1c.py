from __future__ import division
import numpy as np
import pandas as pd
import print_actor_vector as AV
import dataframes as df
from plotly.figure_factory._distplot import scipy
import sys
import os
import random


def __get_tag_vector__(tag):
    tag_vector = {}
    actor_ids = df.movie_actors_data['actorid'].unique().tolist()
    for actor_id in actor_ids:
        record = __actorModelTable[(__actorModelTable['tag'] == tag) & (__actorModelTable['actorid'] == actor_id)]
        tf_idf_value = 0
        if len(record) > 0:
            tf_idf_value = record.iloc[0]['tfidfweight']
        tag_vector[actor_id] = tf_idf_value

    return tag_vector


def __get_actor_tag_vector__(actor_id):
    actor_tag_vector = {}
    tags = df.genome_data['tag'].unique().tolist()
    for tag in tags:
        record = __actorModelTable[(__actorModelTable['tag'] == tag) & (__actorModelTable['actorid'] == actor_id)]
        tf_idf_value = 0
        if len(record) > 0:
            tf_idf_value = record.iloc[0]['tfidfweight']
        actor_tag_vector[tag] = tf_idf_value

    return actor_tag_vector


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


# Returns the input matrix with actors as objects and tags as features
def __get_actor_tag_matrix__():
    matrix = []
    actors = __actorModelTable['actorid'].unique().tolist()
    tags = df.genome_data['tag'].unique().tolist()
    random.shuffle(tags)
    for aid in actors:
        actor_tag_vector = __get_actor_tag_vector__(aid)
        matrix.append(actor_tag_vector.values())

    return {'row_values': actors, 'column_values': tags, 'matrix': np.array(matrix)}


# Returns similar actor to the given actor
def __process_actor_similarity__(pca, actors, tags, actor_id):
    Y_new = pca['Y']
    top_latents = pca['top_latent_index']
    dist = {}

    print('\nReducing no of dimensions from ' + str(len(tags)) + ' to ' + str(len(top_latents)))

    actor_name = df.imdb_actor_data[df.imdb_actor_data['actorid'] == actor_id]['name'].iloc[0]
    print('\nTop 10 related actors to the given actor ' + str(actor_name) + '\n')

    for aid in actors:
        if aid != actor_id:
            dist[aid] = scipy.spatial.distance.euclidean(Y_new[actors.index(actor_id)], Y_new[actors.index(aid)])

    count = 0
    sorted_dict = sorted(dist.items(), key=lambda x: x[1])
    for key, value in sorted_dict:
        if count < 10:
            actor_name = df.imdb_actor_data[df.imdb_actor_data['actorid'] == key]['name'].iloc[0]
            print(str(actor_name) + ' (' + str(key) + ') ' + str(value))
        else:
            break
        count += 1


def get_similar_actor(actor_id):
    matrix = __get_actor_tag_matrix__()
    actors = matrix['row_values']
    tags = matrix['column_values']

    pca = __step_by_step_pca__(matrix['matrix'], None)
    __process_actor_similarity__(pca, actors, tags, actor_id)

    pca = __step_by_step_pca__(matrix['matrix'], 5)
    __process_actor_similarity__(pca, actors, tags, actor_id)

actor_id = ''
if len(sys.argv) > 1:
    actor_id = int(sys.argv[1])

__current_directory = os.getcwd()
__output_folder = __current_directory + '/Output'

__actorTable = pd.merge(df.movie_actors_data, df.movies_data, on='movieid')
__tagTable = pd.merge(df.tags_data, df.genome_data, on='tagid')
__actorTagTable = pd.merge(__actorTable, __tagTable, on='movieid')

actor_model_file_name = __output_folder + '/ActorModelTFIDF.csv'
if os.path.isfile(actor_model_file_name):
    __actorModelTable = pd.read_csv(actor_model_file_name)
else:
    AV.get_tf_idf_for_all_actors()
    __actorModelTable = pd.read_csv(actor_model_file_name)

if len(sys.argv) > 1 and 'task1c.py' == os.path.basename(sys.argv[0]):
    get_similar_actor(actor_id)
