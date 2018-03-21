from __future__ import division
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np
import dataframes as df
import scipy.spatial as scp
import scipy.sparse
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import os
import sys
import math
from scipy.sparse import coo_matrix


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


def is_tie_break(votes):
    pre_count = 0
    tie_break = False
    for label, count in sorted(votes.items(), key=lambda x: x[1], reverse=True):
        if pre_count == 0:
            pre_count = count
        else:
            if count == pre_count:
                tie_break = True
                break

    return tie_break


def get_majority_votes(index, user_labeled_movies, radius):
    votes = {}
    np_movies = np.array(movies)
    last_movie_index = similarity[index].argsort()[::1][-1]

    while len(votes) <= 0 or is_tie_break(votes):
        votes = {}
        similar_movie_index = np.where(similarity[index] < radius)
        similar_movie_index = np.delete(similar_movie_index, 0)
        neighbours = np_movies.take(similar_movie_index)
        seed_movies = user_labeled_movies.keys()
        seed_movies_in_range = neighbours[np.isin(neighbours, seed_movies)]

        if len(seed_movies_in_range) > 0:
            candidates = [user_labeled_movies[x] for x in seed_movies_in_range]
            unique_elements, counts_elements = np.unique(candidates, return_counts=True)
            for x in range(0, len(unique_elements)):
                votes[unique_elements[x]] = counts_elements[x]

        radius += 0.5
        if radius > similarity[index, last_movie_index]:
            break

    return votes


def classify_movies(input_movies):
    user_labeled_movies = {}
    for label, lebeled_movies in input_movies.items():
        for mid in lebeled_movies:
            user_labeled_movies[mid] = label
    labeled_points = {}
    for index in range(0,movie_matrix['matrix'].shape[0]):
        votes = get_majority_votes(index, user_labeled_movies, 1)
        label, count = sorted(votes.items(), key=lambda x: x[1], reverse=True)[0]
        movie_name = __actorTable[__actorTable['movieid'] == movies[index]].iloc[0]['moviename']
        if label not in labeled_points:
            labeled_points[label] = [{
                'movieid': movies[index],
                'moviename': movie_name,
                # 'count': count,
                'label': label
            }]
        else:
            labeled_points[label].append({
                'movieid': movies[index],
                'moviename': movie_name,
                # 'count': count,
                'label': label
            })

    for label, movie_details in labeled_points.items():
        print('\nList of movies in ' + label + ' and their minimum distance to seed movie\n')
        new_df = pd.DataFrame(movie_details)
        new_df.index = np.arange(1, len(new_df) + 1)
        with pd.option_context('display.max_rows', None, 'display.max_columns', 4):
            print(new_df[['label','movieid','moviename']])


def get_user_input():
    inp = raw_input("\nPress 1 to Enter label\nPress 2 to start process\n")
    if int(inp) == 1:
        lab = raw_input("\nEnter label\n")
        u_mov = raw_input("\nEnter list of movies separated by space\n")
        u_inp[lab] = []
        for u in u_mov.split(' '):
            u_inp[lab].append(int(u))
        get_user_input()
    else:
        classify_movies(u_inp)


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
u_inp = {}
# Get movie tag matrix
movie_matrix = __get_movie_tag_matrix__()
movies = movie_matrix['row_values']

similarity = euclidean_distances(movie_matrix['matrix'], movie_matrix['matrix'])
get_user_input()
# classify_movies([
#     {'action': [23,42,152,324]},
#     {'comedy': [330,391,410,631]},
#     {'drama': [630,643,661,676]}
# ])