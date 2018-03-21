import dataframes as df
import numpy as np
import pandas as pd
import math


def zero_case_resolution(args):
    tagid = args[0]
    a = args[1]
    b = args[2]
    c = args[3]
    d = args[4]
    rside = args[5]
    if d > 0:
        return pd.Series({'tagid': tagid, 'l_num_num': a, 'l_num_denom': b, 'l_denom_num': c,
                          'l_denom_denom': d, 'r_side': rside})
    if d == 0:
        return pd.Series({'tagid': tagid, 'l_num_num': a + 0.5, 'l_num_denom': b + 0.5,
                          'l_denom_num': c + 0.5, 'l_denom_denom': d + 0.5, 'r_side': rside})


def get_p_diff_2(genre1, genre2):
    movies_in_genre1_genre2 = df.movies_data.loc[df.movies_data['genre'].isin([genre1, genre2])]
    del movies_in_genre1_genre2['moviename']

    # CALCULATING M
    ###################################################################################################
    # getting total unique movies for both genres
    M = len(movies_in_genre1_genre2['movieid'].unique())

    # CALCULATING R
    ###################################################################################################
    # getting total unique movies for genre1
    R = len(movies_in_genre1_genre2.loc[movies_in_genre1_genre2['genre'] == genre2]['movieid'].unique())

    if M == 0:
        return None
    if R == 0:
        return None
    if M == R:
        return None

    # CALCULATING r1j
    ###################################################################################################
    # dropping movies where there are no tags
    movies_in_genre2 = movies_in_genre1_genre2.loc[movies_in_genre1_genre2['genre'] == genre2]
    tags_movies_for_genre2 = movies_in_genre2.merge(df.tags_data, on='movieid', how='left').dropna()
    del tags_movies_for_genre2['userid']
    del tags_movies_for_genre2['epoch']
    del tags_movies_for_genre2['genre']
    movie_count_for_tags_genre1 = tags_movies_for_genre2.groupby(['tagid', 'movieid']).size().reset_index().groupby(
        ['tagid']).size().reset_index()
    movie_count_for_tags_genre1['r'] = R - movie_count_for_tags_genre1[0]
    del movie_count_for_tags_genre1[0]
    # print movie_count_for_tags
    ####################################################################################################

    # CALCULATING m1j
    ###################################################################################################
    # dropping movies where there are no tags
    movies_in_genre1_genre2 = movies_in_genre1_genre2.groupby('movieid').size().reset_index()
    del movies_in_genre1_genre2[0]
    tags_movies_for_genre1_genre2 = movies_in_genre1_genre2.merge(df.tags_data, on='movieid', how='left').dropna()
    del tags_movies_for_genre1_genre2['epoch']
    del tags_movies_for_genre1_genre2['userid']
    movie_count_tags_for_genre1_genre2 = tags_movies_for_genre1_genre2.groupby(
        ['movieid', 'tagid']).size().reset_index().groupby(['tagid']).size().reset_index()
    movie_count_tags_for_genre1_genre2['m'] = M - movie_count_tags_for_genre1_genre2[0]
    del movie_count_tags_for_genre1_genre2[0]
    # print movie_count_tags_for_genre1_genre2
    ####################################################################################################

    weights_table_for_g1 = movie_count_for_tags_genre1.merge(movie_count_tags_for_genre1_genre2, on='tagid', how='left')

    if len(weights_table_for_g1) == 0:
        return None

    weights_table_for_g1['l_num_num'] = weights_table_for_g1['r']
    weights_table_for_g1['l_num_denom'] = R - weights_table_for_g1['r']
    weights_table_for_g1['l_denom_num'] = (weights_table_for_g1['m'] - weights_table_for_g1['r'])
    weights_table_for_g1['l_denom_denom'] = (M - weights_table_for_g1['m'] - R + weights_table_for_g1['r'])
    weights_table_for_g1['r_side'] = np.abs(
        (weights_table_for_g1['r'] / R) - ((weights_table_for_g1['m'] - weights_table_for_g1['r']) / (M - R)))

    del weights_table_for_g1['r']
    del weights_table_for_g1['m']

    # print weights_table_for_g1
    weights_table_for_g1 = weights_table_for_g1.apply(zero_case_resolution, axis=1)

    weights_table_for_g1['l_num'] = weights_table_for_g1['l_num_num'] / weights_table_for_g1['l_num_denom']
    weights_table_for_g1['l_denom'] = weights_table_for_g1['l_denom_num'] / weights_table_for_g1['l_denom_denom']
    weights_table_for_g1['lside'] = weights_table_for_g1['l_num'] / weights_table_for_g1['l_denom']
    weights_table_for_g1['log'] = np.log(weights_table_for_g1['lside'])
    weights_table_for_g1['weight'] = weights_table_for_g1['log'] * weights_table_for_g1['r_side']

    del weights_table_for_g1['l_num_num']
    del weights_table_for_g1['l_num_denom']
    del weights_table_for_g1['l_denom_num']
    del weights_table_for_g1['l_denom_denom']
    del weights_table_for_g1['l_num']
    del weights_table_for_g1['l_denom']

    minn = min(weights_table_for_g1['weight'])
    maxx = max(weights_table_for_g1['weight'])
    weights_table_for_g1['norm_weight'] = (weights_table_for_g1['weight'] - minn) / (maxx - minn) + 1
    del weights_table_for_g1['r_side']
    del weights_table_for_g1['log']
    del weights_table_for_g1['lside']
    weights_table_for_g1 = weights_table_for_g1.merge(df.genome_data, on='tagid', how='left')
    del weights_table_for_g1['weight']
    del weights_table_for_g1['tagid']
    return weights_table_for_g1


def p_distance(genre1, genre2):
    v1 = get_p_diff_2(genre1, genre2)
    v2 = get_p_diff_2(genre2, genre1)
    if v1 is None and v2 is None:
        return None
    elif v1 is None:
        v2['diff'] = (v2['norm_weight'] - v2['norm_weight']) ** 2
        return math.sqrt(v2['diff'].sum())
    elif v2 is None:
        v1['diff'] = (v1['norm_weight'] - v1['norm_weight']) ** 2
        return math.sqrt(v1['diff'].sum())
    else:
        final = v2.merge(v1, on='tag', how='outer').fillna(0)
        final['diff'] = (final['norm_weight_x'] - final['norm_weight_y']) ** 2
        print final.sort_values('tag')
        return math.sqrt(final['diff'].sum())


def for_all_genres():
    final_data = pd.DataFrame(columns = ['genre1','genre2','diff'])
    i = 0
    for index, row in df.movies_data.groupby('genre').size().reset_index().iterrows():
        for index1, row1 in df.movies_data.groupby('genre').size().reset_index().iterrows():
            if row['genre'] != row1['genre']:
                diff = p_distance(row['genre'], row1['genre'])
                if diff is not None:
                    final_data.loc[i] = [row['genre'], row1['genre'], diff]
                    i += 1

    final_data = final_data.dropna()
    final_data = final_data [final_data['diff'] != 0]

    minn = min(final_data['diff'])
    maxx = max(final_data['diff'])
    print maxx, minn
    final_data['norm_diff'] = (((final_data['diff'] - minn) / (maxx - minn)) + 1)
    final_data.to_csv('output/pdiff2.csv')
