import dataframes as df
import numpy as np
import math
import pandas as pd


def get_genre_vector(genre1, genre2):
    movies = df.movies_data.loc[df.movies_data['genre'] == genre1]
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
    movies_tags = movies_tags.groupby('tagid').mean().reset_index()
    movies_data_for_both_genres = df.movies_data.loc[df.movies_data['genre'].isin([genre1, genre2])]
    movies_for_both_genres = movies_data_for_both_genres.groupby('movieid').size().reset_index()
    movies_for_both_genres = movies_for_both_genres.merge(df.movies_data, on='movieid', how='left')

    # this is the only part that has changed from task 2 here in this calculation of
    # mapping of genre onto the tag space
    # instead of considering the entire dataset as corpus
    # only movies of the two given genres are considered as corpus
    genres_for_both_genres = movies_for_both_genres.groupby('genre').size().reset_index()
    total_genres = len(genres_for_both_genres)
    tags_genres_data = df.tags_data.loc[df.tags_data['tagid'].isin(movies_tags['tagid'])].merge(movies_for_both_genres, on='movieid', how='left')
    tag_genre_count = tags_genres_data.groupby(['tagid', 'genre']).size().reset_index().groupby('tagid').size().reset_index()

    movies_tags = tag_genre_count.merge(movies_tags)

    movies_tags['count_genres_tagid'] = tag_genre_count[0]
    del movies_tags[0]
    if len(movies_tags) == 0:
        return pd.DataFrame()

    movies_tags['idf'] = np.vectorize(math.log)(total_genres/movies_tags['count_genres_tagid'])
    movies_tags['tfidf'] = movies_tags['idf']*movies_tags['tf']
    movies_tags = movies_tags.sort_values('tfidf', ascending=False)
    tags_names = df.genome_data.loc[df.genome_data['tagid'].isin(movies_tags['tagid'])]
    movies_tags = movies_tags.merge(tags_names, on='tagid', how='left')
    del movies_tags['tf']
    del movies_tags['idf']
    del movies_tags['count_genres_tagid']
    del movies_tags['tagid']
    return movies_tags


def get_difference_in_genres(genre1, genre2):
    vector1 = get_genre_vector(genre1, genre2)
    vector2 = get_genre_vector(genre2, genre1)
    if len(vector1) and len(vector2) > 0:
        difference = vector1.merge(vector2, on='tag', how='outer').fillna(0)
        difference['diff'] = (difference['tfidf_y'] - difference['tfidf_x'])**2
        print difference.sort_values('tag')
        return np.sqrt(difference['diff'].sum())
    elif len(vector1) > 0:
        print vector1
        return np.sqrt(vector1['tfidf'].sum())
    elif len(vector2) > 0:
        print vector1
        return np.sqrt(vector2['tfidf'].sum())
    else:
        return None


def for_all_genres():
    final_data = pd.DataFrame(columns = ['genre1','genre2','diff'])
    i = 0
    for index, row in df.movies_data.groupby('genre').size().reset_index().iterrows():
        for index1, row1 in df.movies_data.groupby('genre').size().reset_index().iterrows():
            if row['genre'] != row1['genre']:
                diff = get_difference_in_genres(row['genre'], row1['genre'])
                if diff is not None:
                    final_data.loc[i] = [row['genre'], row1['genre'], diff]
                    i += 1

    final_data = final_data.dropna()
    final_data = final_data [final_data['diff'] != 0]
    minn = min(final_data['diff'])
    maxx = max(final_data['diff'])
    final_data['norm_diff'] = (((final_data['diff'] - minn) / (maxx - minn)) + 1)
    final_data.to_csv('output/tfidfdiff.csv')
