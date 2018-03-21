import dataframes as df
import numpy as np
import math
import sys

genre = ''
model = ''
if len(sys.argv) > 1:
    genre = sys.argv[1]
    model = sys.argv[2]



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
    movies_tags = movies_tags.groupby('tag').mean().reset_index()
    if model == 'tf':
        movies_tags = movies_tags.sort_values('tf', ascending=False)
        for data in range(len(movies_tags)):
            output += " <{} {}>".format(movies_tags['tag'].iloc[data], movies_tags[model].iloc[data])
        return output
    elif model == 'tfidf':
        movies_tags = movies_tags.groupby('tagid').mean().reset_index()

        total_genres = len(df.movies_data.groupby('genre').size())

        # number of genres where that particular tag appears
        tag_genre_count = df.tags_data.loc[df.tags_data['tagid'].isin(movies_tags['tagid'])].merge(df.movies_data, on=['movieid'], how='left').groupby(['tagid', 'genre']).size().reset_index().groupby('tagid').size().reset_index()

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
        return output


def for_all_genres(model):
    s = ''
    for index, row in df.movies_data.groupby('genre').size().reset_index().iterrows():
        t = (str(row['genre']), str(get_genre_vector(row['genre'], model)))
        s += ' '.join(t)
        s += '\n'
    return s

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
    print get_genre_vector(genre, model)

