import dataframes as df
import numpy as np
import math
import sys

user = ''
model = ''

if len(sys.argv) > 1:
    user = int(sys.argv[1])
    model = sys.argv[2]


# tf = count of the given tag for the genre / total number of tags for that genre
# tfweighted = epoch * tf in the movie in which the tag has occured
# now we have multiple tags repeated for the same actor but different timestamps
# should we take average of the values of tf? makes more sense to take average if one tag is repeated more than once

# idf = log(total number of actors/number of actors where this tag has appeared)


def get_users_tags():
    ratings_movies = df.ratings_data.groupby(
        ['movieid', 'userid']).size().reset_index()
    tags_movies = df.tags_data.groupby(['movieid', 'userid']).size().reset_index()
    all_data = ratings_movies.append(tags_movies).groupby(['movieid', 'userid']).size().reset_index()
    del all_data[0]
    userid_tagid = all_data.merge(df.tags_data, on='movieid', how='left').dropna().groupby(
        ['userid_x', 'tagid']).size().reset_index().groupby(['tagid']).size().reset_index()
    return userid_tagid

global user_tags
user_tags = get_users_tags()


def get_user_vector(user, model):
    tags = df.tags_data.loc[df.tags_data['userid'] == user]
    ratings = df.ratings_data.loc[df.ratings_data['userid'] == user]
    movies = tags['movieid'].append(ratings['movieid'])
    tags = df.tags_data.loc[df.tags_data['movieid'].isin(movies)]
    del tags['userid']
    movies_tags = tags['movieid']
    del tags['movieid']

    tags_count = tags.groupby('tagid').size().reset_index()
    tags = tags.merge(tags_count, on='tagid')
    tags['tag_count'] = tags[0]
    del tags[0]

    total_tags = len(tags)
    tags['tf'] = (tags['tag_count']/total_tags)*tags['epoch']
    del tags['epoch']
    del tags['tag_count']
    tags = tags.groupby('tagid').mean().reset_index()
    tags_names = df.genome_data.loc[df.genome_data['tagid'].isin(tags['tagid'])]
    tags = tags.merge(tags_names, on='tagid', how='left')
    tags = tags.sort_values('tf', ascending=False)
    if model == 'tf':
        pass
    elif model == 'tfidf':
        total_users = len(df.ratings_data['userid'].append(df.tags_data['userid']).unique())
        tags = tags.merge(user_tags, on='tagid', how='left')
        tags['idf'] = np.vectorize(math.log)(total_users/tags[0])
        del tags[0]
        tags['tfidf'] = tags['tf'] * tags['idf']
        tags = tags.sort_values('tfidf', ascending=False)
    output = ''
    for data in range(len(tags)):
        output += " <{} {}>".format(tags['tag'].iloc[data], tags[model].iloc[data])
    return output


def for_all_users(model):
    s = ''
    for index, row in df.tags_data.groupby('userid').size().reset_index().iterrows():
        t = (str(row['userid']), str(get_user_vector(row['userid'], model)))
        print t
        s += ' '.join(t)
        s += '\n'
    return s


if user == '' or model == '':
#    f = open('allusers_tf.txt', 'w')
#    f.write(for_all_users('tf'))
#    f.close()

#    f = open('allusers_tfidf.txt', 'w')
#    f.write(for_all_users('tfidf'))
#    f.close()
    pass
else:
    print get_user_vector(user, model)