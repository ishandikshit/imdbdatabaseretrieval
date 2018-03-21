import dataframes as df
import pandas as pd
import numpy as np
import math
import sys
import os

actorid = ''
model = ''
if len(sys.argv) > 2:
    actorid = int(sys.argv[1])
    model = sys.argv[2]

__current_directory = os.getcwd()
__output_folder = __current_directory + '/Output'

# tf = count of the given tag for the actor / total number of tags for that actor
# tfweighted = epoch * tf / rank of actor in the movie in which the tag has occured
# now we have multiple tags repeated for the same actor but different timestamps
# should we take average of the values of tf? makes more sense to take average if one tag is repeated more than once


# idf = log(total number of actors/number of actors where this tag has appeared)


# movie actor data for given actor
def get_actor_vector(actorid, model):

    # given actor's data
    movies_actors = df.movie_actors_data.loc[df.movie_actors_data['actorid'] == actorid]

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
    actors_tags_counts = actors_tags.merge(tagid_count_in_corpus, on='tagid', how='left')

    # weighted tf calulcation
    actors_tags_counts['tf'] = actors_tags_counts['total_appearances']/len(actors_tags_counts)
    actors_tags_counts['tf'] *= actors_tags_counts['epoch']/actors_tags_counts['actor_movie_rank']

    del actors_tags_counts['actor_movie_rank']
    del actors_tags_counts['total_appearances']
    del actors_tags_counts['epoch']

    actors_tags_counts = actors_tags_counts.groupby('tagid').mean().reset_index()
    actors_tags_counts = actors_tags_counts.sort_values('tf', ascending=[False])
    tags_names = df.genome_data.loc[df.genome_data['tagid'].isin(actors_tags_counts['tagid'])]
    actors_tags_counts = actors_tags_counts.merge(tags_names, on='tagid', how='left')

    # tf is done. return tf if model is tf
    if model=='tf':
        output = ''
        for data in range(len(actors_tags_counts)):
            output += " <{} {}>".format(actors_tags_counts['tag'].iloc[data], actors_tags_counts[model].iloc[data])
        return output

    elif model == 'tfidf':
        # now idf = total number of actors / actors for which this tag has appeared
        total_actors = len(df.movie_actors_data.groupby('actorid').size())

        tags = df.tags_data.loc[df.tags_data['tagid'].isin(actors_tags_counts['tagid'])]

        tag_actor_count = tags.merge(df.movie_actors_data, on=['movieid'], how='left') \
            .groupby(['tagid', 'actorid']).size().reset_index() \
            .groupby('tagid').size().reset_index()

        # print tag_actor_count
        actors_tags_counts = actors_tags_counts.merge(tag_actor_count, on='tagid')
        actors_tags_counts['idf'] = np.vectorize(math.log)(total_actors / actors_tags_counts[0])
        actors_tags_counts['tfidf'] = actors_tags_counts['tf'] * actors_tags_counts['idf']
        del actors_tags_counts[0]
        del actors_tags_counts['tf']
        del actors_tags_counts['idf']
        del actors_tags_counts['tagid']
        actors_tags_counts = actors_tags_counts.sort_values('tfidf', ascending=[False])

        all_tag_list = []
        for data in range(len(actors_tags_counts)):
            tag = actors_tags_counts['tag'].iloc[data]
            tf_idf = actors_tags_counts[model].iloc[data]
            all_tag_list.append({'actorid': actorid, 'tag': tag, 'tfidfweight': tf_idf})
            # print("<{} {}>".format(tag, tf_idf))

        return all_tag_list


def for_all_actors(model):
    s = ''
    for index, row in df.movie_actors_data.groupby('actorid').size().reset_index().iterrows():
        t = (str(row['actorid']), str(get_actor_vector(row['actorid'], model)))
        print t
        s += ' '.join(t)
        s += '\n'
    return s


def get_tf_idf_for_all_actors():
    tf_idf_list = []
    for index, row in df.movie_actors_data.groupby('actorid').size().reset_index().iterrows():
        tf_idf_data = get_actor_vector(row['actorid'], 'tfidf')
        if tf_idf_data is not None:
            for entry in tf_idf_data:
                entry_dict = {}
                for key, value in entry.iteritems():
                    entry_dict[key] = value
                tf_idf_list.append(entry_dict)
    tf_idf_data_frame = pd.DataFrame(tf_idf_list)
    tf_idf_data_frame.to_csv(__output_folder + '/ActorModelTFIDF.csv', index=False, encoding='utf-8')
    return tf_idf_list


if actorid == '' or model == '':
    """
    f = open('output/allactors_tf.txt', 'w')
    f.write(for_all_actors('tf'))
    f.close()

    f = open('output/allactors_tfidf.txt', 'w')
    f.write(for_all_actors('tfidf'))
    f.close()
    """
    pass
else:
    get_actor_vector(actorid, model)