import print_actor_vector as pav
import sklearn.decomposition as sd
import sklearn.cluster as sc
import numpy as np
import dataframes as df
import sys
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import distance


# to generate the actor vs tag matrix
# where each actor is one row and is represented in terms of tfidf of tags
# that has appeared for that particular actor
def __get_actor_tag_matrix__(consider_zero_tag_vectors=True):

    # getting tfidf tags for all vectors (this function was written in phase 1)
    actor_data = pav.get_tf_idf_for_all_actors()
    tag_vector = {}
    actor_vector = {}
    actor_list = []
    actor_count = 0
    tag_count = 0

    # generating a list of all actors present in the dataset to associate with the matrix
    # the index of the actorid is the same as the index/row number in the actor_tag_matrix
    for actor_datum in actor_data:
        if not actor_vector.has_key(actor_datum['actorid']):
            actor_vector[actor_datum['actorid']] = actor_count
            actor_list.append(actor_datum['actorid'])
            actor_count += 1
        if not tag_vector.has_key(actor_datum['tag']):
            tag_vector[actor_datum['tag']] = tag_count
            tag_count += 1

    # check if the user wants to ignore zero tag actors
    if consider_zero_tag_vectors:

        # if user wants to keep the actors with no tags
        # adds the user tags to the list
        for actor in df.movie_actors_data['actorid'].unique():
            if not actor_vector.has_key(actor):
                actor_vector[actor] = actor_count
                actor_list.append(actor)
                actor_count += 1

    actor_tag_matrix = [[[] for _ in range(tag_count)] for _ in range(len(actor_list))]

    # generating the actor_tag matrix from tag vectors received above
    for actor_datum in actor_data:
        actor_index = actor_vector.get(int(actor_datum['actorid']))
        tag_index = tag_vector.get(str(actor_datum['tag']))
        actor_tag_matrix[actor_index][tag_index] = float(actor_datum['tfidfweight'])

    # filling 0 for all tag values that were not present in the actor for the given row
    for row_index in range(len(actor_tag_matrix)):
        for column_index in range(len(actor_tag_matrix[row_index])):
            if not actor_tag_matrix[row_index][column_index]:
                actor_tag_matrix[row_index][column_index] = 0
    return actor_list, actor_tag_matrix


# calculates simple euclidean distance but returns 1/distance
# to convert the distance metric into a similarity metric
# 1 means most similar, 0 means most distant
def __get_actor_actor_euclidean_similarity__(actor1_tags, actor2_tags):
    sm = 0
    for tag in range(len(actor1_tags)):
        sm += pow(actor1_tags[tag] - actor2_tags[tag], 2)
    sm = pow(sm, 0.5)
    if sm == 0:
        return 1
    else:
        return 1 / (1 + sm)


def __get_similarity_from_distance_matrix__(distance_matrix):
    for index in range(len(distance_matrix)):
        for index_2 in range(len(distance_matrix[index])):
            distance_matrix[index][index_2] = 1/1 + (distance_matrix[index][index_2])
    return distance_matrix


# generates the actor-actor similarity matrix based on the distances between their tags
# this is capable of generating similarity matrix using a total of
# 4 different distance/similarity measures
# all distance measures consider either inverse or 1/measure to get similarity
# metric instead of distance measure
def get_actor_actor_similarity_matrix(similarity_measure=3, consider_zero_tag_vectors=True):
    actor_list, actor_tag_matrix = __get_actor_tag_matrix__(consider_zero_tag_vectors)
    actor_actor_similarity_matrix = [[[] for _ in range(len(actor_tag_matrix))]
                                     for _ in range(len(actor_tag_matrix))]
    if similarity_measure == 0:

        # inverse of manhattan distance matrix generation
        actor_actor_similarity_matrix = \
            __get_similarity_from_distance_matrix__(manhattan_distances(actor_tag_matrix))

    elif similarity_measure == 1:

        # cosine similarity matrix generation
        actor_actor_similarity_matrix = cosine_similarity(actor_tag_matrix)

    elif similarity_measure == 2:

        # inverse of cosine distance matrix generation
        actor_actor_similarity_matrix = \
            __get_similarity_from_distance_matrix__(cosine_distances(actor_tag_matrix))

    elif similarity_measure == 3:

        # mahalanobis distance calculation but the distance value is inverted i.e 1/value
        # once the distance is calculated and put in the similarity metric
        cov_matrix = np.cov(np.transpose(actor_tag_matrix))
        for actor_index in range(len(actor_tag_matrix)):
            for actor_index_2 in range(len(actor_tag_matrix)):
                dist = distance.mahalanobis(u=actor_tag_matrix[actor_index],
                                            v=actor_tag_matrix[actor_index_2],
                                            VI=cov_matrix)
                if dist == 0:
                    actor_actor_similarity_matrix[actor_index][actor_index_2] = 1
                else:
                    actor_actor_similarity_matrix[actor_index][actor_index_2] = 1/(1 + dist)

    else:

        # Euclidean distance measure
        # uses the distance function given above that returns reciprocal values of distance
        # making it mimic a similarity measure
        for actor_index in range(len(actor_tag_matrix)):
            for actor_index_2 in range(len(actor_tag_matrix)):
                dist = __get_actor_actor_euclidean_similarity__(
                    actor_tag_matrix[actor_index],
                    actor_tag_matrix[actor_index_2])
                actor_actor_similarity_matrix[actor_index][actor_index_2] = dist

    return actor_list, actor_actor_similarity_matrix


# performs SVD to get actor against latent semantic matrix
# using the actor-actor similarity metric generated above
def __get_actor_latent_matrix__(similarity_measure, consider_zero_tag_vectors=True):
    actor_list, actor_tag_matrix = \
        get_actor_actor_similarity_matrix(similarity_measure, consider_zero_tag_vectors)
    svd = sd.TruncatedSVD(n_components=3)
    svd.fit(actor_tag_matrix)
    semantics = []

    # generating the semantics in terms of actors and their importance
    # present in each of the calculated semantics
    for i in range(len(svd.components_)):
        semantics.append({})
        for actor in range(len(svd.components_[i])):
            semantics[i][actor_list[actor]] = svd.components_[i][actor]

    return semantics, actor_list, svd.transform(actor_tag_matrix)


# k means clustering is done to put the actors into 3 clusters based on
# their distances with respect to the latent semantics generated above.
def get_clusters(similarity_measure, consider_zero_tag_vectors=True):
    semantics, actor_list, latent_matrix = __get_actor_latent_matrix__(similarity_measure, consider_zero_tag_vectors)

    # scikit clustering is used for kmean clustering
    kmeans = sc.KMeans(n_clusters=3)
    kmeans.fit(latent_matrix)
    kmeans.transform(latent_matrix)
    print "Total number of Actors: ", len(latent_matrix)

    # separating clusters into 2-D arrays for easy representation later
    cluster1 = []
    cluster1_actors = []
    cluster2 = []
    cluster2_actors = []
    cluster3 = []
    cluster3_actors = []
    for label in range(len(kmeans.labels_)):
        if kmeans.labels_[label] == 0:
            cluster1.append(latent_matrix[label])
            cluster1_actors.append(actor_list[label])
        if kmeans.labels_[label] == 1:
            cluster2.append(latent_matrix[label])
            cluster2_actors.append(actor_list[label])
        if kmeans.labels_[label] == 2:
            cluster3.append(latent_matrix[label])
            cluster3_actors.append(actor_list[label])

    # clusters is a list of three clusters
    # each of these three clusters are a list of actorids
    clusters = [cluster1, cluster2, cluster3]

    # cluster_actors is a list of three clusters
    # each of these three clusters are a list of actor names
    cluster_actors = [cluster1_actors, cluster2_actors, cluster3_actors]
    return semantics, cluster_actors, clusters


# gets the names of actors according to their actorids
# and prints the actor's name present in data
def print_cluster_output(similarity_measure, consider_zero_tag_vectors=True):
    semantics, cluster_actors, clusters = get_clusters(similarity_measure, consider_zero_tag_vectors)
    actors = df.imdb_actor_data
    actor_names_clusters = []
    count = 0
    for semantic in semantics:
        count += 1
        print "latent semantic", count, "components", semantic

    for i in range(len(cluster_actors)):
        actor_names_clusters.append([])
        for actorid in cluster_actors[i]:
            actor_names_clusters[i].append(actors.loc[actors['actorid'] == actorid].iloc[0].iloc[1])
    print "Cluster1: ", actor_names_clusters[0]
    print "Cluster2: ", actor_names_clusters[1]
    print "Cluster3: ", actor_names_clusters[2]


if str(sys.argv[0]).find("task2a") > -1:
    a = raw_input("Please input 1-5 to choose the similarity/distance measure to use for finding similarity:\n"
                  "1: Manhattan Distance"
                  "\n2: Cosine Similarity"
                  "\n3: Cosine Distance"
                  "\n4: Mahalanobis Distance"
                  "\n5: Euclidean Distance"
                  "\nNote: Anything besides these will automatically consider Cosine Similarity\n"
                  "Enter here: ")
    try:
        a = int(a)
    except TypeError:
        a = 2
    except ValueError:
        a = 2
    finally:
        if a > 5:
            a = 2

    b = raw_input("\nDo you want to ignore actors with whom no tags were associated?\n"
                  "0 to ignore, anything else to include: ")
    if b == 0 or b == '0':
        b = False
    else:
        b = True
    print_cluster_output(a-1, b)
