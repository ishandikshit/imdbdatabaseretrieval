import dataframes as df
from Data import reader as dr
import numpy as np
import math
import sys
import pandas as pd

from sklearn import decomposition
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Tree structure referred from https://www.youtube.com/watch?v=LDRbO9a6XPU&t=187s

# tf = count of the given tag for the genre / total number of tags for that genre
# tfweighted = epoch * tf in the movie in which the tag has occured
# now we have multiple tags repeated for the same actor but different timestamps
# should we take average of the values of tf? makes more sense to take average if one tag is repeated more than once

# idf = log(total number of actors/number of actors where this tag has appeared)


#PHASE-3
def unique_vals(rows, col):
    return set([row[col] for row in rows])

def dict_classes(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def isnum(value):
    return isinstance(value, int) or isinstance(value, float)

def get_movie_vector(movieid, model):
    output = ''
    movies = df.movies_data.loc[df.movies_data['movieid'] == movieid]
    tags = df.tags_data.loc[df.tags_data['movieid'].isin(movies['movieid'])]
    movies_tags = movies.merge(tags, on='movieid')
    total_tags_for_movie = len(movies_tags)
    movies_tags = movies_tags.merge(movies_tags.groupby('tagid').size().reset_index())
    movies_tags['tf'] = (movies_tags[0]/total_tags_for_movie) * movies_tags['epoch']
    tags_names = df.genome_data.loc[df.genome_data['tagid'].isin(movies_tags['tagid'])]
    movies_tags = movies_tags.merge(tags_names, on='tagid', how='left')
    structured_output = {}
    movies_tags = movies_tags.groupby('tag').mean().reset_index()
    if model == 'tf':
        movies_tags = movies_tags.sort_values('tf', ascending=False)
        for data in range(len(movies_tags)):
            output += " <{} {}>".format(movies_tags['tag'].iloc[data], movies_tags[model].iloc[data])
            structured_output[movies_tags['tag'].iloc[data]] = movies_tags[model].iloc[data]
        return structured_output

def get_total_movies():
    a =[]
    ml_movies = df.movies_data
    movies = ml_movies.movieid
    a= list(set(movies))
    return a

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if isnum(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if isnum(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

class Leaf:
    def __init__(self, rows):
        self.predictions = dict_classes(rows)

class Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def split(rows, question):
    a=[]
    b=[]
    for row in rows:
        if question.match(row):
            a.append(row)
        else:
            b.append(row)
    return a,b

def get_gini_impurity(rows):
    counts = dict_classes(rows)
    impurity=1
    for x in counts:
        prob_x = counts[x] / float(len(rows))
        impurity -= prob_x**2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * get_gini_impurity(left) - (1 - p) * get_gini_impurity(right)

#gotta find the best split
def find_best_split(rows):
    gain_max = 0
    final_q = None
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            a, b = split(rows, question)
            if len(a) == 0 or len(b) == 0:
                continue
            if info_gain(a, b, get_gini_impurity(rows)) >= gain_max:
                gain_max, final_q = info_gain(a, b, get_gini_impurity(rows)), question
    return gain_max, final_q

def build_tree(rows):
    gain, question = find_best_split(rows)
    if gain < 0.01:
        return Leaf(rows)
    a, b = split(rows, question)
    positive_subtree = build_tree(a)
    negitive_subtree = build_tree(b)
    return Node(question, positive_subtree, negitive_subtree)

def classify_movie(data, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(data):
        return classify_movie(data, node.true_branch)
    else:
        return classify_movie(data, node.false_branch)

'''MAIN PROGRAM'''
a = []
df_movie = pd.DataFrame()
i=0
import os.path
cache=False
if os.path.exists("Data/movie_tag_matrix.csv"):
    cache = True

for movieid in get_total_movies():
    a.append(movieid)
    if not cache:
        vector = get_movie_vector(movieid, "tf")
        print i
        df_movie = df_movie.append(vector, ignore_index=True)
        i=i+1
if not cache:
    df_movie = df_movie.fillna(0)

    pca_movie = decomposition.PCA(n_components=4)
    # lda_movie = LatentDirichletAllocation(n_components=4, max_iter=10, learning_method='online', learning_offset=50., random_state=0)
    # movie_lda = lda_movie.fit_transform(df_movie)
    movie_lda = pca_movie.fit_transform(df_movie)
    np.savetxt('Data/movie_tag_matrix.csv', movie_lda, delimiter=",")

if cache:
    from numpy import genfromtxt
    movie_lda = genfromtxt('Data/movie_tag_matrix.csv', delimiter=',')

#Decision tree making process\

#taking out training data from user input
input_movie_list = raw_input("\nPlease enter movie ids\n")
input_movie_list = input_movie_list.split()
input_movie_list = map(int, input_movie_list)

input_label_list = raw_input("\nPlease enter respective tags\n")
input_label_list = input_label_list.split()
input_label_list = map(str, input_label_list)
vals = set(input_label_list)
vals = list(vals)
for i in range(0, len(input_label_list)):
    if input_label_list[i] in vals:
        input_label_list[i]= vals.index(input_label_list[i])

print input_label_list
training_set = []
for x in range(0, len(input_movie_list)):
    index_of_movie = a.index(input_movie_list[x])
    temp = np.append(movie_lda[index_of_movie], input_label_list[x])
    training_set.append(temp.tolist())

training_data = np.asarray(training_set)

print training_data

header = ["l1", "l2", "l3", "l4", "label"]

print dict_classes(training_data)

print find_best_split(training_data)

decision_tree=build_tree(training_data)

mlist = get_total_movies()
print mlist[:40]
while(True):
    input_target = raw_input("\nPlease enter movie id to predict\n")
    index_of_target = a.index(int(input_target))
    target = movie_lda[index_of_target]
    print "Target: "
    print target
    outp = classify_movie(target, decision_tree) 
    print outp
    for key in outp:
        print vals[int(key)]

