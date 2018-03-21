import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sktensor import dtensor, cp_als

# Task 2d - Part 1

# invert indices and values of movieid, tagids for faster access
def invertDictionary(dt):
    return dict([[v, k] for k, v in dt.items()])

# read mltags, mlrating, mlmovies, movie-actor
mltagsFile = pd.read_csv('C:\\Users\\vaish\\Desktop\\Fall 2017\\Courses\\CSE515\\Project_Phase2\\mltags.csv')
mlratingsFile = pd.read_csv('C:\\Users\\vaish\\Desktop\\Fall 2017\\Courses\\CSE515\\Project_Phase2\\mlratings.csv')
genomeFile = pd.read_csv('C:\\Users\\vaish\\Desktop\\Fall 2017\\Courses\\CSE515\\Project_Phase2\\genome-tags.csv')
movieFile = pd.read_csv('C:\\Users\\vaish\\Desktop\\Fall 2017\\Courses\\CSE515\\Project_Phase2\\mlmovies.csv')

# Exgtract tag from tagid
genomeFile['tagid'] = genomeFile['tagId']
del genomeFile['tagId']
mltagsFile = pd.merge(mltagsFile,genomeFile,on='tagid')

# Extract movie from movieid
del movieFile['year']
del movieFile['genres']
mlratingsFile = pd.merge(mlratingsFile,movieFile,on='movieid')
mltagsFile = pd.merge(mltagsFile,movieFile,on='movieid')
# deleting columns that are not required
del mlratingsFile['timestamp']
del mlratingsFile['imdbid']
del mlratingsFile['userid']
del mltagsFile['timestamp']
del mltagsFile['userid']
del mltagsFile['tagid']

# creating a dictionary with movieid as key and a list of all tags associated with tha movie and removing duplicates
movieTagDict = {k: g['tag'].tolist() for k,g in mltagsFile.groupby('moviename')}
movieTagDict = {k:list(set(j)) for k,j in movieTagDict.items()}

# creating a dictionary with movieid as key and a list of all ratings given by a user for that particular movie and removing duplicates
movieRatingDict = {k: g['rating'].tolist() for k,g in mlratingsFile.groupby('moviename')}
movieRatingDict = {k:list(set(j)) for k,j in movieRatingDict.items()}

# computing the average rating for all movies and storing in a dictionary
avgRating = mlratingsFile.groupby('moviename').mean().reset_index()
avgRatingDict = {k: g['rating'].tolist() for k,g in avgRating.groupby('moviename')}

# List of unique movies, tags and ratings
movieList = mlratingsFile.moviename.unique()
movieList = np.asarray(movieList)
movieListDict = dict(enumerate(movieList))
tagList = mltagsFile.tag.unique()
tagList = np.asarray(tagList)
tagListDict = dict(enumerate(tagList))
ratingList = mlratingsFile.rating.unique()
ratingList = np.asarray(ratingList)
ratingListDict = dict(enumerate(ratingList))

movieListDictInverse = invertDictionary(movieListDict)
tagListDictInverse = invertDictionary(tagListDict)
ratingListDictInverse = invertDictionary(ratingListDict)

# declaring a tensor with three modes - with movie, tags and ratings
T = np.zeros((movieList.shape[0], tagList.shape[0], ratingList.shape[0]))
arrayofvalues = []

for i in movieList:
    if i in movieRatingDict:
        if i in movieTagDict:
            movieTags = movieTagDict[i]
            rList = movieRatingDict[i]
            for j in movieTags:
                for k in rList:
                    mIndex = movieListDictInverse[i]
                    tIndex = tagListDictInverse[j]
                    rIndex = ratingListDictInverse[k]
                    avgRatingValue = avgRatingDict[i][0]
                    if k >= avgRatingValue:
                        T[mIndex, tIndex, rIndex] = 1
                        arrayofvalues.append([mIndex,tIndex,rIndex])
                    else:
                        T[mIndex, tIndex, rIndex] = 0

"""print arrayofvalues
print T"""

# Task 2d - Part 2

# building the tensor using sktensor
tensor = dtensor(T)

# applying CP-decomposition with ALS(Alternating Least Squares)
U, fit, itr, exectimes = cp_als(tensor, 5, init='random')
print U


# Task 2d - Part 3

# Latent Semantics for Movies
latent_semantics_movie = pd.DataFrame(columns=['movie', 'ls1', 'ls2', 'ls3', 'ls4', 'ls5'])
latent_semantics_movie['movie'] = movieList
latent_semantics_movie['ls1'] = U[0][:, 0]
latent_semantics_movie['ls2'] = U[0][:, 1]
latent_semantics_movie['ls3'] = U[0][:, 2]
latent_semantics_movie['ls4'] = U[0][:, 3]
latent_semantics_movie['ls5'] = U[0][:, 4]

print 'Latent semantic for movie sorted by LS1'
ls1 = latent_semantics_movie.sort_values(by='ls1', ascending=False)
print ls1
print 'Latent semantic for movie sorted by LS2'
ls2 = latent_semantics_movie.sort_values(by='ls2', ascending=False)
print ls2
print 'Latent semantic for movie sorted by LS3'
ls3 = latent_semantics_movie.sort_values(by='ls3', ascending=False)
print ls3
print 'Latent semantic for movie sorted by LS4'
ls4 = latent_semantics_movie.sort_values(by='ls4', ascending=False)
print ls4
print 'Latent semantic for movie sorted by LS5'
ls5 = latent_semantics_movie.sort_values(by='ls5', ascending=False)
print ls5

# Latent Semantics for Tag
latent_semantics_tag = pd.DataFrame(columns=['tag', 'ls1', 'ls2', 'ls3', 'ls4', 'ls5'])
latent_semantics_tag['tag'] = tagList
latent_semantics_tag['ls1'] = U[1][:, 0]
latent_semantics_tag['ls2'] = U[1][:, 1]
latent_semantics_tag['ls3'] = U[1][:, 2]
latent_semantics_tag['ls4'] = U[1][:, 3]
latent_semantics_tag['ls5'] = U[1][:, 4]

print 'Latent semantic for tag sorted by LS1'
ls1 = latent_semantics_tag.sort_values(by='ls1', ascending=False)
print ls1
print 'Latent semantic for tag sorted by LS2'
ls2 = latent_semantics_tag.sort_values(by='ls2', ascending=False)
print ls2
print 'Latent semantic for tag sorted by LS3'
ls3 = latent_semantics_tag.sort_values(by='ls3', ascending=False)
print ls3
print 'Latent semantic for tag sorted by LS4'
ls4 = latent_semantics_tag.sort_values(by='ls4', ascending=False)
print ls4
print 'Latent semantic for tag sorted by LS5'
ls5 = latent_semantics_tag.sort_values(by='ls5', ascending=False)
print ls5

# Latent Semantics for Rating
latent_semantics_rating = pd.DataFrame(columns=['rating', 'ls1', 'ls2', 'ls3', 'ls4', 'ls5'])
latent_semantics_rating['rating'] = ratingList
latent_semantics_rating['ls1'] = U[2][:, 0]
latent_semantics_rating['ls2'] = U[2][:, 1]
latent_semantics_rating['ls3'] = U[2][:, 2]
latent_semantics_rating['ls4'] = U[2][:, 3]
latent_semantics_rating['ls5'] = U[2][:, 4]

print 'Latent semantic for rating sorted by LS1'
ls1 = latent_semantics_rating.sort_values(by='ls1', ascending=False)
print ls1
print 'Latent semantic for rating sorted by LS2'
ls2 = latent_semantics_rating.sort_values(by='ls2', ascending=False)
print ls2
print 'Latent semantic for rating sorted by LS3'
ls3 = latent_semantics_rating.sort_values(by='ls3', ascending=False)
print ls3
print 'Latent semantic for rating sorted by LS4'
ls4 = latent_semantics_rating.sort_values(by='ls4', ascending=False)
print ls4
print 'Latent semantic for rating sorted by LS5'
ls5 = latent_semantics_rating.sort_values(by='ls5', ascending=False)
print ls5


# Task 2d - Part 4

# Clusters for movie
kmeans_movie = KMeans(n_clusters=5, random_state=0).fit(U[0])

c1 = []
c2 = []
c3 = []
c4 = []
c5 = []

for i in range(0, len(kmeans_movie.labels_) - 1):
    if kmeans_movie.labels_[i] == 0:
        c1.append(movieList[i])
    elif kmeans_movie.labels_[i] == 1:
        c2.append(movieList[i])
    elif kmeans_movie.labels_[i] == 2:
        c3.append(movieList[i])
    elif kmeans_movie.labels_[i] == 3:
        c4.append(movieList[i])
    elif kmeans_movie.labels_[i] == 4:
        c5.append(movieList[i])

print "Movie clusters -->"
print 'Cluster 1:'
print c1
print 'Cluster 2:'
print c2
print 'Cluster 3:'
print c3
print 'Cluster 4:'
print c4
print 'Cluster 5:'
print c5

# Clusters for tags
kmeans_tag = KMeans(n_clusters=5, random_state=0).fit(U[1])

c1 = []
c2 = []
c3 = []
c4 = []
c5 = []

for i in range(0, len(kmeans_tag.labels_) - 1):
    if kmeans_tag.labels_[i] == 0:
        c1.append(tagList[i])
    elif kmeans_tag.labels_[i] == 1:
        c2.append(tagList[i])
    elif kmeans_tag.labels_[i] == 2:
        c3.append(tagList[i])
    elif kmeans_tag.labels_[i] == 3:
        c4.append(tagList[i])
    elif kmeans_tag.labels_[i] == 4:
        c5.append(tagList[i])

print "Tag Clusters -->"
print 'Cluster 1:'
print c1
print 'Cluster 2:'
print c2
print 'Cluster 3:'
print c3
print 'Cluster 4:'
print c4
print 'Cluster 5:'
print c5

# Clusters for rating
kmeans_rating = KMeans(n_clusters=5, random_state=0).fit(U[2])

c1 = []
c2 = []
c3 = []
c4 = []
c5 = []

for i in range(0, len(kmeans_rating.labels_) - 1):
    if kmeans_rating.labels_[i] == 0:
        c1.append(ratingList[i])
    elif kmeans_rating.labels_[i] == 1:
        c2.append(ratingList[i])
    elif kmeans_rating.labels_[i] == 2:
        c3.append(ratingList[i])
    elif kmeans_rating.labels_[i] == 3:
        c4.append(ratingList[i])
    elif kmeans_rating.labels_[i] == 4:
        c5.append(ratingList[i])

print "Rating Clusters -->"
print 'Cluster 1:'
print c1
print 'Cluster 2:'
print c2
print 'Cluster 3:'
print c3
print 'Cluster 4:'
print c4
print 'Cluster 5:'
print c5