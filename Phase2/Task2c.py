import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sktensor import dtensor, cp_als

# Read mlmovies CSV file
mlmovies_df = pd.read_csv("Data/mlmovies.csv", keep_default_na=False, na_values=[""])

# Read movie-actor CSV file
movieactor_df = pd.read_csv("Data/movie-actor.csv", keep_default_na=False, na_values=[""])

# Read imdb-actor-info CSV file
imdbactorinfo_df = pd.read_csv("Data/imdb-actor-info.csv", keep_default_na=False, na_values=[""])
imdbactorinfo_df['actorid'] = imdbactorinfo_df['id']
del imdbactorinfo_df['id']
del imdbactorinfo_df['gender']

# Merge the two csv files
merged = pd.merge(mlmovies_df, movieactor_df, on='movieid')
del merged['actor_movie_rank']
del merged['genres']
del merged['movieid']

merged = pd.merge(merged, imdbactorinfo_df, on='actorid')
del merged['actorid']

# convert dataframe to array
conv_arr = merged.values

# split matrix into 3 columns each into 1d array
actor = np.delete(conv_arr, [0, 1], axis=1)
movie = np.delete(conv_arr, [1, 2], axis=1)
year = np.delete(conv_arr, [0, 2], axis=1)

# converting into 1D array, removing duplicates and convert list to array
movie = np.asarray(list(set(movie.ravel())))
year = np.asarray(list(set(year.ravel())))
actor = np.asarray(list(set(actor.ravel())))

# Create 3d array representation of tensor
T = np.zeros((year.shape[0], movie.shape[0], actor.shape[0]))

for i in range(len(year)):
    for j in range(len(movie)):
        if ((merged['year'] == year[i]) & (merged['moviename'] == movie[j])).any():
            for k in range(len(actor)):
                if ((merged['moviename'] == movie[j]) & (merged['name'] == actor[k])).any():
                    T[i, j, k] = 1
                else:
                    T[i, j, k] = 0

tensor = dtensor(T)

# Decompose tensor using CP-ALS
U, fit, itr, exectimes = cp_als(tensor, 5, init='random')
print U

# Latent Semantics for Year
latent_semantics_year = pd.DataFrame(columns=['year', 'ls1', 'ls2', 'ls3', 'ls4', 'ls5'])
latent_semantics_year['year'] = year
latent_semantics_year['ls1'] = U[0][:, 0]
latent_semantics_year['ls2'] = U[0][:, 1]
latent_semantics_year['ls3'] = U[0][:, 2]
latent_semantics_year['ls4'] = U[0][:, 3]
latent_semantics_year['ls5'] = U[0][:, 4]

print 'Latent Semantic for Year sorted by LS1'
ls1 = latent_semantics_year.sort_values(by='ls1', ascending=False)
print ls1
print 'Latent Semantic for Year sorted by LS2'
ls2 = latent_semantics_year.sort_values(by='ls2', ascending=False)
print ls2
print 'Latent Semantic for Year sorted by LS3'
ls3 = latent_semantics_year.sort_values(by='ls3', ascending=False)
print ls3
print 'Latent Semantic for Year sorted by LS4'
ls4 = latent_semantics_year.sort_values(by='ls4', ascending=False)
print ls4
print 'Latent Semantic for Year sorted by LS5'
ls5 = latent_semantics_year.sort_values(by='ls5', ascending=False)
print ls5

# Latent Semantics for Movie
latent_semantics_movie = pd.DataFrame(columns=['movie', 'ls1', 'ls2', 'ls3', 'ls4', 'ls5'])
latent_semantics_movie['movie'] = movie
latent_semantics_movie['ls1'] = U[1][:, 0]
latent_semantics_movie['ls2'] = U[1][:, 1]
latent_semantics_movie['ls3'] = U[1][:, 2]
latent_semantics_movie['ls4'] = U[1][:, 3]
latent_semantics_movie['ls5'] = U[1][:, 4]

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

# Latent semantic for actor
latent_semantics_actor = pd.DataFrame(columns=['actor', 'ls1', 'ls2', 'ls3', 'ls4', 'ls5'])
latent_semantics_actor['actor'] = actor
latent_semantics_actor['ls1'] = U[2][:, 0]
latent_semantics_actor['ls2'] = U[2][:, 1]
latent_semantics_actor['ls3'] = U[2][:, 2]
latent_semantics_actor['ls4'] = U[2][:, 3]
latent_semantics_actor['ls5'] = U[2][:, 4]

print 'Latent semantic for actor sorted by LS1'
ls1 = latent_semantics_actor.sort_values(by='ls1', ascending=False)
print ls1
print 'Latent semantic for actor sorted by LS2'
ls2 = latent_semantics_actor.sort_values(by='ls2', ascending=False)
print ls2
print 'Latent semantic for actor sorted by LS3'
ls3 = latent_semantics_actor.sort_values(by='ls3', ascending=False)
print ls3
print 'Latent semantic for actor sorted by LS4'
ls4 = latent_semantics_actor.sort_values(by='ls4', ascending=False)
print ls4
print 'Latent semantic for actor sorted by LS5'
ls5 = latent_semantics_actor.sort_values(by='ls5', ascending=False)
print ls5

# Clusters for year
kmeans_year = KMeans(n_clusters=5, random_state=0).fit(U[0])

c1 = []
c2 = []
c3 = []
c4 = []
c5 = []

for i in range(0, len(kmeans_year.labels_) - 1):
    if kmeans_year.labels_[i] == 0:
        c1.append(year[i])
    elif kmeans_year.labels_[i] == 1:
        c2.append(year[i])
    elif kmeans_year.labels_[i] == 2:
        c3.append(year[i])
    elif kmeans_year.labels_[i] == 3:
        c4.append(year[i])
    elif kmeans_year.labels_[i] == 4:
        c5.append(year[i])

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

# Clusters for movie
kmeans_movie = KMeans(n_clusters=5, random_state=0).fit(U[1])

c1 = []
c2 = []
c3 = []
c4 = []
c5 = []

for i in range(0, len(kmeans_movie.labels_) - 1):
    if kmeans_movie.labels_[i] == 0:
        c1.append(movie[i])
    elif kmeans_movie.labels_[i] == 1:
        c2.append(movie[i])
    elif kmeans_movie.labels_[i] == 2:
        c3.append(movie[i])
    elif kmeans_movie.labels_[i] == 3:
        c4.append(movie[i])
    elif kmeans_movie.labels_[i] == 4:
        c5.append(movie[i])

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

# Clusters for actor
kmeans_actor = KMeans(n_clusters=5, random_state=0).fit(U[2])

c1 = []
c2 = []
c3 = []
c4 = []
c5 = []

for i in range(0, len(kmeans_actor.labels_) - 1):
    if kmeans_actor.labels_[i] == 0:
        c1.append(actor[i])
    elif kmeans_actor.labels_[i] == 1:
        c2.append(actor[i])
    elif kmeans_actor.labels_[i] == 2:
        c3.append(actor[i])
    elif kmeans_actor.labels_[i] == 3:
        c4.append(actor[i])
    elif kmeans_actor.labels_[i] == 4:
        c5.append(actor[i])

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
