import datetime as dt
import math
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from sktensor import dtensor, cp_als


# invert indices and values of movieid, tagids for faster access
def invertDictionary(dt):
    return dict([[v, k] for k, v in dt.items()])


def task1cFunc(userid):
    # read mltags, mlrating, mlmovies, movie-actor
    mltagsFile = pd.read_csv('mltags.csv')
    mlratingsFile = pd.read_csv('mlratings.csv')
    genomeFile = pd.read_csv('genome-tags.csv')
    movieFile = pd.read_csv('smallmlmovies.csv')

    # Exgtract tag from tagid
    genomeFile['tagid'] = genomeFile['tagId']
    del genomeFile['tagId']
    mltagsFile = pd.merge(mltagsFile, genomeFile, on='tagid')

    s = movieFile["genres"].str.split('|', expand=True).stack()
    i = s.index.get_level_values(0)
    movieFile = movieFile.loc[i].copy()
    movieFile["genres"] = s.values

    # Extract movie from movieid
    del movieFile['year']
    mlratingsFile = pd.merge(mlratingsFile, movieFile, on='movieid')
    mltagsFile = pd.merge(mltagsFile, movieFile, on='movieid')

    mltagsFileUser = mltagsFile.loc[mltagsFile['userid'] == userid]
    tagUserMovies = mltagsFileUser['moviename'].values
    mlratingsFileUser = mlratingsFile.loc[mlratingsFile['userid'] == userid]
    ratingUserMovies = mlratingsFileUser['moviename'].values
    tagRatingUserMovies = list(set(tagUserMovies) | set(ratingUserMovies))

    mltagsFileUser['timestamp'] = pd.to_datetime(mltagsFileUser['timestamp'])
    mltagsFileUser['timestamp'] = (mltagsFileUser['timestamp'] - dt.datetime(1970, 1, 1)).dt.total_seconds()
    mltagsFileUser['timestamp'] = \
        ((mltagsFileUser['timestamp'] - mltagsFileUser['timestamp'].min()) / (mltagsFileUser['timestamp'].max() - mltagsFileUser['timestamp'].min()+1))+1

    mlratingsFileUser['timestamp'] = pd.to_datetime(mlratingsFileUser['timestamp'])
    mlratingsFileUser['timestamp'] = (mlratingsFileUser['timestamp'] - dt.datetime(1970, 1, 1)).dt.total_seconds()
    mlratingsFileUser['timestamp'] = \
        ((mlratingsFileUser['timestamp'] - mlratingsFileUser['timestamp'].min()) / (mlratingsFileUser['timestamp'].max() - mlratingsFileUser['timestamp'].min()+1))+1

    commonTagRating = list(set(tagUserMovies) & set(ratingUserMovies))
    uncommonTag = list(set(tagUserMovies) ^ set(commonTagRating))
    uncommonRating = list(set(ratingUserMovies) ^ set(commonTagRating))

    timeWeights = {}
    for i in range(len(commonTagRating)):
        tag = mltagsFileUser.loc[mltagsFileUser['moviename'] == commonTagRating[i]]['timestamp'].values[0]
        rating = mlratingsFileUser.loc[mlratingsFileUser['moviename'] == commonTagRating[i]]['timestamp'].values[0]
        if tag > rating:
            timeWeights[commonTagRating[i]] = tag
        else:
            timeWeights[commonTagRating[i]] = rating

    for i in range(len(uncommonRating)):
        rating = mlratingsFileUser.loc[mlratingsFileUser['moviename'] == uncommonRating[i]]['timestamp'].values[0]
        timeWeights[uncommonRating[i]] = rating

    for i in range(len(uncommonTag)):
        tag = mltagsFileUser.loc[mltagsFileUser['moviename'] == uncommonTag[i]]['timestamp'].values[0]
        timeWeights[uncommonTag[i]] = tag
    #
    # deleting columns that are not required
    del mlratingsFile['timestamp']
    del mlratingsFile['imdbid']
    del mlratingsFile['userid']
    del mltagsFile['timestamp']
    del mltagsFile['userid']
    del mltagsFile['tagid']

    # creating a dictionary with movieid as key and a list of all tags associated with tha movie and removing duplicates
    movieGenreDict = {k: g['genres'].tolist() for k, g in movieFile.groupby('moviename')}
    movieGenreDict = {k: list(set(j)) for k, j in movieGenreDict.items()}

    # creating a dictionary with movieid as key and a list of all ratings given by a user for that particular movie and removing duplicates
    movieRatingDict = {k: g['rating'].tolist() for k, g in mlratingsFile.groupby('moviename')}
    movieRatingDict = {k: list(set(j)) for k, j in movieRatingDict.items()}

    # computing the average rating for all movies and storing in a dictionary
    avgRating = mlratingsFile.groupby('moviename').mean().reset_index()
    avgRatingDict = {k: g['rating'].tolist() for k, g in avgRating.groupby('moviename')}

    # List of unique movies, genres and ratings
    movieList = mlratingsFile.moviename.unique()
    movieList = np.asarray(movieList)
    movieListDict = dict(enumerate(movieList))
    genreList = movieFile.genres.unique()
    genreList = np.asarray(genreList)
    genreListDict = dict(enumerate(genreList))
    ratingList = mlratingsFile.rating.unique()
    ratingList = np.asarray(ratingList)
    ratingListDict = dict(enumerate(ratingList))

    movieListDictInverse = invertDictionary(movieListDict)
    genreListDictInverse = invertDictionary(genreListDict)
    ratingListDictInverse = invertDictionary(ratingListDict)

    movieNotWatched = list(set(movieList) ^ set(tagRatingUserMovies))

    # declaring a tensor with three modes - with movie, tags and ratings
    T = np.zeros((movieList.shape[0], genreList.shape[0], ratingList.shape[0]))
    arrayofvalues = []

    for i in movieList:
        if i in movieRatingDict:
            if i in movieGenreDict:
                movieTags = movieGenreDict[i]
                rList = movieRatingDict[i]
                for j in movieTags:
                    for k in rList:
                        mIndex = movieListDictInverse[i]
                        gIndex = genreListDictInverse[j]
                        rIndex = ratingListDictInverse[k]
                        avgRatingValue = avgRatingDict[i][0]
                        if k >= avgRatingValue:
                            T[mIndex, gIndex, rIndex] = 1
                            arrayofvalues.append([mIndex, gIndex, rIndex])
                        else:
                            T[mIndex, gIndex, rIndex] = 0

    # building the tensor using sktensor
    tensor = dtensor(T)

    # applying CP-decomposition with ALS(Alternating Least Squares)
    U, fit, itr, exectimes, P = cp_als(tensor, 5, init='random')

    latent_semantics_movie = pd.DataFrame(columns=['movie', 'ls1', 'ls2', 'ls3', 'ls4', 'ls5'])
    latent_semantics_movie['movie'] = movieList
    latent_semantics_movie['ls1'] = U[0][:, 0]
    latent_semantics_movie['ls2'] = U[0][:, 1]
    latent_semantics_movie['ls3'] = U[0][:, 2]
    latent_semantics_movie['ls4'] = U[0][:, 3]
    latent_semantics_movie['ls5'] = U[0][:, 4]

    x = latent_semantics_movie.loc[latent_semantics_movie['movie'].isin(tagRatingUserMovies)].values
    for i in range(len(x)):
        for j in range(1, len(x[0])):
            x[i][j] = x[i][j]*timeWeights.get(x[i][0])
    y = latent_semantics_movie.loc[latent_semantics_movie['movie'].isin(movieNotWatched)].values

    cossim = cosine_similarity(x[:, 1:], y[:, 1:])
    simDF = pd.DataFrame(cossim, index=tagRatingUserMovies, columns=movieNotWatched)
    simDF.to_csv('cos.csv')

    temp = simDF.values.tolist()
    sorted_movies_for_each_watched_movieDict = []
    for i in range(len(temp)):
        sorted_movies_for_each_watched_movie = np.argsort(temp[i])
        sorted_movies_for_each_watched_movieDict.append(sorted_movies_for_each_watched_movie.tolist()[:10])

    sortedMoviesRavel = [item for sublist in sorted_movies_for_each_watched_movieDict for item in sublist]
    freq = {}
    for i in range(len(sorted_movies_for_each_watched_movieDict)):
        for j in range(len(sorted_movies_for_each_watched_movieDict[0])):
            freq[sorted_movies_for_each_watched_movieDict[i][j]] = 0

    for i in range(len(sorted_movies_for_each_watched_movieDict)):
        for j in range(len(sorted_movies_for_each_watched_movieDict[0])):
            freq[sorted_movies_for_each_watched_movieDict[i][j]] += (10-j)

    freq = OrderedDict(sorted(freq.items(), reverse=True, key=lambda t: t[1]))
    freq = freq.items()

    recommendedMovies = []
    for i in range(5):
        index = freq[i][0]
        recommendedMovies.append(y[index][0])

    relevant = []
    notRelevant = []

    choice = 'y'
    while choice != 'n':
        rel_dict = {}
        selected_dict = {}
        N = 5
        R = 0
        for i in range(len(recommendedMovies)):
            print "If ", recommendedMovies[i], " is relevant, enter 1. If it is not relevant, enter 0"
            relevant.append(int(raw_input()))
            rel_dict[recommendedMovies[i]] = relevant[i]
            if relevant[i] == 1:
                R = R + 1
            else:
                notRelevant.append(recommendedMovies[i])

        genreset = set()
        for movie in recommendedMovies:
            genres_list = movieGenreDict[movie]
            selected_dict[movie] = genres_list
            genreset = genreset.union(set(genres_list))

        genreTop5 = list(genreset)
        ri = []
        ni = []
        for i in range(0, len(genreTop5)):
            ri.append(0)
            ni.append(0)
        for m in recommendedMovies:
            for i in range(0, len(genreTop5)):
                l1 = selected_dict[m]
                rval = rel_dict[m]
                if genreTop5[i] in l1:
                    ni[i] = ni[i] + 1
                    if rval == 1:
                        ri[i] = ri[i] + 1

        pr_feedback = {}

        for i in range(0, len(genreTop5)):
            try:
                numerator = ri[i] / (R - ri[i])
                denominator = (ni[i] - ri[i]) / (N - R - ni[i] + ri[i])
                pr = math.log((numerator / denominator), 2)
            except:
                numerator = (ri[i] + 0.5) / (R - ri[i] + 1)
                denominator = (ni[i] - ri[i] + 0.5) / (N - R - ni[i] + ri[i] + 1)
                pr = math.log((numerator / denominator), 2)

            pr_feedback[genreTop5[i]] = pr

        for key, value in pr_feedback.iteritems():
            pr_feedback[key] = (pr_feedback[key] - min(pr_feedback.values())) / max(pr_feedback.values())

        pr_dict = {}
        for i in movieList:
            if i in movieRatingDict:
                if i in movieGenreDict:
                    movieTags = movieGenreDict[i]
                    rList = movieRatingDict[i]
                    for j in movieTags:
                        for k in rList:
                            mIndex = movieListDictInverse[i]
                            tIndex = genreListDictInverse[j]
                            rIndex = ratingListDictInverse[k]
                            avgRatingValue = avgRatingDict[i][0]
                            if k >= avgRatingValue:
                                if j in genreTop5:
                                    T[mIndex, tIndex, rIndex] *= pr_feedback[j]

        tensor = dtensor(T)

        # applying CP-decomposition with ALS(Alternating Least Squares)
        U, fit, itr, exectimes, P = cp_als(tensor, 5, init='random')

        latent_semantics_movie = pd.DataFrame(columns=['movie', 'ls1', 'ls2', 'ls3', 'ls4', 'ls5'])
        latent_semantics_movie['movie'] = movieList
        latent_semantics_movie['ls1'] = U[0][:, 0]
        latent_semantics_movie['ls2'] = U[0][:, 1]
        latent_semantics_movie['ls3'] = U[0][:, 2]
        latent_semantics_movie['ls4'] = U[0][:, 3]
        latent_semantics_movie['ls5'] = U[0][:, 4]

        x = latent_semantics_movie.loc[latent_semantics_movie['movie'].isin(tagRatingUserMovies)].values
        for i in range(len(x)):
            for j in range(1, len(x[0])):
                x[i][j] = x[i][j]*timeWeights.get(x[i][0])
        y = latent_semantics_movie.loc[latent_semantics_movie['movie'].isin(movieNotWatched)].values
        cossim = cosine_similarity(x[:, 1:], y[:, 1:])
        simDF = pd.DataFrame(cossim, index=tagRatingUserMovies, columns=movieNotWatched)

        temp = simDF.values.tolist()
        sorted_movies_for_each_watched_movieDict = []
        for i in range(len(temp)):
            sorted_movies_for_each_watched_movie = np.argsort(temp[i])
            sorted_movies_for_each_watched_movieDict.append(sorted_movies_for_each_watched_movie.tolist()[:10])

        sortedMoviesRavel = [item for sublist in sorted_movies_for_each_watched_movieDict for item in sublist]
        freq = {}
        for i in range(len(sorted_movies_for_each_watched_movieDict)):
            for j in range(len(sorted_movies_for_each_watched_movieDict[0])):
                freq[sorted_movies_for_each_watched_movieDict[i][j]] = 0

        for i in range(len(sorted_movies_for_each_watched_movieDict)):
            for j in range(len(sorted_movies_for_each_watched_movieDict[0])):
                freq[sorted_movies_for_each_watched_movieDict[i][j]] += (10-j)

        freq = OrderedDict(sorted(freq.items(), reverse=True, key=lambda t: t[1]))
        freq = freq.items()

        recommendedMovies = []
        for i in range(5):
            index = freq[i][0]
            recommendedMovies.append(y[index][0])
        print recommendedMovies
        relevant = []

        print('Do you want to continue? Enter Y for yes and N for No')
        choice = raw_input()
        while choice not in ['y', 'n']:
            print('invalid input')
            choice = input()


print "Enter a user id: "
x = raw_input()
task1cFunc(int(x))
