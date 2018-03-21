import differentiate_genre_1 as tfidf
import differentiate_genre_2 as pdiff1
import differentiate_genre_3 as pdiff2
import sys

genre1 = ''
genre2 = ''
model = ''

if len(sys.argv) > 2:
    genre1 = sys.argv[1]
    genre2 = sys.argv[2]
    model = sys.argv[3]


if genre1 != '' and genre2 != '':
    if model == 'tfidf':
        print 'Euclidean distance of genre vectors: ' + str(tfidf.get_difference_in_genres(genre1, genre2))
    elif model == 'pdiff1':
        print 'Euclidean distance of genre vectors: ' + str(pdiff1.p_distance(genre1, genre2))
    elif model == 'pdiff2':
        print 'Euclidean distance of genre vectors: ' + str(pdiff2.p_distance(genre1, genre2))