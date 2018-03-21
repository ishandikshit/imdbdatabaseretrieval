import differentiate_genre_3 as pdiff2
import differentiate_genre_2 as pdiff1
import differentiate_genre_1 as tfidfdiff
import print_actor_vector as actor
import print_genre_vector as genre
import print_user_vector as user

pdiff1.for_all_genres()
pdiff2.for_all_genres()
tfidfdiff.for_all_genres()
actor.for_all_actors('tf')
actor.for_all_actors('tfidf')
genre.for_all_genres('tf')
genre.for_all_genres('tfidf')
user.for_all_users('tf')
user.for_all_users('tfidf')
