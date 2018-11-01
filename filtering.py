import numpy as np
import pandas as pd

names = ['userId', 'movieId', 'rating', 'timestamp']
df = pd.read_csv('movie_ratings.csv', sep='\t', names=names)
df.head()

n_users = df.userId.unique().shape[0]
n_items = df.movieId.unique().shape[0]

print(str(n_users) + ' users')
print(str(n_items) + 'items')

# TODO: separate the U_ratings,
ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1] - 1, row[2]-1] = row[3]

def cosign_similarity(user_ratings, kind='user', epsilon=1e-9):
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])

    return sim / norms / norms.T

%timeit cosign_similarity(train, kind='user')
