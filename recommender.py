# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Code based on https://pub.towardsai.net/recommendation-system-in-depth-tutorial-with-python-for-netflix-using-collaborative-filtering-533ff8a0e444

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
# -

# # Settings

RATINGS_FILE = "movie_dataset/movielens_movie_ratings.csv"
TITLES_FILE = "movie_dataset/movielens_movie_titles.csv"

# # Data

# ## Retrieve data files

ratings = pd.read_csv(RATINGS_FILE, sep="|")
ratings.head(2)

titles = pd.read_csv(TITLES_FILE, sep="|")
titles.head(2)

# ## Sample

# + active=""
# ratings = ratings[:10000]
# -

# ## Drop duplicates

ratings = ratings.drop_duplicates(["userId", "movieId"])

titles = titles.drop_duplicates(["movieId"])

# ## Split train test

# TODO: Make the split more intelligent

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=31)

# ## Analysis

# ### Number of ratings

plt.figure(figsize = (12, 8))
ax = sns.countplot(x="rating", data=train_data)
ax.set_yticklabels([num for num in ax.get_yticks()])
plt.tick_params(labelsize = 15)
plt.title("Count Ratings in train data", fontsize = 20)
plt.xlabel("Ratings", fontsize = 20)
plt.ylabel("Number of Ratings", fontsize = 20)
plt.show()

plt.figure(figsize = (12, 8))
ax = sns.countplot(x="rating", data=test_data)
ax.set_yticklabels([num for num in ax.get_yticks()])
plt.tick_params(labelsize = 15)
plt.title("Count Ratings in train data", fontsize = 20)
plt.xlabel("Ratings", fontsize = 20)
plt.ylabel("Number of Ratings", fontsize = 20)
plt.show()

# ### Number of rated movies per user

train_counts = train_data["userId"].value_counts()
test_counts = test_data["userId"].value_counts()

counts = pd.DataFrame({"train_counts": train_counts,
                       "test_counts": test_counts})
counts = counts.fillna(0).astype(int)
counts.head()


# # Recommender

# ## User-item sparse matrix

def get_user_item_sparse_matrix(df):
    sparse_data = sparse.csr_matrix((df["rating"], (df["userId"], df["movieId"])))
    return sparse_data


train_sparse_data = get_user_item_sparse_matrix(train_data)

test_sparse_data = get_user_item_sparse_matrix(test_data)

# ## Global average rating

global_average_rating = train_sparse_data.sum()/train_sparse_data.count_nonzero()
print("Global Average Rating: {}".format(global_average_rating))


# ## Cold start

# ### Average rating

def get_average_rating(sparse_matrix, is_user):
    ax = 1 if is_user else 0
    sum_of_ratings = sparse_matrix.sum(axis = ax).A1  
    no_of_ratings = (sparse_matrix != 0).sum(axis = ax).A1 
    rows, cols = sparse_matrix.shape
    average_ratings = {i: sum_of_ratings[i]/no_of_ratings[i] for i in range(rows if is_user else cols) if no_of_ratings[i] != 0}
    return average_ratings


average_rating_user = get_average_rating(train_sparse_data, True)

avg_rating_movie = get_average_rating(train_sparse_data, False)

# +
total_users = len(np.unique(ratings["userId"]))
train_users = len(average_rating_user)
uncommonUsers = total_users - train_users
                  
print(f"Total no. of Users = {total_users}")
print(f"No. of Users in train data = {train_users}")
print(f"No. of Users not present in train data = {uncommonUsers} ({np.round((uncommonUsers/total_users)*100, 2)}%)")


# -

# # Find similar movies

def find_similar_movie_ids(sparse_matrix, movie_id):
    similarity = cosine_similarity(sparse_matrix.T, dense_output = False)
    similarities = similarity[movie_id]
    return list(reversed(similarities.todense().argsort().tolist()[0][-10:]))


similar_ids = find_similar_movie_ids(train_sparse_data, 8844)

similar_movies = list()
for movie_id in similar_ids:
    similar_movies = similar_movies + titles[titles["movieId"] == movie_id]["title"].tolist()
similar_movies
