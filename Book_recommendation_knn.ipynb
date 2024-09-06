# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
# get data files
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

!unzip book-crossings.zip
books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'
# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})
df1=df_ratings.groupby(["isbn"]).count().reset_index()
hundred_ratings=df1.loc[df1["rating"]>=100]["isbn"]

hundred_ratings=df_books.loc[df_books["isbn"].isin(hundred_ratings)]
df2=df_ratings[["user","rating"]].groupby(["user"]).count().reset_index()

higher_users=df2.loc[df2["rating"]>=200]["user"]

df=df_ratings.loc[df_ratings["user"].isin(higher_users)]
df=df.loc[df["isbn"].isin(hundred_ratings["isbn"])]
df_pivot = df.pivot(
    index='isbn',
    columns='user',
    values='rating'
).fillna(0)

df_matrix = csr_matrix(df_pivot.values)
df_pivot.head()
model_knn = NearestNeighbors(metric = 'cosine')
model_knn.fit(df_matrix)
def get_recommends(book_title):

  recommended_books=[]
  book=hundred_ratings.loc[hundred_ratings["title"]==book_title]
  book_index=df_pivot.loc[df_pivot.index.isin(book["isbn"])]
  distances, indices = model_knn.kneighbors([x for x in book_index.values], n_neighbors = 6)

  distance =  distances[0][1:]
  indice = indices[0][1:]

  books=[df_books.loc[df_books["isbn"]==df_pivot.iloc[i].name]["title"].values[0] for i in indice]

  recommended_books = [list(z) for z in zip(books, distance)][::-1]
  return [book_title,recommended_books]
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)
def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2):
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()
