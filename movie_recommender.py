import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]
    

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
df = pd.read_csv('movie_dataset.csv')

# printing the columns 
#print(df.columns)


##Step 2: Select Features
## feature means columns which we can use and also remove which we are not going to use 
features = ['keywords','cast','genres','director']



##Step 3: Create a column in DF which combines all selected features
# combine all the above feature in one large strings
# we are using try and except as we are getting error where we want strings in keyword and genre but nothing is given 
# its showing NaN so we have to remove it so 
# we iterate the above features and use fillna function to remove all NaN to enpty strings

for feature in features:
    df[feature] = df[feature].fillna(" ")

def combine_feature(row):
    try:
        return row['keywords']+" "+ row['cast']+" "+ row['genres']+" "+ row['director']  
    except:
        print("Error : ", row)

df['combined_features'] = df.apply(combine_feature,axis = 1)

print(df['combined_features'].head())

##Step 4: Create count matrix from this new combined column

cv = CountVectorizer()

count_matrix = cv.fit_transform(df['combined_features'])

##Step 5: Compute the Cosine Similarity based on the count_matrix
# cosine similarity is that how much is this movie is similar to the other given movies in term of features

cosine_sim = cosine_similarity(count_matrix)

movie_user_likes = input("Enter the movie that you want similar movies")
## Step 6: Get index of this movie from its title

movie_index  = get_index_from_title(movie_user_likes)
# Now we want to ge into cosine_sim matrix which has the cosine similarity score and take the values

similar_movie = list(enumerate(cosine_sim[movie_index]))

## Step 7: Get a list of similar movies in descending order of similarity score

sorted_similar_movie  = sorted(similar_movie,key =lambda x:x[1],reverse=True)


## Step 8: Print titles of first 50 movies

i = 0
for movie in sorted_similar_movie:
    print(get_title_from_index(movie[0]))
    i  += 1
    if i > 50:
        break