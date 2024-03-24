from bivaecf.dataset import Dataset
import numpy as np
import pandas as pd
import os

# Define custom PyTorch dataset
def get_batch_similarity(batch, bivae, movie_map, user_map):
   movie_ids = np.array(batch[:,0], dtype=np.int) 
   user_ids = np.array(batch[:,1], dtype=np.int)  

   array_list = []
   for id in movie_ids:
      array_list.append(bivae.get_user_vectors()[movie_map[id]])
   batch_movie_ids = np.concatenate(array_list)
   batch_movie_ids = batch_movie_ids.reshape(-1, 50)  # -1 indicates automatic calculation of the batch size
   array_list = []
   for id in user_ids:
      array_list.append(bivae.get_item_vectors()[user_map[id]])
   batch_user_ids = np.concatenate(array_list)
   batch_user_ids = batch_user_ids.reshape(-1, 50)  # -1 indicates automatic calculation of the batch size
   comb = batch_movie_ids * batch_user_ids
   return comb

def PreProcessDataset(data_root: str) -> pd.DataFrame:
    movies_path = os.path.join(data_root, "movies.csv")
    ratings_path = os.path.join(data_root, "ratings.csv")
    movie_data = pd.read_csv(movies_path, index_col=["movieId"], header=0)
    ratings_data = pd.read_csv(ratings_path)
    merged_data = pd.merge(movie_data, ratings_data, on='movieId', how='inner')
    merged_data.drop('genres', axis=1, inplace=True)
    merged_data.drop(['timestamp', 'title'], axis=1, inplace=True)
    return merged_data

# Get unique values from column 'A' and sort them
def get_movie_mapping(data) -> dict:
    unique_values = data['movieId'].unique()
    sorted_values = sorted(unique_values)
    movie_map = dict()
    id = 0
    for i in sorted_values:
        movie_map[i] = id
        id += 1

# Get unique values from column 'A' and sort them
def get_user_mapping(data) -> dict:
    unique_values = data['userId'].unique()
    sorted_values = sorted(unique_values)
    user_map = dict()
    id = 0
    for i in sorted_values:
        user_map[i] = id
        id += 1

class MovieLensDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]