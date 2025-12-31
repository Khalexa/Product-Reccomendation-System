from typing import Optional
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RecommenderSystem:
    def __init__(self):
        self.user_item_matrix : Optional[pd.DataFrame]= None
        self.similarity_matrix = None
        self.items = None

    def train(self, interactions_df):
    
    #Train an item-based collaborative filtering model
    
    # Build user-item matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index="user_id",
            columns="product_id",
            values="weight",
            fill_value=0
        )

        assert self.user_item_matrix is not None

    # Item-item similarity (transpose!)
        self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)

    # Store product IDs in order
        self.items = self.user_item_matrix.columns.to_list()


    def recommend(self, user_id, top_k=5):
        if self.user_item_matrix is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        if user_id not in self.user_item_matrix.index:
            return []
        '''#Return top_k product recommendations for a user based on
    user-based collaborative filtering.
        '''

        # Cosine similarity between users
        user_sim = cosine_similarity(self.user_item_matrix)
        user_idx = self.user_item_matrix.index.get_loc(user_id)

        # Ignore self
        user_sim[user_idx, user_idx] = 0

        # Find top similar users
        top_n = 20
        similar_users_idx = np.argsort(user_sim[user_idx])[::-1][:top_n]

        # Aggregate products from top similar users using weighted average
        similarities = user_sim[user_idx, similar_users_idx]  # similarity scores of top similar users
        scores = np.array([self.user_item_matrix.iloc[idx].values for idx in similar_users_idx])
        recommended_scores = np.dot(similarities, scores) / (np.sum(similarities) + 1e-9)

        # Mask products the user already interacted with
        already_interacted = self.user_item_matrix.loc[user_id].values > 0
        recommended_scores[already_interacted] = -1

        # Get top product indices
        top_product_idx = np.argsort(recommended_scores)[::-1][:top_k]
        top_product_ids = self.user_item_matrix.columns[top_product_idx].tolist()

        return top_product_ids