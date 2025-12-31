import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from typing import Optional

class RecommenderSystem:
    def __init__(self):
        self.user_item_matrix = None
        self.user_sim_matrix = None  # This starts as None
        self.users = None
        self.items = None
        self.model_path: Optional[str] = None

    def train(self, interactions_df):
        # Create user-item interaction matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index="user_id",
            columns="product_id",
            values="weight",
            aggfunc="sum",
            fill_value=0
        )
        self.users = self.user_item_matrix.index.tolist()
        self.items = self.user_item_matrix.columns.tolist()

        # Calculate similarity matrices
        self.user_sim_matrix = cosine_similarity(self.user_item_matrix)
        self.item_sim_matrix = cosine_similarity(self.user_item_matrix.T)
        
        # Remove self-similarity from diagonal
        np.fill_diagonal(self.user_sim_matrix, 0)

    def save(self, path: str):
        """Save the trained model to disk."""
        payload = {
            'user_item_matrix': self.user_item_matrix,
            'user_sim_matrix': self.user_sim_matrix,
            'users': self.users,
            'items': self.items,
            'item_sim_matrix': getattr(self, 'item_sim_matrix', None)
        }
        joblib.dump(payload, path)
        self.model_path = path

    def load(self, path: str):
        """Load a saved model from disk."""
        payload = joblib.load(path)
        self.user_item_matrix = payload['user_item_matrix']
        self.user_sim_matrix = payload['user_sim_matrix']
        self.users = payload['users']
        self.items = payload['items']
        self.item_sim_matrix = payload.get('item_sim_matrix', None)
        self.model_path = path

    def recommend(self, user_id, top_k=5):
        if self.user_item_matrix is None or self.user_sim_matrix is None:
            return []
            
        if user_id not in self.user_item_matrix.index:
            return []

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_similarities = self.user_sim_matrix[user_idx]

        # Find top similar users
        similar_users_idx = np.argsort(user_similarities)[::-1][:20]

        # Aggregate scores from similar users
        similarities = user_similarities[similar_users_idx]
        scores = np.array([self.user_item_matrix.iloc[idx].values for idx in similar_users_idx])
        recommended_scores = np.dot(similarities, scores) / (np.sum(similarities) + 1e-9)

        # Exclude items already interacted with
        already_interacted = self.user_item_matrix.loc[user_id].values > 0
        recommended_scores[already_interacted] = -1

        top_product_idx = np.argsort(recommended_scores)[::-1][:top_k]
        top_product_ids = self.user_item_matrix.columns[top_product_idx].tolist()
        
        return top_product_ids

    def recommend_for_session(self, session_item_ids, top_k=5):
        """Recommend items similar to ones user viewed in session."""
        if not hasattr(self, 'item_sim_matrix') or self.item_sim_matrix is None:
            return self.items[:top_k]

        item_index = {it: idx for idx, it in enumerate(self.items)}

        sim_sum = None
        count = 0
        for sid in session_item_ids:
            if sid in item_index:
                idx = item_index[sid]
                vec = self.item_sim_matrix[idx]
                if sim_sum is None:
                    sim_sum = vec.copy()
                else:
                    sim_sum += vec
                count += 1

        if sim_sum is None:
            return self.items[:top_k]

        sim_scores = sim_sum / max(1, count)

        # Exclude items already in session
        for sid in session_item_ids:
            if sid in item_index:
                sim_scores[item_index[sid]] = -1

        top_idx = np.argsort(sim_scores)[::-1][:top_k]
        return [self.items[i] for i in top_idx]

    def recommend_for_session_with_weights(self, session_item_weights, top_k=5):
        """Recommend items with weighted user interactions.
        
        Args:
            session_item_weights: dict mapping item_id -> weight
            top_k: number of recommendations
        """
        if not hasattr(self, 'item_sim_matrix') or self.item_sim_matrix is None:
            return self.items[:top_k]

        item_index = {it: idx for idx, it in enumerate(self.items)}

        sim_sum = None
        total_weight = 0.0
        for sid, w in session_item_weights.items():
            if sid in item_index and w > 0:
                idx = item_index[sid]
                vec = self.item_sim_matrix[idx] * float(w)
                if sim_sum is None:
                    sim_sum = vec.copy()
                else:
                    sim_sum += vec
                total_weight += float(w)

        if sim_sum is None:
            return self.items[:top_k]

        sim_scores = sim_sum / (total_weight + 1e-9)

        # Exclude items already interacted with
        for sid in session_item_weights.keys():
            if sid in item_index:
                sim_scores[item_index[sid]] = -1

        top_idx = np.argsort(sim_scores)[::-1][:top_k]
        return [self.items[i] for i in top_idx]