import pandas as pd
import numpy as np

def precision_at_k(true_items, predicted_items, k=5):
	if len(predicted_items) == 0:
		return 0.0
	pred_k = predicted_items[:k]
	return len(set(pred_k) & set(true_items)) / float(k)

def recall_at_k(true_items, predicted_items, k=5):
	if len(true_items) == 0:
		return 0.0
	pred_k = predicted_items[:k]
	return len(set(pred_k) & set(true_items)) / float(len(true_items))

def evaluate_model(recommender, interactions_df, users_sample=None, k=5):
	# interactions_df expected to have columns: user_id, product_id, weight
	users = users_sample if users_sample is not None else interactions_df['user_id'].unique()[:100]
	precisions = []
	recalls = []
	for u in users:
		# Build ground truth as items with weight >= transaction threshold
		user_items = interactions_df[interactions_df['user_id'] == u]['product_id'].unique().tolist()
		preds = recommender.recommend(u, top_k=k)
		precisions.append(precision_at_k(user_items, preds, k))
		recalls.append(recall_at_k(user_items, preds, k))
	return {
		'precision': float(np.mean(precisions)) if precisions else 0.0,
		'recall': float(np.mean(recalls)) if recalls else 0.0,
		'n_users': len(users)
	}

