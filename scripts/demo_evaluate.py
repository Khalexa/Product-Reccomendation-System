# Evaluation script: train and test recommender on sample data
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sample_data_loader import load_events
from sample_recommender import RecommenderSystem
from metrics import evaluate_model

if __name__ == '__main__':
    # Load and train
    events = load_events(sample_frac=0.5, max_users=200, max_items=200, nrows=10000)
    print('Events loaded:', len(events))
    model = RecommenderSystem()
    model.train(events)
    # Evaluate on sample users
    stats = evaluate_model(model, events, users_sample=events['user_id'].unique()[:50].tolist(), k=5)
    print('Evaluation stats:', stats)
