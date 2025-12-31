from sample_recommender import RecommenderSystem
from sample_data_loader import load_events


def test_train_and_recommend():
    df = load_events(sample_frac=0.2, max_users=10, max_items=20, nrows=2000)
    model = RecommenderSystem()
    model.train(df)
    users = df['user_id'].unique()
    if len(users) == 0:
        assert True
    else:
        uid = users[0]
        recs = model.recommend(uid, top_k=3)
        assert isinstance(recs, list)
*** End Patch