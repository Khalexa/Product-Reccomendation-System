# Prewarm recommendation cache for top users
# Usage: python scripts/prewarm_cache.py --top 50 --k 6
import argparse
import os
import sys
import sqlite3
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sample_data_loader import load_events, load_items
from sample_recommender import RecommenderSystem
import requests

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'models' / 'rec_cache.db'
THUMB_DIR = ROOT / 'static' / 'thumbs'
THUMB_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--top', type=int, default=50)
parser.add_argument('--k', type=int, default=6)
args = parser.parse_args()

# Load events and train
print('Loading events...')
events = load_events(sample_frac=1.0, max_users=1000, max_items=1000, nrows=20000)
print('Events:', len(events))
model = RecommenderSystem()
model.train(events)
print('Model trained')

# Pick top users by activity
top_users = events['user_id'].value_counts().nlargest(args.top).index.tolist()
print('Top users:', len(top_users))

# Ensure DB exists
if not DB_PATH.parent.exists():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

for uid in top_users:
    recs = model.recommend(uid, top_k=args.k)
    try:
        c.execute('REPLACE INTO rec_cache (user_id, top_k, recs, ts) VALUES (?,?,?,?)',
                  (int(uid), int(args.k), json.dumps(recs), time.time()))
        conn.commit()
    except Exception as e:
        print('DB insert failed for', uid, e)
    # Pre-download thumbnails for each recommended item
    for item in recs:
        img_path = THUMB_DIR / f"{item}.jpg"
        if img_path.exists():
            continue
        try:
            url = f"https://picsum.photos/seed/{item}/200/200"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(r.content)
        except Exception:
            pass

conn.close()
print('Prewarm complete.')
