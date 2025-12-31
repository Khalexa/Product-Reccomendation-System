from flask import Flask, render_template, request, jsonify, send_file, session
from sample_data_loader import load_events, load_items
from sample_recommender import RecommenderSystem
import os
import pandas as pd
from collections import OrderedDict
import math
import requests
import sqlite3
import json
import time
from io import BytesIO
import threading

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret')

# Global loader flag: switch between sample and full dataset
USE_FULL_DATASET = False

def load_events_smart():
    """Load events from sample or full dataset based on global flag."""
    global USE_FULL_DATASET
    if USE_FULL_DATASET:
        try:
            from backend.data_loader import load_events as load_events_full
            return load_events_full()
        except Exception:
            print("Full dataset loader not available, falling back to sample")
            USE_FULL_DATASET = False
            return load_events()
    return load_events()

# Global data for the demo
events_df = load_events_smart()
model = RecommenderSystem()
MODEL_PATH = os.path.join('models', 'sample_model.joblib')
os.makedirs('models', exist_ok=True)
DB_PATH = os.path.join('models', 'rec_cache.db')

# Simple in-memory LRU cache for recommendations (define early so DB loader can use it)
REC_CACHE_MAX = 200
rec_cache = OrderedDict()
REC_CACHE_TTL = 24 * 3600  # seconds

# Initialize sqlite DB for persisted recommendation cache
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS rec_cache (
            user_id INTEGER,
            top_k INTEGER,
            recs TEXT,
            ts REAL,
            PRIMARY KEY (user_id, top_k)
        )
    ''')
    conn.commit()
    conn.close()

init_db()
# Load recent entries into in-memory cache
def load_db_cache(limit=REC_CACHE_MAX):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_id, top_k, recs, ts FROM rec_cache ORDER BY ts DESC LIMIT ?', (limit,))
    now = time.time()
    for user_id, top_k, recs, ts in c.fetchall():
        # skip expired entries
        try:
            if now - float(ts) > REC_CACHE_TTL:
                continue
        except Exception:
            continue
        key = f"{user_id}:{top_k}"
        try:
            rec_cache[key] = json.loads(recs)
        except Exception:
            continue
    conn.close()

load_db_cache()

# Try loading persisted model if available; otherwise train and save
if os.path.exists(MODEL_PATH):
    try:
        model.load(MODEL_PATH)
    except Exception:
        model.train(events_df)
        model.save(MODEL_PATH)
else:
    model.train(events_df)
    try:
        model.save(MODEL_PATH)
    except Exception:
        pass

# Simple in-memory LRU cache for recommendations
REC_CACHE_MAX = 200
rec_cache = OrderedDict()

def get_cached_recommendations(user_id, top_k=5):
    key = f"{user_id}:{top_k}"
    if key in rec_cache:
        # move to end (most recently used)
        rec_cache.move_to_end(key)
        return rec_cache[key]
    recs = model.recommend(user_id, top_k=top_k)
    rec_cache[key] = recs
    # persist to DB
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('REPLACE INTO rec_cache (user_id, top_k, recs, ts) VALUES (?,?,?,?)',
                  (user_id, top_k, json.dumps(recs), time.time()))
        conn.commit()
        conn.close()
    except Exception:
        pass
    if len(rec_cache) > REC_CACHE_MAX:
        rec_cache.popitem(last=False)
    return recs


@app.route('/thumb/<int:item_id>')
def thumb(item_id):
    """Serve a cached thumbnail, fetching from Picsum on first request."""
    static_dir = os.path.join('static', 'thumbs')
    os.makedirs(static_dir, exist_ok=True)
    img_path = os.path.join(static_dir, f"{item_id}.jpg")
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    # Fetch from picsum and cache
    try:
        url = f"https://picsum.photos/seed/{item_id}/200/200"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(r.content)
            return send_file(img_path, mimetype='image/jpeg')
    except Exception:
        pass
    # Fallback: return a 1x1 transparent
    return send_file(BytesIO(b"\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"), mimetype='image/gif')

@app.route('/')
def index():
    users = events_df['user_id'].unique()[:10].tolist()
    return render_template('index.html', users=users)

@app.route('/get_recommendations/<int:user_id>')
def get_rec(user_id):
    # Optional API key enforcement: set DEMO_API_KEY env var to require header 'X-API-Key'
    api_key = os.environ.get('DEMO_API_KEY')
    if api_key:
        provided = request.headers.get('X-API-Key') or request.args.get('api_key')
        if provided != api_key:
            return jsonify({'error': 'invalid_api_key'}), 401

    recs = get_cached_recommendations(user_id, top_k=6)
    items_info = load_items(relevant_product_ids=recs)
    
    result = []
    for r in recs:
        # Get a property name for the UI 
        details = items_info[items_info['itemid'] == r]
        if details.empty:
            name = f"Item {r}"
        else:
            if 'display_name' in details.columns and pd.notna(details.iloc[0]['display_name']):
                name = details.iloc[0]['display_name']
            elif 'value' in details.columns and pd.notna(details.iloc[0]['value']):
                name = details.iloc[0]['value']
            else:
                name = f"Item {r}"
        # Provide a thumbnail URL (picsum seed) for nicer UI without bundling images
        image_url = f"/thumb/{r}"
        result.append({"id": r, "display_name": name, "image_url": image_url})
        
    return jsonify(result)


@app.route('/cache_status/<int:user_id>')
def cache_status(user_id):
    """Check if user recommendations are cached."""
    top_k = request.args.get('k', default=6, type=int)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT recs FROM rec_cache WHERE user_id=? AND top_k=?', (user_id, top_k))
    row = c.fetchone()
    conn.close()
    return jsonify({'cached': bool(row)})


@app.route('/signin', methods=['POST'])
def signin():
    # create ephemeral session and store session item interactions
    session.clear()
    session['signed_in'] = True
    session['session_items'] = []
    session['session_events'] = {}  # item_id -> {'view':int,'click':int,'timestamps':[]}
    session['session_start_time'] = time.time()
    return jsonify({'status': 'signed_in'})


@app.route('/signout', methods=['POST'])
def signout():
    session.clear()
    return jsonify({'status': 'signed_out'})


@app.route('/session_status')
def session_status():
    # include session summary counts and per-item CTR
    signed = bool(session.get('signed_in', False))
    events = session.get('session_events', {})
    total_views = sum(v.get('view', 0) for v in events.values())
    total_clicks = sum(v.get('click', 0) for v in events.values())
    
    # Compute CTR per item
    per_item_ctr = {}
    for iid, ent in events.items():
        v = ent.get('view', 0)
        c = ent.get('click', 0)
        ctr = c / (v + c + 1e-9) if (v + c) > 0 else 0.0
        per_item_ctr[iid] = round(ctr, 3)
    
    return jsonify({
        'signed_in': signed,
        'total_views': total_views,
        'total_clicks': total_clicks,
        'session_items': session.get('session_items', []),
        'per_item_ctr': per_item_ctr
    })


@app.route('/session_event', methods=['POST'])
def session_event():
    data = request.get_json() or {}
    item_id = data.get('item_id')
    evt = data.get('event', 'view')
    if not session.get('signed_in'):
        return jsonify({'error': 'not_signed_in'}), 400
    events = session.get('session_events', {})
    try:
        iid = int(item_id)
    except Exception:
        return jsonify({'error': 'invalid_item'}), 400
    key = str(iid)
    ent = events.get(key, {'view': 0, 'click': 0, 'timestamps': []})
    if evt == 'click' or evt == 'addtocart' or evt == 'transaction':
        ent['click'] = ent.get('click', 0) + 1
    else:
        ent['view'] = ent.get('view', 0) + 1
    # Track timestamps for recency decay
    if 'timestamps' not in ent:
        ent['timestamps'] = []
    ent['timestamps'].append(time.time())
    # keep only recent 100 timestamps
    ent['timestamps'] = ent['timestamps'][-100:]
    events[key] = ent
    # update session_items list (unique recent)
    items = session.get('session_items', [])
    if iid not in items:
        items.append(iid)
    session['session_items'] = items[-50:]
    session['session_events'] = events
    return jsonify({'status': 'ok', 'session_items': session['session_items']})


@app.route('/get_session_recommendations')
def get_session_recommendations():
    if not session.get('signed_in'):
        return jsonify([])
    events = session.get('session_events', {})
    session_start = session.get('session_start_time', time.time())
    now = time.time()
    
    # Build weights: weight = (views + 2*clicks) * recency_decay
    # Recency decay: exp(-lambda * time_elapsed) where lambda=0.01 (slower decay)
    weights = {}
    for k, v in events.items():
        try:
            iid = int(k)
        except Exception:
            continue
        base_w = v.get('view', 0) + 2 * v.get('click', 0)
        # Use most recent timestamp for recency
        timestamps = v.get('timestamps', [])
        if timestamps:
            most_recent = max(timestamps)
            time_elapsed = now - most_recent
            decay_factor = math.exp(-0.01 * time_elapsed)
            w = base_w * decay_factor
        else:
            w = base_w
        if w > 0:
            weights[iid] = w
    recs = []
    try:
        if weights:
            recs = model.recommend_for_session_with_weights(weights, top_k=6)
        else:
            recs = model.recommend_for_session(session.get('session_items', []), top_k=6)
    except Exception:
        recs = []
    items_info = load_items(relevant_product_ids=recs)
    result = []
    for r in recs:
        details = items_info[items_info['itemid'] == r]
        if details.empty:
            name = f"Item {r}"
        else:
            if 'display_name' in details.columns and pd.notna(details.iloc[0]['display_name']):
                name = details.iloc[0]['display_name']
            elif 'value' in details.columns and pd.notna(details.iloc[0]['value']):
                name = details.iloc[0]['value']
            else:
                name = f"Item {r}"
        image_url = f"/thumb/{r}"
        result.append({"id": r, "display_name": name, "image_url": image_url})
    return jsonify(result)


@app.route('/refresh_recs/<int:user_id>', methods=['POST'])
def refresh_recs(user_id):
    """Recompute recommendations for user."""
    top_k = request.args.get('k', default=6, type=int)
    recs = model.recommend(user_id, top_k=top_k)
    # persist to DB
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('REPLACE INTO rec_cache (user_id, top_k, recs, ts) VALUES (?,?,?,?)',
                  (user_id, top_k, json.dumps(recs), time.time()))
        conn.commit()
        conn.close()
    except Exception:
        pass
    # update in-memory
    rec_cache[f"{user_id}:{top_k}"] = recs
    return jsonify({'recs': recs})


@app.route('/prewarm_top_users', methods=['POST'])
def prewarm_top_users():
    """Prewarm cache for top users."""
    n = request.args.get('n', default=50, type=int)
    k = request.args.get('k', default=6, type=int)
    # find top users
    users = events_df['user_id'].value_counts().nlargest(n).index.tolist()
    for u in users:
        recs = model.recommend(u, top_k=k)
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('REPLACE INTO rec_cache (user_id, top_k, recs, ts) VALUES (?,?,?,?)',
                      (int(u), int(k), json.dumps(recs), time.time()))
            conn.commit()
            conn.close()
        except Exception:
            pass
    return jsonify({'status': 'ok', 'n': len(users)})


@app.route('/switch_loader', methods=['POST'])
def switch_loader():
    """Toggle between sample and full dataset loaders."""
    global USE_FULL_DATASET, events_df
    toggle = request.get_json().get('use_full', False)
    USE_FULL_DATASET = bool(toggle)
    try:
        events_df = load_events_smart()
        # retrain model on new dataset
        model.train(events_df)
        return jsonify({'status': 'switched', 'use_full': USE_FULL_DATASET, 'events_count': len(events_df)})
    except Exception as e:
        USE_FULL_DATASET = not USE_FULL_DATASET
        return jsonify({'error': str(e)}), 500


@app.route('/loader_status')
def loader_status():
    """Return current loader status."""
    return jsonify({'use_full_dataset': USE_FULL_DATASET, 'events_count': len(events_df)})

if __name__ == '__main__':
    # Background prewarm scheduler
    def schedule_prewarm():
        """Prewarm cache every 24 hours."""
        import time
        while True:
            time.sleep(24 * 3600)
            try:
                users = events_df['user_id'].value_counts().nlargest(100).index.tolist()
                for u in users:
                    recs = model.recommend(u, top_k=6)
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute('REPLACE INTO rec_cache (user_id, top_k, recs, ts) VALUES (?,?,?,?)',
                                  (int(u), int(6), json.dumps(recs), time.time()))
                        conn.commit()
                        conn.close()
                    except Exception:
                        pass
                print('Background prewarm completed')
            except Exception as e:
                print('Background prewarm failed:', e)
    
    # Start background thread (daemon so it doesn't block shutdown)
    prewarm_thread = threading.Thread(target=schedule_prewarm, daemon=True)
    prewarm_thread.start()
    
    app.run(debug=True)